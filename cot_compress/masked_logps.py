"""策略 log-prob 在该臂 mask 下重归一化（v4 第 2 条）。
- think 阶段位置（第一个 </think> 之前，含 </think> 本身）：logp = z_t/T − logsumexp_{allowed}(z/T)；其余位置全词表。
- 臂 A / D / C_rand 的 allowed = 全词表（allowed_mask=None）。
- 预算强制写入的 close token 不是策略决策：从 completion_mask 里剔除（不进损失）。
- β>0 时参考模型的 logp 走同一个覆盖 → KL 的参考分布也在同一 mask 下重归一化。
"""
from __future__ import annotations
import torch

def think_position_masks(completion_ids: list[int], lift_id: int, force_start: int, close_ids: list[int]) -> tuple[list[bool], list[bool]]:
    """返回 (think_mask, forced_mask)。think：到第一个 </think>（含）；forced：预算强制的 close 段（无自然 </think> 且长度 ≥ force_start）。"""
    n = len(completion_ids)
    lift_pos = next((i for i, t in enumerate(completion_ids) if t == lift_id), None)
    natural = lift_pos is not None and lift_pos < force_start
    think = [(i <= lift_pos) if lift_pos is not None else True for i in range(n)]
    forced = [False] * n
    if not natural and n > force_start:
        for k, cid in enumerate(close_ids):
            i = force_start + k
            if i < n and completion_ids[i] == cid:
                forced[i] = True
            else:
                break
    return think, forced

def masked_selective_log_softmax(logits: torch.Tensor, ids: torch.Tensor, allowed_mask: torch.Tensor | None, think_mask: torch.Tensor) -> torch.Tensor:
    """logits [B,T,V]（已除温度），ids [B,T]，allowed_mask [V] bool 或 None，think_mask [B,T] bool → logps [B,T]。"""
    if allowed_mask is not None:
        block = (~allowed_mask)[None, None, :] & think_mask[:, :, None]
        logits = logits.masked_fill(block, float("-inf"))
    lse = torch.logsumexp(logits.float(), dim=-1)
    tgt = torch.gather(logits.float(), -1, ids.unsqueeze(-1)).squeeze(-1)
    return tgt - lse

def make_masked_trainer_class(Base, *, allowed_ids, vocab_size: int, lift_id: int, force_start: int, close_ids: list[int]):
    """返回 Base（TRL GRPOTrainer 或其 Unsloth 补丁版）的子类：覆盖 _get_per_token_logps_and_entropies 与 completion_mask。
    allowed_ids=None → 全词表（臂 A/D/C_rand）。镜像 trl 0.24/0.25 的实现（无 VLM 分支）。"""
    allowed_vec = None
    if allowed_ids is not None:
        allowed_vec = torch.zeros(vocab_size, dtype=torch.bool)
        allowed_vec[torch.tensor(sorted(i for i in allowed_ids if i < vocab_size), dtype=torch.long)] = True

    class MaskedTrainer(Base):
        def _think_mask_tensor(self, completion_ids: torch.Tensor) -> torch.Tensor:
            rows = [think_position_masks(r, lift_id, force_start, close_ids)[0] for r in completion_ids.tolist()]
            return torch.tensor(rows, dtype=torch.bool, device=completion_ids.device)

        def _forced_mask_tensor(self, completion_ids: torch.Tensor) -> torch.Tensor:
            rows = [think_position_masks(r, lift_id, force_start, close_ids)[1] for r in completion_ids.tolist()]
            return torch.tensor(rows, dtype=torch.bool, device=completion_ids.device)

        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None,
                                               compute_entropy=False, *extra, **kw):
            # trl 0.24 / Unsloth 编译版会把 pixel_values, image_grid_thw, num_images, ... 按位置传进来（VLM 分支），本任务无图像，忽略。
            batch_size = batch_size or input_ids.size(0)
            all_logps, all_ent = [], []
            am = allowed_vec.to(input_ids.device) if allowed_vec is not None else None
            for start in range(0, input_ids.size(0), batch_size):
                ib = input_ids[start:start + batch_size]; ab = attention_mask[start:start + batch_size]
                mi = {"input_ids": ib, "attention_mask": ab, "use_cache": False}
                if "logits_to_keep" in getattr(self, "model_kwarg_keys", ()):
                    mi["logits_to_keep"] = logits_to_keep + 1
                logits = model(**mi).logits[:, :-1, :][:, -logits_to_keep:, :] / self.temperature
                cids = ib[:, -logits_to_keep:]
                logps = masked_selective_log_softmax(logits, cids, am, self._think_mask_tensor(cids))
                all_logps.append(logps)
                if compute_entropy:
                    with torch.no_grad():
                        p = torch.softmax(logits.float(), -1); all_ent.append(-(p * torch.log(p + 1e-12)).sum(-1))
            return torch.cat(all_logps, 0), (torch.cat(all_ent, 0) if compute_entropy else None)

        def _generate_and_score_completions(self, inputs):
            out = super()._generate_and_score_completions(inputs)
            if isinstance(out, dict) and "completion_ids" in out and "completion_mask" in out:
                forced = self._forced_mask_tensor(out["completion_ids"])
                out["completion_mask"] = out["completion_mask"] * (~forced).to(out["completion_mask"].dtype)
            return out

    MaskedTrainer.__name__ = f"Masked{Base.__name__}"
    return MaskedTrainer
