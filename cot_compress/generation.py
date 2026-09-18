"""生成封装。两条路径：
  hook=generate : 本地路径。包装 HF model.generate，可插 logits_processor（臂 B 的 ThinkSpanLogitsProcessor）。
  hook=rollout  : 云端 vllm 路径。TRL `GRPOTrainer(rollout_func=...)`，**requires trl>=0.25**。
                  本地 trl 0.24.0 没有该参数，这里只保留接口与文档，不实现（README §9）。
思考段用 Qwen3 原生 <think> / </think> special token 界定；直接作答模式用 chat template 的
enable_thinking=False（模板注入空 think 块）并预填 <answer>，限制 max_new_tokens，使模型无法在答案外推理。
"""
from __future__ import annotations
import cot_compress  # noqa: F401  unsloth first
from dataclasses import dataclass
import torch
import time
from transformers import LogitsProcessor, LogitsProcessorList

from .parsing import ANSWER_OPEN
from .span import HOOK

class StepTimer(LogitsProcessor):
    """记录每个解码步的时间戳，用于定位长上下文减速起点（tok/s 按 window 步分段）。"""
    def __init__(self):
        self.stamps: list[float] = []
    def __call__(self, input_ids, scores):
        self.stamps.append(time.time())
        return scores
    def windows(self, window: int = 256) -> list[dict]:
        out = []
        for i in range(0, len(self.stamps) - 1, window):
            seg = self.stamps[i:i + window + 1]
            if len(seg) >= 2:
                out.append(dict(step=i, steps_per_s=round((len(seg) - 1) / (seg[-1] - seg[0]), 1)))
        return out

@dataclass
class GenOut:
    ids: list[int]          # 完成段 token id（已去掉 pad / eos 之后的部分，含 eos 前全部）
    text: str               # 解码文本（skip_special_tokens=False，保留 <think> 等标记）
    finished: bool          # 是否遇到 eos（否则被 max_new_tokens 截断）
    think_tokens: int       # <think> 与 </think> 之间的 token 数（仅诊断；L 的定义见 evaluate._row / reward.span_length）
    has_think_close: bool
    forced: bool = False    # 是否被 SpanController 预算强制收尾（HOOK 已安装时有效）

def special_ids(tok) -> dict:
    ids = {
        "think_open": tok.convert_tokens_to_ids("<think>"),
        "think_close": tok.convert_tokens_to_ids("</think>"),
    }
    assert all(isinstance(v, int) and v >= 0 for v in ids.values()), f"think tokens missing in tokenizer: {ids}"
    return ids

def chat_prompt(tok, system: str, user: str, *, think: bool, prefill: str = "") -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    return text + prefill

def direct_prefill() -> str:
    """直接作答模式的预填：模板已注入 <think>\\n\\n</think>\\n\\n，再补 <answer>。"""
    return ANSWER_OPEN

def _count_think(ids: list[int], sp: dict) -> tuple[int, bool]:
    start = 0
    if sp["think_open"] in ids:
        start = ids.index(sp["think_open"]) + 1
    if sp["think_close"] in ids[start:]:
        end = ids.index(sp["think_close"], start)
        return end - start, True
    return len(ids) - start, False

@torch.no_grad()
def generate_texts(model, tok, prompts: list[str], *, max_new_tokens: int, sampling: dict,
                   batch_size: int = 8, seed: int = 0, logits_processor=None, progress=None,
                   batch_stats: list | None = None) -> list[GenOut]:
    """prompts 已是完整模板文本。返回与 prompts 对齐的 GenOut 列表。
    batch_stats 非 None 时，每个 batch 追加 dict(batch, size, prompt_len, max_new, sec, peak_gib, steps_per_s_by_256)。"""
    sp = special_ids(tok)
    eos_ids = set()
    gc = model.generation_config
    e = gc.eos_token_id if gc.eos_token_id is not None else tok.eos_token_id
    eos_ids.update(e if isinstance(e, (list, tuple)) else [e])
    pad_id = tok.pad_token_id
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=pad_id, use_cache=True)
    if sampling.get("do_sample", True):
        gen_kwargs.update(do_sample=True, temperature=float(sampling.get("temperature", 0.6)),
                          top_p=float(sampling.get("top_p", 0.95)), top_k=int(sampling.get("top_k", 20)))
    else:
        gen_kwargs.update(do_sample=False)
    procs = list(logits_processor) if isinstance(logits_processor, (list, tuple)) else (
        [logits_processor] if logits_processor is not None else [])

    outs: list[GenOut] = []
    for bi in range(0, len(prompts), batch_size):
        chunk = prompts[bi:bi + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(seed * 100003 + bi)
        timer = StepTimer()
        gen_kwargs["logits_processor"] = LogitsProcessorList(procs + [timer])
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        gen = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **gen_kwargs)
        dt = time.time() - t0
        comp = gen[:, enc["input_ids"].shape[1]:].tolist()
        ctrl = HOOK.collect() if HOOK.enabled else None
        forced_flags = list(ctrl.forced) if (ctrl is not None and ctrl.state is not None and len(ctrl.forced) == len(comp)) else [False] * len(comp)
        if batch_stats is not None:
            batch_stats.append(dict(batch=bi // batch_size, size=len(chunk), prompt_len=int(enc["input_ids"].shape[1]),
                                    max_new=len(comp[0]), sec=round(dt, 1),
                                    peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                    reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 2),
                                    steps_per_s_by_256=timer.windows(256)))
        del gen, enc
        # 缓存分配器在多次变长 generate 后碎片化，reserved 会涨到占满显卡（实测 allocated 2.9 GiB 时 nvidia-smi 7.85 GiB），
        # 随后 WSL 把显存换到共享内存，解码 21→5 steps/s。每个 batch 后归还；配合 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
        torch.cuda.empty_cache()
        for ri, row in enumerate(comp):
            finished = False
            cut = len(row)
            for j, t in enumerate(row):
                if t in eos_ids:
                    finished, cut = True, j
                    break
                if t == pad_id:
                    cut = j
                    break
            ids = row[:cut]
            n_think, has_close = _count_think(ids, sp)
            outs.append(GenOut(ids=ids, text=tok.decode(ids, skip_special_tokens=False),
                               finished=finished, think_tokens=n_think, has_think_close=has_close, forced=forced_flags[ri]))
        if progress:
            progress(min(bi + batch_size, len(prompts)), len(prompts))
    return outs

# ---------------------------------------------------------------------------------------------
# rollout 路径（云端 vllm）。requires trl>=0.25。
# ---------------------------------------------------------------------------------------------
def make_rollout_func(*args, **kwargs):
    """返回可传给 `GRPOTrainer(rollout_func=...)` 的函数（trl>=0.25 接口：
        rollout_func(prompts: list[str], args: GRPOConfig, processing_class) -> dict(prompt_ids, completion_ids, logprobs, ...)
    ）。vllm 下 HF generate 包装不生效，臂 B 的 logits mask 须通过 vllm 的 logits_processors / SamplingParams 注入。
    本地 trl 0.24.0 无此参数，未实现；云端建 venv 时 pin trl>=0.25，先跑 scripts/00_check_env.py 确认
    trl_hooks.rollout_func == true，再启用 configs/cloud_*.yaml 的 generation.hook=rollout。见 README §9。
    """
    raise NotImplementedError("generation.hook=rollout requires trl>=0.25 (GRPOTrainer rollout_func); see README §9")
