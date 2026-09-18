"""云端 vLLM 路径（requires trl>=0.25，本地未测）。
- VLLMSpanProcessor：vLLM per-request logits processor（V0 API：fn(token_ids, logits) 或 fn(prompt_ids, token_ids, logits)），
  无状态：由已生成 token 推断阶段——含 </think>（special id）→ answering（不屏蔽）；长度 ≥ force_start → 逐 token 强制 close；否则屏蔽（臂 B）。
  与本地 SpanController 语义一致（README §2）。
- make_rollout_func：给 TRL GRPOTrainer(rollout_func=...) 用；用 trainer 的 colocate LLM 自己采样，返回 prompt_ids / completion_ids / logprobs。
- generate_texts_vllm：eval 用的 vLLM 生成，返回与 generation.GenOut 同字段的对象。
验收点（README §9 第一小时清单）：vLLM 版本是否接受 SamplingParams(logits_processors=...)（V1 引擎可能不接受 → export VLLM_USE_V1=0）；
TRL rollout_func 收到的 prompts 是否已按 num_generations 重复；logprobs 字段格式。
"""
from __future__ import annotations
import torch
import random
from .span import CLOSE_TEXT, LIFT_TEXT
from .generation import GenOut
from . import rewards as _rewards

ANSWER_MAX_TOKENS = 40   # 答案区上限 32 + 收尾余量

class VLLMSpanProcessor:
    def __init__(self, tok, cap: int, reserve: int, banned_ids=None, vocab_size: int | None = None):
        self.tok, self.cap, self.reserve = tok, cap, reserve
        self.close_ids = tok(CLOSE_TEXT, add_special_tokens=False)["input_ids"]
        self.force_start = cap - reserve - len(self.close_ids)
        lift = tok(LIFT_TEXT, add_special_tokens=False)["input_ids"]
        assert len(lift) == 1, f"{LIFT_TEXT!r} must be a single token, got {lift}"
        self.lift_id = lift[0]
        self.banned_ids = list(banned_ids) if banned_ids else None
        self.vocab_size = vocab_size
        self._mask_cache: dict = {}

    def _mask(self, logits):
        key = (logits.device, logits.shape[-1])
        m = self._mask_cache.get(key)
        if m is None:
            V = logits.shape[-1]
            m = torch.zeros(V, dtype=torch.bool)
            m[torch.tensor([i for i in self.banned_ids if i < V], dtype=torch.long)] = True
            m[len(self.tok):] = True
            m = m.to(logits.device); self._mask_cache[key] = m
        return m

    def __call__(self, *args):
        token_ids, logits = args[-2], args[-1]          # 兼容 (token_ids, logits) 与 (prompt_ids, token_ids, logits)
        n = len(token_ids)
        # 自然出现在 force_start 之前的 </think> → answering。（close 的第一个 token 就是 </think>，所以强制段内不能按它判 lift，
        # 否则强制到 </think> 就停了，后面的 "\n\n<answer>" 不再强制——与本地 SpanController 的状态机不一致。）
        if self.lift_id in token_ids[:self.force_start]:
            return logits
        if n >= self.force_start:                        # 预算强制：逐 token 输出 close；close 完成后自然进入 answering
            k = n - self.force_start
            if k < len(self.close_ids):
                logits[:] = float("-inf"); logits[self.close_ids[k]] = 0.0
            return logits
        if self.banned_ids is not None:
            logits[self._mask(logits)] = float("-inf")
        return logits

class _StubSamplingParams:
    """vllm 未安装时（本地单测）用的替身，字段同名。"""
    def __init__(self, **kw): self.__dict__.update(kw)

EXTRA_KEY = "cot_span"   # SamplingParams.extra_args[EXTRA_KEY] = {"cap": int, "reserve": int, "mode": "none"|"letterfree"|...}

class SpanBatchState:
    """vLLM V1 引擎级 logits processor 的核心状态机（不依赖 vllm，可本地单测）。
    vLLM 0.10.2：V1 不支持 per-request logits_processors，V0 也无条件拒绝 → 用引擎级接口（v1/sample/logits_processor）。
    每个请求的配置来自 SamplingParams.extra_args["cot_span"]；out 是该请求已生成 token 列表的**活引用**。
    语义与 SpanController / VLLMSpanProcessor 一致：force_start 之前出现 </think> → answering；≥ force_start → 逐 token 强制 close；否则按 mode 屏蔽。"""
    def __init__(self, tok, vocab_size: int, device="cpu"):
        self.tok, self.V, self.device = tok, vocab_size, device
        self.close_ids = tok(CLOSE_TEXT, add_special_tokens=False)["input_ids"]
        lift = tok(LIFT_TEXT, add_special_tokens=False)["input_ids"]; assert len(lift) == 1
        self.lift_id = lift[0]
        self.reqs: dict[int, dict] = {}
        self._masks: dict[str, torch.Tensor] = {}

    def make_entry(self, extra_args, out_ids):
        cfg = (extra_args or {}).get(EXTRA_KEY)
        if not cfg:
            return None
        cap, reserve = int(cfg["cap"]), int(cfg.get("reserve", 24))
        return dict(force_start=cap - reserve - len(self.close_ids), mode=str(cfg.get("mode", "none")), out=out_ids, lifted=False)

    def banned_mask(self, mode: str) -> torch.Tensor:
        m = self._masks.get(mode)
        if m is None:
            from .vocab_mask import banned_ids
            m = torch.zeros(self.V, dtype=torch.bool)
            ids = banned_ids(self.tok, mode, vocab_size=self.V)
            if ids:
                m[torch.tensor(ids, dtype=torch.long)] = True
            m = m.to(self.device); self._masks[mode] = m
        return m

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        by_mode: dict[str, list[int]] = {}
        for idx, st in self.reqs.items():
            out = st["out"]; n = len(out); fs = st["force_start"]
            if not st["lifted"] and n and n <= fs and out[-1] == self.lift_id:
                st["lifted"] = True                      # 自然 </think>（在 force_start 之前）
            if st["lifted"]:
                continue
            if n >= fs:
                k = n - fs
                if k < len(self.close_ids):
                    logits[idx] = float("-inf"); logits[idx, self.close_ids[k]] = 0.0
                continue
            if st["mode"] != "none":
                by_mode.setdefault(st["mode"], []).append(idx)
        for mode, rows in by_mode.items():
            r = torch.tensor(rows, dtype=torch.long, device=logits.device)
            logits[r] = logits[r].masked_fill(self.banned_mask(mode)[: logits.shape[-1]].to(logits.device), float("-inf"))
        return logits

def _sampling_params(args_or_cfg, cap, span_cfg, temperature, top_p, top_k, logprobs=True):
    """span_cfg: None（不做强制/屏蔽，如只生成答案段）或 {"cap","reserve","mode"} → 走引擎级 SpanLogitsProcessor。"""
    try:
        from vllm import SamplingParams
    except ImportError:
        SamplingParams = _StubSamplingParams
    return SamplingParams(n=1, temperature=float(temperature), top_p=float(top_p), top_k=int(top_k) if top_k else -1,
                          max_tokens=int(cap), logprobs=0 if logprobs else None,
                          extra_args=({EXTRA_KEY: dict(span_cfg)} if span_cfg else None))

def engine_kwargs() -> dict:
    """构造 vllm.LLM 时必须带上的参数：注册引擎级处理器；logprobs 取处理后的（与 trainer 的 mask 重归一化一致）。"""
    from .vllm_v1_proc import SpanLogitsProcessor
    return dict(logits_processors=[SpanLogitsProcessor], logprobs_mode="processed_logprobs")

def patch_vllm_llm_init():
    """TRL colocate 自己构造 LLM(...) 且不透传参数 → 包一层 __init__ 注入引擎级处理器。幂等。"""
    import vllm
    if getattr(vllm.LLM.__init__, "_cot_patched", False):
        return
    orig = vllm.LLM.__init__
    def init(self, *a, **k):
        for kk, vv in engine_kwargs().items():
            k.setdefault(kk, vv)
        if "max_num_seqs" in k:                      # TRL 设成 per_device×steps_per_generation（=32）；eval n=1000 需要更高并发
            k["max_num_seqs"] = max(int(k["max_num_seqs"]), 128)
        return orig(self, *a, **k)
    init._cot_patched = True
    vllm.LLM.__init__ = init

def make_rollout_func(get_llm, tok, *, cap: int, reserve: int, mask_mode: str = "none", banned_ids=None):
    """返回 TRL>=0.25 的 rollout_func(prompts, args, processing_class) -> dict。get_llm() 延迟取 trainer.llm（colocate）。
    屏蔽 / 预算强制由引擎级 SpanLogitsProcessor 施加（LLM 须经 patch_vllm_llm_init / engine_kwargs 构造）。"""
    span_cfg = dict(cap=cap, reserve=reserve, mode=mask_mode)
    def rollout_func(prompts, args, processing_class):
        llm = get_llm()
        sp = _sampling_params(args, cap, span_cfg, args.temperature, args.top_p, args.top_k)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        prompt_ids, completion_ids, logprobs = [], [], []
        for o in outs:
            c = o.outputs[0]
            prompt_ids.append(list(o.prompt_token_ids)); completion_ids.append(list(c.token_ids))
            logprobs.append([lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)] if c.logprobs else None)
        return dict(prompt_ids=prompt_ids, completion_ids=completion_ids, logprobs=logprobs)
    rollout_func.span_cfg = span_cfg
    return rollout_func

@torch.no_grad()
def generate_texts_vllm(llm, tok, prompts: list[str], *, max_new_tokens: int, sampling: dict, reserve: int = 24,
                        mask_mode: str = "none", banned_ids=None, force: bool = True, seed: int = 0) -> list[GenOut]:
    span_cfg = dict(cap=max_new_tokens, reserve=reserve, mode=mask_mode) if force else None
    close_len = len(tok(CLOSE_TEXT, add_special_tokens=False)["input_ids"])
    force_start = max_new_tokens - reserve - close_len
    sp = _sampling_params(None, max_new_tokens, span_cfg, sampling.get("temperature", 0.6), sampling.get("top_p", 0.95),
                          sampling.get("top_k", 20), logprobs=False)
    sp.seed = seed
    outs = llm.generate(prompts, sp, use_tqdm=False)
    res = []
    for o in outs:
        c = o.outputs[0]; ids = list(c.token_ids)
        text = tok.decode(ids, skip_special_tokens=False)
        forced = bool(force and len(ids) >= force_start + close_len and CLOSE_TEXT in text and LIFT_TEXT not in tok.decode(ids[:force_start]))
        res.append(GenOut(ids=ids, text=text, finished=(c.finish_reason == "stop"), think_tokens=0,
                          has_think_close=LIFT_TEXT in text, forced=forced))
    return res


# ---------------------------------------------------------------------------------------------
# C_rand / D：think 段不是策略采样——放进 prompt，策略只生成答案（损失自然只落在答案 token 上）。
# C_rand rollout 循环（README §9）：
#   for prompt, item in batch:
#       L     = stats.sample_length(item)            # 从 B 收敛后的长度分布抽（同题优先）
#       think = stats.sample_unigram(L)              # think token 从 B 收敛轨迹 unigram 抽
#       ctx   = prompt(以 "<think>\n" 结尾) + decode(think) + "\n</think>\n\n<answer>"
#       ans   = llm.generate(ctx, max_tokens=16)     # 策略只采样答案
#       yield prompt_ids=tok(ctx), completion_ids=ans, L=L   # 奖励侧通道拿 L
# D：同上但 L=0（force-close 在位置 0），奖励无长度项。
# ---------------------------------------------------------------------------------------------
def build_prefilled_prompt(tok, prompt: str, think_ids: list[int]) -> str:
    """prompt 已以 "<think>\n" 结尾（所有臂统一预填）；这里只接 think 体 + 收尾。"""
    body = tok.decode(think_ids) if think_ids else ""
    return prompt + body + ("\n" if body else "") + CLOSE_TEXT

def make_prefilled_rollout_func(get_llm, tok, *, mode: str, stats=None, item_id_of=None, seed: int = 0,
                                answer_max_tokens: int = ANSWER_MAX_TOKENS):
    """mode='crand'：think 从 stats 抽；mode='d'：think 为空。item_id_of(prompt_text)->item id（同题优先）。"""
    assert mode in ("crand", "d")
    rng = random.Random(seed)
    def rollout_func(prompts, args, processing_class):
        llm = get_llm()
        ctxs, Ls, think_ids_batch = [], [], []
        for p in prompts:
            if mode == "d":
                ids = []
            else:
                iid = item_id_of(p) if item_id_of else None
                L = stats.sample_length(iid, rng); ids = stats.sample_unigram(L, rng)
            ctxs.append(build_prefilled_prompt(tok, p, ids)); Ls.append(len(ids)); think_ids_batch.append(ids)
        rollout_func.last_think_ids = think_ids_batch          # 审计：think 段 token 全部来自采样器，不来自策略
        sp = _sampling_params(args, answer_max_tokens, None, args.temperature, args.top_p, args.top_k)   # 只生成答案段：无强制 / 屏蔽
        outs = llm.generate(ctxs, sp, use_tqdm=False)
        prompt_ids, completion_ids, logprobs = [], [], []
        for o in outs:
            c = o.outputs[0]
            prompt_ids.append(list(o.prompt_token_ids)); completion_ids.append(list(c.token_ids))
            logprobs.append([lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)] if c.logprobs else None)
        _rewards.SIDE["L"] = Ls
        return dict(prompt_ids=prompt_ids, completion_ids=completion_ids, logprobs=logprobs)
    rollout_func.mode = mode
    return rollout_func
