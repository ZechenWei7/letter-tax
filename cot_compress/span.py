"""思考段控制器：预算强制收尾 + 臂 B 词表屏蔽，作为 LogitsProcessor 注入 HF generate。

边界定义（README §2）：思考段 L = 生成开始 → 模型输出 "<answer>" 之前的全部 token，与 <think> 标记无关。
每行三态：thinking →（到顶时 forcing：逐 token 强制输出 close_text）→ answering。
  - thinking 阶段：若给了 banned_ids，把它们置 -inf（臂 B）。
  - 生成长度到 cap - reserve - len(close) 仍未出现 "</think>" 的行进入 forcing。
  - answering 阶段（"</think>" 之后）不再屏蔽、不再强制。L 的边界仍是最后一个 <answer>（在 </think> 之后紧跟）。

注入方式：给模型类打补丁 `_get_logits_processor`，把控制器**插到处理器列表最前**（先于 temperature / top-k / top-p），
这样屏蔽不会与 top-k 冲突（否则 top-k 保留的 20 个全被屏蔽会得到全 -inf）；Unsloth 反复替换实例级 model.generate 也不影响。
"""
from __future__ import annotations
import time
import torch
from transformers import LogitsProcessor, LogitsProcessorList

CLOSE_TEXT = "</think>\n\n<answer>"
ANSWER_OPEN = "<answer>"
LIFT_TEXT = "</think>"   # 进入 answering 的标志。Qwen3 会在思考里预演 "<answer>X</answer>"，若以 <answer> 为界，臂 B 可用假标签逃出屏蔽

class SpanController(LogitsProcessor):
    def __init__(self, tok, prompt_len: int, cap: int, reserve: int, banned_ids=None, close_text: str = CLOSE_TEXT):
        self.tok, self.prompt_len, self.cap, self.reserve = tok, prompt_len, cap, reserve
        self.close_ids = tok(close_text, add_special_tokens=False)["input_ids"]
        self.force_start = cap - reserve - len(self.close_ids)
        self.banned_ids = list(banned_ids) if banned_ids else None
        self.banned = None                       # 首次调用时按 scores 的 vocab 维度建布尔掩码（禁集约 14.5 万个 id）
        self.state: list[int] | None = None      # 0 thinking / 1 forcing / 2 answering
        self.force_k: list[int] = []
        self.forced: list[bool] = []
        self.answered_at: list[int | None] = []  # 进入 answering 时的生成长度（≈ L）
        self.stamps: list[float] = []            # 每个解码步的时间戳（训练里的 generate 不经过 generate_texts，靠这个看分段 steps/s）

    def _init(self, B, device, vocab):
        self.state = [0] * B; self.force_k = [0] * B; self.forced = [False] * B; self.answered_at = [None] * B
        if self.banned_ids is not None:
            m = torch.zeros(vocab, dtype=torch.bool)
            ids = torch.tensor([i for i in self.banned_ids if i < vocab], dtype=torch.long)
            m[ids] = True
            m[len(self.tok):] = True             # tokenizer 之外的 logits 槽位也禁
            self.banned = m.to(device)

    def __call__(self, input_ids, scores):
        self.stamps.append(time.time())
        B = input_ids.shape[0]
        if self.state is None or len(self.state) != B:
            self._init(B, scores.device, scores.shape[-1])
        gen_len = input_ids.shape[1] - self.prompt_len
        neg = float("-inf")
        for i in range(B):
            if self.state[i] == 2:
                continue
            if self.state[i] == 0:
                if gen_len > 0:
                    tail = input_ids[i, max(self.prompt_len, input_ids.shape[1] - 8):]
                    if LIFT_TEXT in self.tok.decode(tail):
                        self.state[i] = 2; self.answered_at[i] = gen_len
                        continue
                if gen_len >= self.force_start:
                    self.state[i] = 1; self.force_k[i] = 0; self.forced[i] = True
            if self.state[i] == 1:
                k = self.force_k[i]
                scores[i, :] = neg
                scores[i, self.close_ids[k]] = 0.0
                self.force_k[i] = k + 1
                if k + 1 >= len(self.close_ids):
                    self.state[i] = 2; self.answered_at[i] = self.cap
                continue
            if self.banned is not None:
                scores[i, self.banned] = neg
        return scores

    def windows(self, window: int = 256) -> list[float]:
        out = []
        for i in range(0, len(self.stamps) - 1, window):
            seg = self.stamps[i:i + window + 1]
            if len(seg) >= 2:
                out.append(round((len(seg) - 1) / (seg[-1] - seg[0]), 1))
        return out

    @property
    def stats(self) -> dict:
        n = len(self.forced)
        return dict(rows=n, forced=sum(self.forced), natural_end=(n - sum(self.forced)) / n if n else None)


class SpanHook:
    """类级补丁的状态容器。install() 后每次 generate 都会新建一个 SpanController，last 指向最近一次。"""
    def __init__(self):
        self.tok = None; self.reserve = 24; self.banned_ids = None; self.mask_mode = "none"; self.enabled = False
        self.last: SpanController | None = None
        self.totals = dict(rows=0, forced=0)
        self._patched: set = set()

    def configure(self, tok, *, reserve: int, banned_ids=None, mask_mode: str | None = None):
        """mask_mode：vLLM 引擎级处理器按模式名自建掩码（HF 路径用 banned_ids）。未给时：有 banned_ids → letterfree，否则 none。"""
        self.tok, self.reserve, self.banned_ids, self.enabled = tok, reserve, banned_ids, True
        self.mask_mode = mask_mode if mask_mode is not None else ("letterfree" if banned_ids else "none")
        return self

    def install(self, model):
        """给 model（及 PEFT 包装下的底层模型）的类打补丁。幂等。"""
        targets = []
        for m in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
            if m is not None and hasattr(type(m), "_get_logits_processor"):   # PEFT 包装类靶 __getattr__ 委托，不能在类上补丁
                targets.append(type(m))
        for cls in targets:
            if cls in self._patched:
                continue
            orig = cls._get_logits_processor
            hook = self
            def patched(self_, generation_config, input_ids_seq_length=None, *a, **kw):
                lst = orig(self_, generation_config, input_ids_seq_length, *a, **kw)
                if hook.enabled and input_ids_seq_length is not None:
                    cap = generation_config.max_new_tokens
                    if cap is None and generation_config.max_length is not None:
                        cap = generation_config.max_length - input_ids_seq_length
                    if cap is not None and cap > hook.reserve + 8:
                        ctrl = SpanController(hook.tok, input_ids_seq_length, cap, hook.reserve, hook.banned_ids)
                        hook.last = ctrl
                        lst.insert(0, ctrl)
                return lst
            cls._get_logits_processor = patched
            self._patched.add(cls)
        return self

    def collect(self):
        """把 last 的统计累加进 totals（每次 generate 后调一次）。"""
        if self.last is not None and self.last.state is not None:
            s = self.last.stats
            self.totals["rows"] += s["rows"]; self.totals["forced"] += s["forced"]
        return self.last

HOOK = SpanHook()
