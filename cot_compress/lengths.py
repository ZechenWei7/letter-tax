"""长度单位（v4 第 6 条）：token 数、code point 数（不是字节）、自适应编码器比特数、zstd 压缩比。
自适应编码器：每臂轨迹对半分，一半拟合 token 5-gram（绝对折扣 + 逐阶回退），另一半算每条轨迹的总比特。"""
from __future__ import annotations
import math, random
from collections import defaultdict

def code_points(text: str) -> int:
    return len(text)

class NGramCoder:
    """token id 序列的 n-gram 模型（默认 5），绝对折扣 D=0.75 + 回退到低阶，最低阶为 (unigram + 均匀) 混合。"""
    def __init__(self, n: int = 5, vocab_size: int = 152_000, discount: float = 0.75):
        self.n, self.V, self.D = n, vocab_size, discount
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]   # order k: context tuple(len k) -> token -> count

    def fit(self, seqs):
        for s in seqs:
            for i, t in enumerate(s):
                for k in range(self.n):
                    if i - k < 0: break
                    self.counts[k][tuple(s[i - k:i])][t] += 1
        return self

    def prob(self, ctx, t) -> float:
        p = 1.0 / self.V
        for k in range(0, self.n):
            if len(ctx) < k: break
            h = tuple(ctx[len(ctx) - k:]) if k else ()
            d = self.counts[k].get(h)
            if not d: continue
            tot = sum(d.values()); types = len(d)
            p = max(d.get(t, 0) - self.D, 0) / tot + (self.D * types / tot) * p
        return p

    def bits(self, seq) -> float:
        return sum(-math.log2(self.prob(seq[max(0, i - self.n + 1):i], t)) for i, t in enumerate(seq))

def adaptive_bits(token_seqs: list[list[int]], vocab_size: int, seed: int = 0, n: int = 5) -> dict:
    """对半分：一半拟合、另一半评估；返回评估半的每条比特均值及 bits/token。"""
    rng = random.Random(seed); idx = list(range(len(token_seqs))); rng.shuffle(idx)
    half = len(idx) // 2
    fit = [token_seqs[i] for i in idx[:half]]; ev = [token_seqs[i] for i in idx[half:]]
    if not fit or not ev:
        return dict(bits_mean=None, bits_per_token=None, n_fit=len(fit), n_eval=len(ev))
    coder = NGramCoder(n=n, vocab_size=vocab_size).fit(fit)
    b = [coder.bits(s) for s in ev]; toks = sum(len(s) for s in ev)
    return dict(bits_mean=sum(b) / len(b), bits_per_token=(sum(b) / toks) if toks else None, n_fit=len(fit), n_eval=len(ev))

def zstd_ratio(text: str) -> tuple[float | None, str]:
    b = text.encode("utf-8")
    if len(b) < 64:
        return None, "n/a"
    try:
        import zstandard
        return len(zstandard.ZstdCompressor(level=19).compress(b)) / len(b), "zstd19"
    except ImportError:
        import zlib
        return len(zlib.compress(b, 9)) / len(b), "zlib9"

def letter_fraction(text: str) -> float:
    """think 段中字母（Unicode L*）字符占非空白字符的比例（A 的字母占比曲线："低" = < 30%）。"""
    import unicodedata
    chars = [c for c in text if not c.isspace()]
    return (sum(unicodedata.category(c)[0] == "L" for c in chars) / len(chars)) if chars else 0.0
