"""策略描述量（2026-09-18）：每条轨迹是"每步重写全状态"还是"每步只写改动位置"。
按操作把 think 段切成段，数每段提到的位置数（状态 token "6D" 按出现个数计，显式 "position 3" / "3:" 按去重位置计）：
mean_positions_per_step ≥ 0.75·n → full_rewrite；≤ 0.25·n（且 >0）→ changed_only；其余 mixed；切不出 ≥2 段 → unparsed。
"""
from __future__ import annotations
import re, statistics

OP_RE = re.compile(r"\b(?:swap|flip)\s+\d+", re.I)
STATE_TOK = re.compile(r"\b\d+\s?[UDud]\b")
POS_EXPLICIT = re.compile(r"(?:position|pos\.?|p)\s*(\d+)\b|\b(\d+)\s*[:=]\s*\d+", re.I)

def positions_per_step(think: str, n: int) -> list[int]:
    cuts = [m.start() for m in OP_RE.finditer(think)]
    if len(cuts) < 2:
        return []
    segs = [think[a:b] for a, b in zip(cuts, cuts[1:] + [len(think)])]
    out = []
    for seg in segs:
        st = len(STATE_TOK.findall(seg))
        ex = {int(a or b) for a, b in POS_EXPLICIT.findall(seg) if (a or b) and 1 <= int(a or b) <= n}
        out.append(min(n, max(st, len(ex))))
    return out

def rewrite_style(think: str, n: int) -> dict:
    pps = positions_per_step(think, n)
    if not pps:
        return dict(style="unparsed", mean_positions_per_step=None, n_steps_parsed=0)
    m = statistics.mean(pps)
    style = "full_rewrite" if m >= 0.75 * n else ("changed_only" if 0 < m <= 0.25 * n else ("mixed" if m > 0 else "unparsed"))
    return dict(style=style, mean_positions_per_step=round(m, 2), n_steps_parsed=len(pps),
                frac_full_steps=round(sum(x >= 0.75 * n for x in pps) / len(pps), 3))

def style_summary(thinks: list[str], n: int) -> dict:
    from collections import Counter
    res = [rewrite_style(t, n) for t in thinks]
    c = Counter(r["style"] for r in res); tot = max(1, len(res))
    mp = [r["mean_positions_per_step"] for r in res if r["mean_positions_per_step"] is not None]
    return dict(style_frac={k: v / tot for k, v in c.items()}, mean_positions_per_step=(statistics.mean(mp) if mp else None), n=len(res))
