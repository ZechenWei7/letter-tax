"""模板检测器（v7 §2）：四个正则/启发式检测器，训练前发布并冻结。输入 think 段 + IR 题面 + N，输出命中类型列表。
  dpll_cdcl      : 显式算法骨架——出现 dpll / cdcl / unit propagat* / pure literal / clause learn* / backjump / decision level / "backtrack" 与 "branch" 同现
  bitmask_enum   : 位掩码枚举——"2^N"、2^N 的十进制值、"for mask"、"all assignments/combinations"、或 ≥ 8 个不同的 N 位 0/1 串
  english_assume : 英文模板 "assume (person) i is a knight/knave, then …"——≥ 3 处（含 suppose）
  ir_copy_assign : IR 复制 + 尾随分配——≥ 50% 的 IR 陈述行原样出现在 think 里，且 think 末尾 200 字符内有 N 位 0/1 串
准入第 7 条：≥ 5% 原生轨迹命中任一类型 → 拒格（admission.py: template_hit_rate）。
"""
from __future__ import annotations
import re
from .strategy import binary_strings

_DPLL = re.compile(r"\b(dpll|cdcl|unit[- ]propagat\w*|pure[- ]literal|clause[- ]learn\w*|backjump\w*|decision[- ]level)\b", re.I)
_BACKTRACK = re.compile(r"\bbacktrack\w*", re.I); _BRANCH = re.compile(r"\bbranch\w*", re.I)
_MASK = re.compile(r"\b(for\s+mask|bit\s*mask|all\s+(?:2\^?\w*\s+)?(?:assignments|combinations|possibilities)|enumerate\s+all)\b", re.I)
_ASSUME = re.compile(r"\b(assume|suppose)\s+(?:that\s+)?(?:person\s+)?(\d+)\s+(?:is|be|were|was)\s+(?:a\s+|an\s+)?(knight|knave)\b[^.\n]{0,60}?\bthen\b", re.I)

def detect(think: str, prompt: str, n_persons: int) -> list[str]:
    hits = []
    if _DPLL.search(think) or (_BACKTRACK.search(think) and _BRANCH.search(think)):
        hits.append("dpll_cdcl")
    bs = binary_strings(think, n_persons)
    if (f"2^{n_persons}" in think or f"2**{n_persons}" in think or str(2 ** n_persons) in think or _MASK.search(think) or len(set(bs)) >= 8):
        hits.append("bitmask_enum")
    if len(_ASSUME.findall(think)) >= 3:
        hits.append("english_assume")
    ir_lines = [ln.split(": ", 1)[1] for ln in prompt.split("\n") if re.match(r"^s\d+: ", ln)]
    if ir_lines:
        copied = sum(1 for ln in ir_lines if ln in think)
        tail = think[-200:]
        if copied / len(ir_lines) >= 0.5 and binary_strings(tail, n_persons):
            hits.append("ir_copy_assign")
    return hits

def hit_rate(thinks: list[str], prompts: list[str], n_persons: int) -> dict:
    hs = [detect(t, p, n_persons) for t, p in zip(thinks, prompts)]
    from collections import Counter
    c = Counter(h for hh in hs for h in set(hh)); tot = max(1, len(hs))
    return dict(any=sum(1 for hh in hs if hh) / tot, by_type={k: c[k] / tot for k in ("dpll_cdcl", "bitmask_enum", "english_assume", "ir_copy_assign")}, n=len(hs))
