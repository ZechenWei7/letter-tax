"""策略类检测器（v7 §3）：每条轨迹标注 分情况标记数、传播步数、枚举、dump-and-verify、提示词复制率。
规则训练前发布、冻结（不看 B 的轨迹再改）。词表 = warm.CONTROL 的子集及其 WARM_MAP 符号，保证 B_warm 变换下计数不变。
  case_splits   : {assume, suppose, case, if} ∪ {», §, ¿}
  propagation   : {then, so, therefore, thus, hence} ∪ {→, ⇒}
  contradiction : {contradiction, contradicts, impossible} ∪ {×}
  consistent    : {consistent} ∪ {✓}
  enumeration   : ≥3 个不同的 N 位 0/1 串，或出现 "2^N" / 2^N 的十进制值（N 由题给）
  dump_verify   : ≥2 个 N 位 0/1 串 且 (contradiction + consistent) 标记 ≥ 1
  copy_rate     : think 段中被"≥4 token 的 IR 复制片段"覆盖的 token 比例（token = 分词器 id；无分词器用空白切分）
策略类 strategy_class：enumeration → "enumerate"；dump_verify → "dump_verify"；case_splits ≥ 1 → "case_split"；propagation ≥ 1 → "propagate_only"；否则 "other"。
"""
from __future__ import annotations
import re
from collections import Counter
from .warm import WARM_MAP

ROLES = {
    "case_splits": {"assume", "suppose", "case", "if"},
    "propagation": {"then", "so", "therefore", "thus", "hence"},
    "contradiction": {"contradiction", "contradicts", "impossible"},
    "consistent": {"consistent"},
}
ROLE_SYMBOLS = {role: {WARM_MAP[w] for w in words} for role, words in ROLES.items()}
_WORD = re.compile(r"[^\W\d_]+", re.U)

def _count_role(text: str, role: str) -> int:
    words = Counter(w.lower() for w in _WORD.findall(text))
    n = sum(words[w] for w in ROLES[role])
    for sym in ROLE_SYMBOLS[role]:
        n += text.count(sym)
    return n

def binary_strings(text: str, n: int) -> list[str]:
    """N 位 0/1 串（允许单空格分隔的 N 个 0/1）。"""
    tight = re.findall(rf"(?<![0-9])[01]{{{n}}}(?![0-9])", text)
    spaced = [" ".join(m.split()) for m in re.findall(rf"(?<![0-9])(?:[01] ){{{n - 1}}}[01](?![0-9])", text)]
    return tight + spaced

def copy_rate(think_tokens: list, prompt_tokens: list, n: int = 4) -> float | None:
    """think 中被长度 ≥ n 的 prompt n-gram 覆盖的 token 比例。"""
    if not think_tokens:
        return None
    grams = {tuple(prompt_tokens[i:i + n]) for i in range(len(prompt_tokens) - n + 1)}
    cover = [False] * len(think_tokens)
    for i in range(len(think_tokens) - n + 1):
        if tuple(think_tokens[i:i + n]) in grams:
            for k in range(i, i + n): cover[k] = True
    return sum(cover) / len(cover)

def copied_mask(think_tokens: list, prompt_tokens: list, n: int = 4) -> list[bool]:
    grams = {tuple(prompt_tokens[i:i + n]) for i in range(len(prompt_tokens) - n + 1)}
    cover = [False] * len(think_tokens)
    for i in range(len(think_tokens) - n + 1):
        if tuple(think_tokens[i:i + n]) in grams:
            for k in range(i, i + n): cover[k] = True
    return cover

def tokens_of(text: str, tok=None) -> list:
    return tok(text, add_special_tokens=False)["input_ids"] if tok is not None else text.split()

def annotate(think: str, prompt: str, n_persons: int, tok=None) -> dict:
    bs = binary_strings(think, n_persons)
    c = {role: _count_role(think, role) for role in ROLES}
    enum = len(set(bs)) >= 3 or f"2^{n_persons}" in think or str(2 ** n_persons) in think
    dump = len(bs) >= 2 and (c["contradiction"] + c["consistent"]) >= 1
    cr = copy_rate(tokens_of(think, tok), tokens_of(prompt, tok))
    cls = "enumerate" if enum else ("dump_verify" if dump else ("case_split" if c["case_splits"] >= 1 else ("propagate_only" if c["propagation"] >= 1 else "other")))
    return dict(case_splits=c["case_splits"], propagation=c["propagation"], contradiction=c["contradiction"], consistent=c["consistent"],
                n_binary_strings=len(bs), enumeration=enum, dump_verify=dump, copy_rate=cr, strategy_class=cls)

def class_distribution(annots: list[dict]) -> dict:
    cnt = Counter(a["strategy_class"] for a in annots); tot = max(1, len(annots))
    return {k: cnt[k] / tot for k in ("enumerate", "dump_verify", "case_split", "propagate_only", "other")}
