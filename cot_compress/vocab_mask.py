"""臂 B 系列词表屏蔽（v4）。think 阶段 allowed set：
  letterfree            = 解码后只含 N*/P*/S*/空白 的 token（无任何字母类 L*、无 M*/C*、无 U+FFFD 字节碎片）∪ {</think>}
                          − 审计追加禁集（results/whitelist_extra_banned.json：带圈/括号拉丁字母、区域指示符、可 leet 反解的英文词）
                          注意：<answer>/</answer> 分词片段和除 </think> 外的所有 special token **不在**白名单（"answer" 含字母）。
  letterfree_plus_single = letterfree ∪ 单字母 token（拉丁大小写，带空格 / 不带空格两种形式）      → 臂 A″
  none                   = 全词表（臂 A / D / C_rand 的 allowed set = 词表大小，不要误用白名单）
"""
from __future__ import annotations
import json, pathlib, string, unicodedata

MODES = ("none", "letterfree", "letterfree_plus_single")
ALLOWED_CATS = ("N", "P", "S", "Z")
LIFT_TEXT = "</think>"
EXTRA_BANNED_PATH = pathlib.Path(__file__).resolve().parents[1] / "results" / "whitelist_extra_banned.json"

def is_whitelisted_text(s: str) -> bool:
    if s == "":
        return True
    for ch in s:
        if ch.isspace():
            continue
        if ch == "�":
            return False
        if unicodedata.category(ch)[0] not in ALLOWED_CATS:
            return False
    return True

def lift_id(tok) -> int:
    ids = tok(LIFT_TEXT, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, ids
    return ids[0]

def load_extra_banned(path=EXTRA_BANNED_PATH) -> set[int]:
    p = pathlib.Path(path)
    return set(json.load(open(p))["ids"]) if p.exists() else set()

def single_letter_ids(tok) -> set[int]:
    out = set()
    for ch in string.ascii_letters:
        for form in (ch, " " + ch):
            ids = tok(form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out.add(ids[0])
    return out

def think_allowed_ids(tok, mode: str = "letterfree", extra_banned: set[int] | None = None, vocab_size: int | None = None) -> set[int]:
    assert mode in MODES, mode
    n = vocab_size or len(tok)
    if mode == "none":
        return set(range(n))
    extra = load_extra_banned() if extra_banned is None else set(extra_banned)
    specials = set(tok.all_special_ids)
    lid = lift_id(tok)
    allowed = set()
    for tid in range(len(tok)):
        if tid in specials or tid in extra:
            continue
        if is_whitelisted_text(tok.decode([tid])):
            allowed.add(tid)
    allowed.add(lid)
    if mode == "letterfree_plus_single":
        allowed |= single_letter_ids(tok) - extra
    return allowed

def banned_ids(tok, mode: str = "letterfree", vocab_size: int | None = None, extra_banned: set[int] | None = None) -> list[int]:
    n = vocab_size or len(tok)
    allowed = think_allowed_ids(tok, mode, extra_banned, n)
    return [i for i in range(n) if i not in allowed]

def masked_fraction(tok, span_text: str, banned: set[int]) -> float | None:
    ids = tok(span_text, add_special_tokens=False)["input_ids"]
    return (sum(i in banned for i in ids) / len(ids)) if ids else None
