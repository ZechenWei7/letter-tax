"""B_warm 变换（v7 §4 / v8 §3）：把臂 A 的自然语言轨迹确定性地改写成无字母形式，作 B_warm 的 SFT 数据。
规则（训练前发布、冻结）：
  1. 控制词词典 CONTROL（大小写不敏感、整词匹配）→ 固定符号 WARM_MAP（全部是 Qwen3 词表里"带前导空格"与"不带"都是单个白名单 token 的符号，见 docs/warm_map.md）；
  2. 其余含字母（Unicode L*）的 word 删除；
  3. 数字、运算符、标点、空白保留；连续空白折叠为一个空格，行结构保留。
策略类分布不变性：cot_compress/strategy.py 只按 CONTROL 里的词（或其映射符号）计数，因此对 100 条 A 轨迹应用后策略类分布与源一致（tests/test_warm.py）。
"""
from __future__ import annotations
import re, unicodedata

WARM_MAP = {
    "if": "¿", "then": "→", "so": "⇒", "therefore": "⇒", "thus": "⇒", "hence": "⇒",
    "assume": "»", "suppose": "»", "case": "§",
    "contradiction": "×", "contradicts": "×", "impossible": "×", "consistent": "✓",
    "else": "«", "otherwise": "«", "but": "~",
    "and": "&", "or": "|", "not": "!",
    "knight": "1", "knave": "0", "true": "+", "false": "-", "lies": "0", "truthful": "1",
    # v8 ordering 控制词
    "before": "<", "after": ">", "cycle": "×", "first": "¡", "last": "#", "next": "^", "ready": "%", "order": "::",
}
CONTROL = tuple(WARM_MAP)
SYMBOLS = tuple(sorted(set(WARM_MAP.values())))
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.U)      # 字母串（含 don't 类缩写）
_SPACES = re.compile(r"[ \t]+")

def _has_letter(s: str) -> bool:
    return any(unicodedata.category(ch)[0] == "L" for ch in s)

def warm_text(text: str) -> str:
    def sub(m):
        w = m.group(0).lower()
        return WARM_MAP.get(w, "")
    out = _WORD.sub(sub, text)
    out = "".join(ch for ch in out if not (unicodedata.category(ch)[0] == "L"))     # 残余字母（如 CJK、混合串）删除
    lines = [_SPACES.sub(" ", ln).strip() for ln in out.split("\n")]
    return "\n".join(lines).strip()

def assert_letter_free(text: str):
    assert not _has_letter(text), text[:80]

def make_sft_rows(rows: list[dict], think_key: str = "think", require_correct: bool = True) -> list[dict]:
    """rows: A 轨迹（含 prompt、think、answer、correct）。输出 SFT 行：prompt（同 IR 题面）+ completion = warm(think) + "</think>\\n\\n<answer>…</answer>"。"""
    out = []
    for r in rows:
        if require_correct and not r.get("correct"):
            continue
        w = warm_text(r[think_key]); assert_letter_free(w)
        out.append(dict(id=r.get("id"), prompt=r["prompt"], completion=f"{w}\n</think>\n\n<answer>{r['gold'] if 'gold' in r else r['answer']}</answer>",
                        think_warm=w, src_len_chars=len(r[think_key]), warm_len_chars=len(w)))
    return out
