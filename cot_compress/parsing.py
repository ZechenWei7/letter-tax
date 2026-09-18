"""输出解析。思考段定界分两种（README §2）：
    span="tags"       (a) 原生思考：<think> ... </think> ... <answer>X</answer>；</think> 缺失判格式错误
    span="pre_answer" 思考段 = 最后一个 <answer> 之前的全部生成文本；唯一性只看第一个 </think> 之后的标签
    span="none"       直接作答：text 以 <answer> 开头（调用方补回预填）
共同严格规则：只接受 <answer> 内的单一合法答案；<answer> 缺失或出现多次（不同值）判为格式错误。
pre_answer 额外规则：</think> 与最终 <answer> 之间只允许空白（text_after_think）。
"""
from __future__ import annotations
import re
from dataclasses import dataclass

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"
_ANS_RE = re.compile(re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.S)

@dataclass
class Parsed:
    think: str | None          # 思考段文本（不含标记）；None 表示没找到 </think>
    answer: str | None         # <answer> 内文本（strip 后）；None 表示格式错误
    format_ok: bool
    reason: str                # ok / no_think_close / no_answer / multi_answer / bad_answer / unfinished / text_after_think
    tail: str = ""             # </think> 之后的全部文本（含 answer）

def split_think(text: str) -> tuple[str | None, str]:
    """返回 (think, tail)。think 为 <think>..</think> 之间文本；没有 </think> 则 (None, text)。"""
    if THINK_CLOSE not in text:
        return None, text
    head, tail = text.split(THINK_CLOSE, 1)
    if THINK_OPEN in head:
        head = head.split(THINK_OPEN, 1)[1]
    return head, tail

SPANS = ("tags", "pre_answer", "none")

def parse_completion(text: str, *, span: str = "tags", validate=None, validate_norm=None) -> Parsed:
    """解析模型完整输出。span 见模块说明。validate: 可选 callable(str)->bool，任务级答案合法性检查；
    validate_norm: 可选 callable(str)->str|None，答案归一化（pre_answer 模式判断多个标签是否同值）。"""
    assert span in SPANS, span
    if span == "tags":
        think, tail = split_think(text)
        if think is None:
            return Parsed(None, None, False, "no_think_close", text)
    elif span == "pre_answer":
        # L 段 = 最后一个 <answer> 之前的全部文本（Qwen3 会在思考里预演 "<answer>X</answer>"，预演不算结束）
        think = text.rsplit(ANSWER_OPEN, 1)[0] if ANSWER_OPEN in text else None
        tail = text
    else:
        think, tail = "", text
    found = _ANS_RE.findall(tail)
    if len(found) == 0:
        reason = "unfinished" if ANSWER_OPEN in tail else "no_answer"
        return Parsed(think, None, False, reason, tail)
    if span == "pre_answer":
        # 只看第一个 </think> 之后的标签（思考内草稿忽略）："唯一" = 唯一的答案值；不同值 → multi_answer
        region = text.split(THINK_CLOSE, 1)[1] if THINK_CLOSE in text else text
        found = _ANS_RE.findall(region)
        if not found:
            return Parsed(think, None, False, ("unfinished" if ANSWER_OPEN in region else "no_answer"), tail)
        vals = {(validate_norm(x) if validate_norm else x.strip()) for x in found}
        if len(vals) > 1:
            return Parsed(think, None, False, "multi_answer", tail)
        found = found[-1:]
        # </think> 与其后第一个 <answer> 之间只允许空白（防止把推理挪到 </think> 之后）
        if THINK_CLOSE in text and region.split(ANSWER_OPEN, 1)[0].strip() != "":
            return Parsed(think, None, False, "text_after_think", tail)
    elif len(found) > 1:
        return Parsed(think, None, False, "multi_answer", tail)
    ans = found[0].strip()
    if validate is not None and not validate(ans):
        return Parsed(think, None, False, "bad_answer", tail)
    return Parsed(think, ans, True, "ok", tail)
