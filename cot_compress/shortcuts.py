"""v8 捷径检测器（进准入门：原生轨迹命中 < 5%）与策略类检测器（不进门，只报告），针对 ordering 题面。训练前发布、冻结。
捷径（detect_shortcuts）：
  ir_copy_order   : ≥50% 的 IR 约束 token（硬对 a<b、析取项）作为**整 token** 出现在 think 里（且至少一个析取项被整体抄写），且 think 末尾 200 字符内有 n 事件的全排列
  perm_enum       : 全排列枚举——出现 "n!"、n! 的数值（40320 / 362880）、"all permutations"、"itertools"，或 ≥ 8 个不同的 n 事件全排列
  assign_enum     : 析取分配枚举 = 对剩余分配立方体的系统覆盖：≥4 个不同的 d 位 0/1 串；或编号 case 最大号 ≥ 2^(d−1) 且不同编号 ≥ max(4, 2^(d−1))；或提到 Gray code；或嵌套 try-both 标记 ≥ d
  guess_verify    : （D9 定义 C）≥3 个"带验证词的完整顺序"（全排列后 80 字符内有 check / verify / valid / satisf / violat / holds / fails / ✓ / ✗），且存在一对相邻、顺序不同的此类候选，
                    二者之间没有任何传播 / 分情况标记（assume suppose case if / then so therefore thus hence 及其符号）；同一顺序的重述与逐条复核不算
策略类（strategy_class）——**纯描述量**，不进任何门、不参与任何判定或读法：
  english_case（含字母的分情况叙述：assume/suppose/case + then）、symbolic_case（» § ¿ 等符号分情况、字母占比 < 0.3）、propagation（传播词/符号为主、无分情况标记）、
  enumeration（探索分支数 > 3 × 2^S，分支 = 分情况标记 + try）、mixed_probe_enum（enum 且（propagation 标记 ≥1 或 try-both 标记））、
  path_stitching（无分情况标记；断言的 a<b 对（链展开）≥90% 是题面边，且存在长度 ≥ n/2 的链）、other。优先级：enum/mixed → case → path → propagation → other。
复制率 copy_rate：≥4 token 的 IR 复制片段占 think 段比例（strategy.copy_rate）。
"""
from __future__ import annotations
import math, re
from collections import Counter
from .strategy import copy_rate, tokens_of, _count_role
from .lengths import letter_fraction

_PERM_WORDS = re.compile(r"\b(all\s+permutations|itertools|permutations?\s+of\s+all|every\s+possible\s+order)\b", re.I)
_GRAY = re.compile(r"\bgray\s*code\b", re.I)
_TRYBOTH = re.compile(r"\b(try\s+both|either\s+way|both\s+branches|both\s+cases)\b|\?\?", re.I)
_CASE_NUM = re.compile(r"\b(?:case|branch|option)\s*#?\s*(\d+)\b", re.I)
_VERIFY = re.compile(r"\b(check|verif\w*|valid|satisf\w*|violat\w*|holds|fails?)\b|[✓✗]", re.I)
_PAIR = re.compile(r"(?<![\d(])(\d+)\s*<\s*(\d+)(?![\d)])")           # 断言 a<b（排除括号内的析取复制）

def permutations_in(text: str, n: int) -> list[str]:
    out = []
    for m in re.finditer(r"\d+(?:[ \t]*(?:[ ,<→>\-]|->)[ \t]*\d+)+", text):       # 不跨行
        vals = [int(x) for x in re.findall(r"\d+", m.group(0))]
        if len(vals) == n and sorted(vals) == list(range(n)): out.append(" ".join(map(str, vals)))
    return out

_IR_ITEM = re.compile(r"\(\d+<\d+\)\|\(\d+<\d+\)|\d+<\d+")
def ir_items(prompt: str) -> list[str]:
    """IR 约束 token（硬对 a<b、析取项 (a<b)|(c<d)），只按形状解析（对 warm 变换后的题面同样有效）。"""
    return _IR_ITEM.findall(prompt)

def detect_shortcuts(think: str, prompt: str, n: int, d: int, legacy_gv: bool = False) -> list[str]:
    """legacy_gv=True：guess_verify 用预注册（r4）的原定义（≥3 个带验证词的完整顺序），只用于并报，不用于判定。"""
    hits = []
    items = ir_items(prompt); perms = permutations_in(think, n)
    toks = {t.strip(".,;:") for t in think.split()}                     # 整 token 级匹配（嵌在链 "2<3<0<1" 里的不算抄写）
    disj_items = [it for it in items if "|" in it]
    if items and sum(1 for it in items if it in toks) / len(items) >= 0.5 and any(it in toks for it in disj_items) and permutations_in(think[-200:], n):
        hits.append("ir_copy_order")                                          # 至少抄了一个析取项（复述硬约束是正常推理）
    if (f"{n}!" in think or str(math.factorial(n)) in think or _PERM_WORDS.search(think) or len(set(perms)) >= 8):
        hits.append("perm_enum")
    bits = set(re.findall(rf"(?<![0-9])[01]{{{d}}}(?![0-9])", think))
    nums = [int(x) for x in _CASE_NUM.findall(think)]
    if (len(bits) >= 4 or (nums and max(nums) >= 2 ** (d - 1) and len(set(nums)) >= max(4, 2 ** (d - 1))) or _GRAY.search(think) or len(_TRYBOTH.findall(think)) >= d):
        hits.append("assign_enum")
    # D9（2026-09-19）定义 C：先取"带验证词的完整顺序"（完整顺序后 80 字符内有验证词），要求 ≥3 个（原条件）；
    # 再要求存在一对**相邻且顺序不同**的此类候选，二者之间没有任何传播 / 分情况标记（strategy.ROLES 的 case_splits ∪ propagation 词及其符号）。
    # 即"换了候选却没有推理"才算 guess-then-verify；同一顺序的重述 / 逐条复核不算。
    cands = []
    for m in re.finditer(r"\d+(?:[ \t]*(?:[ ,<→>\-]|->)[ \t]*\d+)+", think):
        vals = [int(x) for x in re.findall(r"\d+", m.group(0))]
        if len(vals) == n and sorted(vals) == list(range(n)) and _VERIFY.search(think[m.end():m.end() + 80]):
            cands.append((m.start(), m.end(), tuple(vals)))
    if len(cands) >= 3 and (legacy_gv or any(a[2] != b[2] and reasoning_markers(think[a[1]:b[0]]) == 0 for a, b in zip(cands, cands[1:]))):
        hits.append("guess_verify")
    return hits

_MARK_WORDS = None
def reasoning_markers(s: str) -> int:
    """传播 / 分情况标记数：strategy.ROLES["case_splits"] ∪ ["propagation"] 的词（整词、大小写不敏感）与其 B_warm 符号。"""
    global _MARK_WORDS
    from .strategy import ROLES, ROLE_SYMBOLS
    if _MARK_WORDS is None:
        _MARK_WORDS = (re.compile(r"\b(" + "|".join(sorted(ROLES["case_splits"] | ROLES["propagation"])) + r")\b", re.I), ROLE_SYMBOLS["case_splits"] | ROLE_SYMBOLS["propagation"])
    return len(_MARK_WORDS[0].findall(s)) + sum(s.count(x) for x in _MARK_WORDS[1])

_CHAIN_LT = re.compile(r"(?<![\d(])\d+(?:\s*<\s*\d+)+(?![\d)])")
def asserted_pairs(think: str) -> list[tuple[int, int]]:
    """括号外的 a<b 断言；链 a<b<c 展开为相邻对。"""
    out = []
    for m in _CHAIN_LT.finditer(think):
        vals = [int(x) for x in re.findall(r"\d+", m.group(0))]
        out += list(zip(vals, vals[1:]))
    out += [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s+before\s+(\d+)", think, re.I)]      # 英文 "a before b" ≡ a<b（warm 把 before 映成 <）
    out += [(int(b), int(a)) for a, b in re.findall(r"(\d+)\s+after\s+(\d+)", think, re.I)]
    return out

def longest_chain(think: str, n: int) -> int:
    best = 0
    for m in re.finditer(r"\d+(?:\s*(?:<|→|->)\s*\d+)+", think):
        vals = [int(x) for x in re.findall(r"\d+", m.group(0))]
        if all(0 <= v < n for v in vals): best = max(best, len(dict.fromkeys(vals)))
    return best

def strategy_class(think: str, prompt: str, n: int, S: int, tok=None) -> dict:
    case = _count_role(think, "case_splits"); prop = _count_role(think, "propagation")
    tries = len(re.findall(r"\btry\b", think, re.I)); branches = case + tries
    lf = letter_fraction(think)
    ir_edges = set()
    for it in ir_items(prompt):
        for a, b in re.findall(r"(\d+)<(\d+)", it): ir_edges.add((int(a), int(b)))
    ap = asserted_pairs(think); on_ir = (sum(1 for p in ap if p in ir_edges) / len(ap)) if ap else None
    enum = branches > 3 * (2 ** max(S, 0))
    mixed = enum and (prop >= 1 or bool(_TRYBOTH.search(think)))            # r3：mixed_probe_enum = enum 且（propagation 标记 或 try-both 标记）
    if mixed: cls = "mixed_probe_enum"
    elif enum: cls = "enumeration"
    elif case >= 1 and lf >= 0.3: cls = "english_case"
    elif case >= 1: cls = "symbolic_case"
    elif ap and on_ir is not None and on_ir >= 0.9 and longest_chain(think, n) >= n / 2: cls = "path_stitching"   # 无分情况、只用题面边连路
    elif prop >= 1: cls = "propagation"
    else: cls = "other"
    return dict(strategy_class=cls, case_splits=case, propagation=prop, branches=branches, dpll_min_branches=2 ** max(S, 0), letter_frac=round(lf, 3),
                n_asserted_pairs=len(ap), asserted_on_ir_frac=on_ir, longest_chain=longest_chain(think, n),
                copy_rate=copy_rate(tokens_of(think, tok), tokens_of(prompt, tok)))

CLASSES = ("english_case", "symbolic_case", "propagation", "enumeration", "mixed_probe_enum", "path_stitching", "other")

def class_distribution(annots: list[dict]) -> dict:
    c = Counter(a["strategy_class"] for a in annots); tot = max(1, len(annots))
    return {k: c[k] / tot for k in CLASSES}

def shortcut_rate(thinks, prompts, n, d, legacy_gv: bool = False) -> dict:
    hs = [detect_shortcuts(t, p, n, d, legacy_gv) for t, p in zip(thinks, prompts)]
    c = Counter(h for hh in hs for h in set(hh)); tot = max(1, len(hs))
    return dict(any=sum(1 for hh in hs if hh) / tot, by_type={k: c[k] / tot for k in ("ir_copy_order", "perm_enum", "assign_enum", "guess_verify")}, n=len(hs))
