"""合成状态跟踪任务：n 个编号杯子排成一行，k 步 swap(i,j) / flip(i)，问**完整终态**。
难度键: cups_n{n}_k{k}，如 cups_n8_k20。答案形如 "6D 4U 8D 5D 3D 1D 2D 7D"（位置 1–n 顺序，杯号 + U/D，空格分隔），exact match。
2026-09-18 由单点查询改为完整终态：单点查询允许"只反向追踪查询位置"的捷径，长度惩罚下压缩的主要来源会变成换算法而不是换编码。
chance = 1/(n!·2^n) ≈ 0。难度只靠 n、k 调。
2026-09-16 去掉 move（带顺移）操作：语义易歧义且是最大的 token 黑洞（基线每步约 700 token）。
meta.states 保存每步之后的完整状态（线性探针用）。
"""
from __future__ import annotations
import random, re

KEY_RE = re.compile(r"^cups_n(\d+)_k(\d+)$")
ANSWER_RE = re.compile(r"^(\d+)-(up|down)$")          # 旧单点格式（保留给历史数据解析）
STATE_TOK_RE = re.compile(r"^(\d+)([UD])$")
FULL_STATE_RE = re.compile(r"^\d+[UD](?: \d+[UD])*$")
OPS = ("swap", "flip")

SYSTEM_PROMPT = (
    "You are solving a cup-tracking puzzle. First think inside <think> and </think>, "
    "then give only the final answer inside <answer> and </answer>. "
    "The answer is the final state of every position in order: for each position write the cup number followed by "
    "U (facing up) or D (facing down), separated by single spaces, e.g. <answer>6D 4U 8D 5D 3D 1D 2D 7D</answer>."
)
SYSTEM_PROMPT_GUIDED = (  # (b) guided / direct：关原生思考，思考段 = <answer> 之前的文本
    "You are solving a cup-tracking puzzle. Show your step-by-step working in plain text first; "
    "do not answer directly. After the working, give the final answer inside <answer> and </answer>. No LaTeX or markup. "
    "The answer is the final state of every position in order: for each position write the cup number followed by "
    "U (facing up) or D (facing down), separated by single spaces, e.g. <answer>6D 4U 8D 5D 3D 1D 2D 7D</answer>."
)

def parse_key(key: str) -> dict:
    m = KEY_RE.match(key)
    if not m:
        raise ValueError(f"bad cups difficulty key: {key}")
    return dict(n=int(m.group(1)), k=int(m.group(2)))

def apply_op(state: list[tuple[int, bool]], op: tuple) -> list[tuple[int, bool]]:
    """state[i] = (cup_id, up) 表示位置 i+1。返回新状态。"""
    s = list(state)
    kind = op[0]
    if kind == "swap":
        a, b = op[1] - 1, op[2] - 1
        s[a], s[b] = s[b], s[a]
    elif kind == "flip":
        a = op[1] - 1
        s[a] = (s[a][0], not s[a][1])
    else:
        raise ValueError(kind)
    return s

def simulate(n: int, ops: list[tuple]) -> list[list[tuple[int, bool]]]:
    """返回每步之后的状态列表（长度 k）。初始：位置 i 放杯子 i，全部朝上。"""
    s = [(i + 1, True) for i in range(n)]
    states = []
    for op in ops:
        s = apply_op(s, op)
        states.append(s)
    return states

def _rand_op(rng: random.Random, n: int) -> tuple:
    kind = rng.choice(OPS)
    if kind == "flip":
        return ("flip", rng.randint(1, n))
    a, b = rng.sample(range(1, n + 1), 2)
    return (kind, a, b)

def format_op(op: tuple) -> str:
    return " ".join(str(x) for x in op)

def state_str(state) -> str:
    return " ".join(f"{c}{'U' if u else 'D'}" for c, u in state)

def build_prompt(n: int, ops: list[tuple], query_pos: int | None = None) -> str:
    lines = [
        f"There are {n} cups in a row at positions 1 to {n}. At the start, position i holds cup number i, "
        f"and every cup is facing up.",
        '"swap A B" exchanges the two cups at positions A and B (each keeps its own facing); '
        '"flip A" turns over the cup at position A (up becomes down, down becomes up).',
        f"Apply the following {len(ops)} operations in order:",
    ]
    lines += [f"{i + 1}. {format_op(op)}" for i, op in enumerate(ops)]
    lines.append(f"Question: after all operations, give the final state of all {n} positions in order (position 1 to {n}): "
                 "for each position write the cup number followed by U (facing up) or D (facing down), separated by single spaces.")
    ex = state_str([(n, False), (1, True)] + [(i, True) for i in range(2, n)])
    lines.append(f"Answer as <answer>{ex}</answer> (this example is just the format, not the answer).")
    return "\n".join(lines)

def generate(rng: random.Random, key: str) -> dict:
    p = parse_key(key)
    n, k = p["n"], p["k"]
    ops = [_rand_op(rng, n) for _ in range(k)]
    states = simulate(n, ops)
    q = rng.randint(1, n)          # 保留 query_pos（诊断 5 的前缀评分仍按查询位置比对）
    answer = state_str(states[-1])
    return dict(
        prompt=build_prompt(n, ops, q), answer=answer,
        meta=dict(task="cups", key=key, n=n, k=k, ops=ops, query_pos=q,
                  states=[[(c, int(u)) for c, u in s] for s in states]),
    )

def normalize(ans: str) -> str | None:
    """完整终态："6D 4U 8D …"（大小写、多空格容忍）；每个 token 必须是 数字+U/D。不合法 → None。"""
    toks = ans.strip().upper().split()
    if not toks or not all(STATE_TOK_RE.match(t) for t in toks):
        return None
    return " ".join(toks)

def is_valid(ans: str) -> bool:
    return normalize(ans) is not None

def check(pred: str, gold: str) -> bool:
    p = normalize(pred)
    return p is not None and p == normalize(gold)          # exact match（位置顺序、杯号、朝向全对）

_LENIENT_TOK = re.compile(r"\b(\d+)\s*([UDud])\b")
def lenient_extract(text: str) -> str | None:
    """宽松提取：文本中最后一段连续（≥2 个）"数字+U/D" token。仅用于诊断，不用于奖励。"""
    runs, cur, last_end = [], [], None
    for m in _LENIENT_TOK.finditer(text):
        if last_end is not None and text[last_end:m.start()].strip() not in ("", ","):
            if len(cur) >= 2: runs.append(cur)
            cur = []
        cur.append(f"{m.group(1)}{m.group(2).upper()}"); last_end = m.end()
    if len(cur) >= 2: runs.append(cur)
    return " ".join(runs[-1]) if runs else None

def chance(key: str) -> float:
    """随机猜完整终态的准确率：1/(n!·2^n)。"""
    import math
    p = parse_key(key)
    return 1.0 / (math.factorial(p["n"]) * 2 ** p["n"])
