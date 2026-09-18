"""多位数乘法 / 加减链。
难度键:
  mul_{a}x{b}   a 位数 × b 位数，如 mul_4x4
  add_{m}x{d}   m 个 d 位数的加减链（首项为正，其余随机 +/-），如 add_6x4
答案为整数（允许负号），仅数字。
"""
from __future__ import annotations
import random, re

KEY_RE = re.compile(r"^(mul|add)_(\d+)x(\d+)$")
ANSWER_RE = re.compile(r"^-?\d+$")

SYSTEM_PROMPT = (
    "You are solving an arithmetic problem. First think inside <think> and </think>, "
    "then give only the final answer inside <answer> and </answer>. "
    "The answer must be an integer written with digits only (a leading minus sign is allowed; "
    "no commas, spaces or other symbols), e.g. <answer>1887257</answer>."
)
SYSTEM_PROMPT_GUIDED = (  # (b) guided / direct：关原生思考，思考段 = <answer> 之前的文本
    "You are solving an arithmetic problem. Show your step-by-step working in plain text first; "
    "do not answer directly. After the working, give the final answer inside <answer> and </answer>. No LaTeX or markup. "
    "The answer must be an integer written with digits only (a leading minus sign is allowed; "
    "no commas, spaces or other symbols), e.g. <answer>1887257</answer>."
)

def parse_key(key: str) -> dict:
    m = KEY_RE.match(key)
    if not m:
        raise ValueError(f"bad arith difficulty key: {key}")
    return dict(kind=m.group(1), a=int(m.group(2)), b=int(m.group(3)))

def _rand_digits(rng: random.Random, d: int) -> int:
    return rng.randint(10 ** (d - 1), 10 ** d - 1)

def generate(rng: random.Random, key: str) -> dict:
    p = parse_key(key)
    if p["kind"] == "mul":
        x, y = _rand_digits(rng, p["a"]), _rand_digits(rng, p["b"])
        expr = f"{x} * {y}"
        ans = x * y
        meta = dict(task="arith", key=key, kind="mul", x=x, y=y)
    else:
        m, d = p["a"], p["b"]
        terms = [_rand_digits(rng, d) for _ in range(m)]
        signs = [1] + [rng.choice((1, -1)) for _ in range(m - 1)]
        expr = str(terms[0])
        for t, s in zip(terms[1:], signs[1:]):
            expr += f" {'+' if s > 0 else '-'} {t}"
        ans = sum(t * s for t, s in zip(terms, signs))
        meta = dict(task="arith", key=key, kind="add", terms=terms, signs=signs)
    prompt = (f"Compute {expr}.\n"
              "Answer as <answer>INTEGER</answer> using digits only (no commas, spaces or other symbols; "
              "a leading minus sign is allowed).")
    return dict(prompt=prompt, answer=str(ans), meta=meta)

def normalize(ans: str) -> str | None:
    s = ans.strip()
    if not ANSWER_RE.match(s):
        return None
    return str(int(s))  # 去前导零

def is_valid(ans: str) -> bool:
    return normalize(ans) is not None

def check(pred: str, gold: str) -> bool:
    p = normalize(pred)
    return p is not None and p == gold

_LENIENT_RE = re.compile(r"-?\d[\d,]*")
def lenient_extract(text: str) -> str | None:
    m = _LENIENT_RE.findall(text)
    if not m:
        return None
    s = m[-1].replace(",", "")
    try:
        return str(int(s))
    except ValueError:
        return None

def chance(key: str) -> float:
    """随机猜整数答案：≈0。"""
    return 0.0
