"""v7 主任务：稀疏 Knights-and-Knaves（IR 题面）。难度键 kk_n{N}_s{S}；主格 kk_n10_s1，备用 kk_n12_s1。
- 人 1..N，knight=1 / knave=0。每人恰一条陈述，宽度固定 2，形式 ∈ FORMS 均匀（x、y 为**其他人**编号，x≠y）。
- 均匀构造式生成 + 拒绝：先抽真值向量；逐条加陈述（与真值一致：knight 的陈述为真，knave 的为假），每人被引用 ≤ MAX_REF 次；
  暴力 2^N 验证唯一解；计算最小分情况深度 S（见 case_split_depth）；只保留 S == 目标值（S=1：单元传播 + failed-literal probing 解不掉）。
- 规范形（canonical_form）：人编号置换不变（颜色细化 + 平局枚举），用于去重与 split 零重叠断言。
- 题面 IR（所有臂相同；语义解释在 SYSTEM_PROMPT 里）：
    persons 1..N (1 = knight, 0 = knave)
    s1: 4 says 3 & !7
    ...
    answer: N digits for persons 1..N
  gloss=True 时把每条陈述改写成英文（题面格式检查用：冻结模型 IR vs 英文原生准确率 ≤ 10pp）。
- 答案 <answer>1 0 1 ... </answer>：N 个空格分隔的 0/1，严格正则、exact match；chance = 2^-N。
- S 的算法（case_split_depth）：
    传播 propagate = 广义单元传播：每条约束 (B_i ⇔ φ_i(B_x,B_y)) 是 3 变量上的允许三元组集合；在当前部分赋值下过滤三元组，
      某变量在全部剩余三元组里取同一值 → 强制赋值；无剩余三元组 → 冲突；迭代到不动点。
    探测 probe = failed-literal probing：对每个未赋值变量 v 和值 b，propagate(assign ∪ {v=b}) 冲突 → 赋 v=¬b 并 propagate；迭代到不动点。
    S(assign) = 0 若 propagate+probe 后完整（或冲突）；否则 S = 1 + min_v max_b S(assign ∪ {v=b})。即最小决策树深度（叶 = 完整/冲突）。
    实现带截断：只需区分 0/…/目标/>目标。
- oracle 下限：最小决策树里（深度最小者中被迫文字数最少的那棵）所有节点新得到的文字数（决策 + 传播 + 探测强制），× c，c∈{2,2.5,3}。
- 每题 meta：truth（唯一解）、S、oracle_literals、oracle_lb、gold_tree（决策树，供解码器标签）、decoder_labels[k]（k=0..S，深度 k 单元传播后的部分赋值，沿正解路径；S=1 时 k=1 = 分情况之后）。
"""
from __future__ import annotations
import itertools, json, pathlib, random, re
from collections import Counter

KEY_RE = re.compile(r"^kk_n(\d+)_s(\d+)$")
FORMS = ("x", "!x", "x&y", "x|y", "x=y", "x!=y", "!x&y", "x&!y", "!x|y")
SYMMETRIC = {"x&y", "x|y", "x=y", "x!=y"}
UNARY = {"x", "!x"}
MAX_REF = 3
MAX_ANSWER_TOKENS = 32
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "kk"
SPLITS = {"train": 2000, "stop": 500, "report": 500}      # stopping-eval / reporting-eval 各 500，不混用

SYSTEM_PROMPT = (
    "You are solving a knights-and-knaves puzzle given in a compact notation. First think inside <think> and </think>, "
    "then give only the final answer inside <answer> and </answer>.\n"
    "Notation: persons are numbered 1..N. Each person is a knight (1, always tells the truth) or a knave (0, always lies). "
    "'p says E' means person p asserts E. In E, a number k means 'person k is a knight'; !k means 'person k is a knave'; "
    "& means and; | means or; = means 'same kind'; != means 'different kinds'. "
    "The answer is N digits (0 or 1) separated by single spaces, for persons 1..N in order, e.g. <answer>1 0 1 1 0</answer>."
)

SYSTEM_PROMPT_GUIDED = (   # direct 模式（<answer> 预填，无思考）：同一套记法说明
    "You are solving a knights-and-knaves puzzle given in a compact notation. Answer directly without working.\n"
    "Notation: persons are numbered 1..N. Each person is a knight (1, always tells the truth) or a knave (0, always lies). "
    "'p says E' means person p asserts E. In E, a number k means 'person k is a knight'; !k means 'person k is a knave'; "
    "& means and; | means or; = means 'same kind'; != means 'different kinds'. "
    "The answer is N digits (0 or 1) separated by single spaces, for persons 1..N in order, inside <answer> and </answer>."
)

def parse_key(key: str) -> dict:
    m = KEY_RE.match(key)
    if not m:
        raise ValueError(f"bad kk difficulty key: {key}")
    return dict(n=int(m.group(1)), s=int(m.group(2)))

# ---------------- semantics ----------------
def eval_form(form: str, bx: int, by: int | None) -> int:
    nx = 1 - bx
    if form == "x": return bx
    if form == "!x": return nx
    ny = 1 - by
    return {"x&y": bx & by, "x|y": bx | by, "x=y": int(bx == by), "x!=y": int(bx != by),
            "!x&y": nx & by, "x&!y": bx & ny, "!x|y": nx | by}[form]

def statement_ok(truth, speaker, form, x, y) -> bool:
    """陈述 (speaker, form, x, y) 与真值向量一致：knight 说真话，knave 说假话。truth 下标 0..N-1；人编号 1..N。"""
    val = eval_form(form, truth[x - 1], None if y is None else truth[y - 1])
    return val == truth[speaker - 1]

def all_solutions(n: int, stmts: list[tuple]) -> list[tuple[int, ...]]:
    """暴力 2^N。stmts = [(speaker, form, x, y)]。"""
    sols = []
    for bits in itertools.product((0, 1), repeat=n):
        if all(statement_ok(bits, s, f, x, y) for s, f, x, y in stmts):
            sols.append(bits)
    return sols

# ---------------- propagation / probing / S ----------------
def _constraints(n: int, stmts: list[tuple]):
    """每条约束 → (vars, allowed): vars = (speaker, x, y)（0 基；一元用 y=None → vars 长 2），allowed = 允许的取值元组集合。"""
    cons = []
    for s, f, x, y in stmts:
        vs = (s - 1, x - 1) if y is None else (s - 1, x - 1, y - 1)
        allowed = set()
        for bits in itertools.product((0, 1), repeat=len(vs)):
            if eval_form(f, bits[1], bits[2] if len(bits) == 3 else None) == bits[0]:
                allowed.add(bits)
        cons.append((vs, tuple(sorted(allowed))))
    return cons

CONFLICT = None

def propagate(cons, assign: list[int]) -> list[int] | None:
    """广义单元传播到不动点；返回新赋值列表或 None（冲突）。assign: -1 未赋值。"""
    a = list(assign)
    changed = True
    while changed:
        changed = False
        for vs, allowed in cons:
            cand = [t for t in allowed if all(a[v] < 0 or a[v] == t[k] for k, v in enumerate(vs))]
            if not cand:
                return CONFLICT
            for k, v in enumerate(vs):
                if a[v] < 0:
                    vals = {t[k] for t in cand}
                    if len(vals) == 1:
                        a[v] = vals.pop(); changed = True
    return a

def probe(cons, assign: list[int]) -> list[int] | None:
    """failed-literal probing 到不动点（含传播）。"""
    a = propagate(cons, assign)
    if a is CONFLICT:
        return CONFLICT
    changed = True
    while changed:
        changed = False
        for v in range(len(a)):
            if a[v] >= 0:
                continue
            for b in (0, 1):
                t = list(a); t[v] = b
                if propagate(cons, t) is CONFLICT:
                    a[v] = 1 - b
                    a = propagate(cons, a)
                    if a is CONFLICT:
                        return CONFLICT
                    changed = True
                    break
    return a

def _complete(a) -> bool:
    return all(v >= 0 for v in a)

def _split_search(cons, a, budget: int, want_tree: bool):
    """返回 (depth | None(>budget), literals, tree)。a 已是 probe 后的状态（非冲突）。
    literals = 该子树内新得到的文字数（不含进入 a 前的）。tree = dict（want_tree 时）。"""
    if _complete(a):
        return 0, 0, (dict(assign=list(a), split=None) if want_tree else None)
    if budget == 0:
        return None, None, None
    best = None
    for v in range(len(a)):
        if a[v] >= 0:
            continue
        kids, depth_v, lits_v, ok = {}, 0, 0, True
        for b in (0, 1):
            t = list(a); t[v] = b
            child = probe(cons, t)
            if child is CONFLICT:
                kids[b] = dict(assign=None, split=None, conflict=True) if want_tree else None
                lits_v += 1                       # 决策文字本身；冲突叶不再计
                continue
            d, l, tr = _split_search(cons, child, budget - 1, want_tree)
            if d is None:
                ok = False; break
            new = sum(1 for i in range(len(a)) if a[i] < 0 and child[i] >= 0)
            depth_v = max(depth_v, d); lits_v += new + l; kids[b] = tr
        if not ok:
            continue
        cand = (1 + depth_v, lits_v)
        if best is None or cand < best[:2]:
            best = (cand[0], cand[1], (dict(assign=list(a), split=v, children=kids) if want_tree else None))
            if want_tree:
                best[2]["children"] = kids
    if best is None:
        return None, None, None
    return best

def case_split_depth(n: int, stmts: list[tuple], max_depth: int = 3, want_tree: bool = False) -> dict:
    """返回 dict(S=深度或 None(>max_depth), root_literals, oracle_literals, tree)。"""
    cons = _constraints(n, stmts)
    a0 = probe(cons, [-1] * n)
    if a0 is CONFLICT:
        return dict(S=None, unsat=True)
    root_lits = sum(1 for v in a0 if v >= 0)
    d, l, tree = _split_search(cons, a0, max_depth, want_tree)
    if d is None:
        return dict(S=None, root_literals=root_lits, oracle_literals=None, tree=None)
    return dict(S=d, root_literals=root_lits, oracle_literals=root_lits + l, tree=tree)

def decoder_labels(n: int, stmts: list[tuple], tree: dict, truth) -> list[str]:
    """沿正解路径，深度 k=0..S 处**单元传播后**（不含探测）的部分赋值，'0'/'1'/'?' 串。"""
    cons = _constraints(n, stmts)
    labels = []
    node = tree; a = propagate(cons, [-1] * n)
    labels.append("".join("?" if v < 0 else str(v) for v in a))
    while node is not None and node.get("split") is not None:
        v = node["split"]; b = truth[v]
        t = list(node["assign"]); t[v] = b
        a = propagate(cons, t)
        labels.append("".join("?" if x < 0 else str(x) for x in a))
        node = node["children"][b]
    return labels

# ---------------- canonical form ----------------
def _stmt_key(form, x, y):
    if form in SYMMETRIC and y is not None and x > y:
        x, y = y, x
    return (form, x, y)

def _h(obj) -> str:
    import hashlib
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def canonical_form(n: int, stmts: list[tuple], max_tie_perms: int = 5000) -> str:
    """人编号置换不变的规范形：颜色细化（稳定 sha1 颜色：说话形式 + 引用对象颜色 + 被引用情况）到不动点；
    类按颜色串排序（颜色本身置换不变，故类序确定）；同色类内枚举置换取字典序最小的陈述表。
    平局置换数 > max_tie_perms 时退化为颜色多重集签名（近似；返回值带 '~' 前缀）。"""
    by_speaker = {s: (f, x, y) for s, f, x, y in stmts}
    color = {p: _h(by_speaker[p][0]) for p in range(1, n + 1)}
    for _ in range(n + 1):
        new = {}
        for p in range(1, n + 1):
            f, x, y = by_speaker[p]
            refs = sorted((f2, (k if f2 not in SYMMETRIC else 0), color[s2]) for s2, (f2, x2, y2) in by_speaker.items() for k, z in enumerate((x2, y2)) if z == p)
            cx, cy = color[x], (None if y is None else color[y])
            if f in SYMMETRIC and cy is not None and cy < cx:
                cx, cy = cy, cx
            new[p] = _h([f, cx, cy, refs])
        stable = len(set(new.values())) == len(set(color.values()))
        color = new
        if stable:
            break
    classes = {}
    for p in range(1, n + 1):
        classes.setdefault(color[p], []).append(p)
    order_classes = [classes[c] for c in sorted(classes)]
    n_perms = 1
    for ps in order_classes:
        for i in range(2, len(ps) + 1): n_perms *= i
    if n_perms > max_tie_perms:
        return "~" + json.dumps(sorted(color.values()))
    best = None
    for perm in itertools.product(*[itertools.permutations(ps) for ps in order_classes]):
        flat = [p for grp in perm for p in grp]
        relabel = {old: new for new, old in enumerate(flat, 1)}
        rows = sorted((relabel[s], *_stmt_key(f, relabel[x], None if y is None else relabel[y])) for s, (f, x, y) in by_speaker.items())
        s_ = json.dumps(rows)
        if best is None or s_ < best:
            best = s_
    return best

# ---------------- generation ----------------
# 均匀构造式采样 + 拒绝（2026-09-18 定稿）：先抽真值；按随机顺序每人一条陈述，形式在 FORMS 上均匀、x/y 在"其他人且被引用 < MAX_REF"里均匀，
# 与真值一致；暴力 2^N 验证唯一解；只保留 case_split_depth == 目标 S（主格 S=1：单元传播 + failed-literal probing 解不掉、一次分情况即可）。
# 实测 N=10 唯一解题里 S=1 约 0.2–0.4%（每题几百次尝试，约 1–2 s/题/核）；S=2 在均匀采样下接受率 < 0.03%，不用作数据集。
def _pick_statement(rng, speaker, truth, pool, refs, tries=30):
    others = [p for p in pool if p != speaker and refs[p] < MAX_REF]
    for _ in range(tries):
        form = rng.choice(FORMS)
        if form in UNARY:
            if not others: return None
            x, y = rng.choice(others), None
        else:
            if len(others) < 2: return None
            x, y = rng.sample(others, 2)
        if statement_ok(truth, speaker, form, x, y):
            return (speaker, form, x, y)
    return None

def sample_raw(rng: random.Random, n: int, max_attempts: int = 200):
    """构造一题（未过 S 筛选）：返回 (truth, stmts) 或 None。唯一解已验证。"""
    persons = list(range(1, n + 1))
    for _ in range(max_attempts):
        truth = tuple(rng.randint(0, 1) for _ in range(n))
        refs = Counter(); stmts = []
        for speaker in rng.sample(persons, n):
            st = _pick_statement(rng, speaker, truth, persons, refs)
            if st is None:
                break
            stmts.append(st); refs[st[2]] += 1
            if st[3] is not None: refs[st[3]] += 1
        else:
            if len(all_solutions(n, stmts)) == 1:
                return truth, sorted(stmts)
    return None

def generate(rng: random.Random, key: str, want_tree: bool = True) -> dict:
    """拒绝采样直到 S == 目标。"""
    p = parse_key(key); n, target = p["n"], p["s"]
    while True:
        raw = sample_raw(rng, n)
        if raw is None:
            continue
        truth, stmts = raw
        info = case_split_depth(n, stmts, max_depth=target, want_tree=want_tree)
        if info.get("S") == target:
            return make_item(rng, key, truth, stmts, info)

def make_item(rng: random.Random, key: str, truth, stmts, info: dict) -> dict:
    n = len(truth)
    order = list(range(len(stmts))); rng.shuffle(order)                     # 说话人（行）顺序随机
    shown = []
    for i in order:
        s, f, x, y = stmts[i]
        if f in SYMMETRIC and rng.random() < 0.5:                             # 对称形式的操作数顺序随机
            x, y = y, x
        shown.append((s, f, x, y))
    labels = decoder_labels(n, stmts, info["tree"], truth) if info.get("tree") else None
    return dict(
        prompt=build_prompt(n, shown), prompt_gloss=build_prompt(n, shown, gloss=True), answer=" ".join(map(str, truth)),
        meta=dict(task="kk", key=key, n=n, stmts=[list(s) for s in stmts], shown=[list(s) for s in shown], truth=list(truth),
                  S=info["S"], root_literals=info["root_literals"], oracle_literals=info["oracle_literals"],
                  oracle_lb={str(c): info["oracle_literals"] * c for c in (2, 2.5, 3)},
                  gold_tree=info.get("tree"), decoder_labels=labels, canonical=canonical_form(n, stmts)))

# ---------------- rendering ----------------
def render_expr(form: str, x: int, y: int | None) -> str:
    return {"x": f"{x}", "!x": f"!{x}", "x&y": f"{x} & {y}", "x|y": f"{x} | {y}", "x=y": f"{x} = {y}", "x!=y": f"{x} != {y}",
            "!x&y": f"!{x} & {y}", "x&!y": f"{x} & !{y}", "!x|y": f"!{x} | {y}"}[form]

def gloss_expr(form: str, x: int, y: int | None) -> str:
    K = lambda k: f"person {k} is a knight"; V = lambda k: f"person {k} is a knave"
    return {"x": K(x), "!x": V(x), "x&y": f"{K(x)} and {K(y)}", "x|y": f"{K(x)} or {K(y)}",
            "x=y": f"person {x} and person {y} are the same kind", "x!=y": f"person {x} and person {y} are different kinds",
            "!x&y": f"{V(x)} and {K(y)}", "x&!y": f"{K(x)} and {V(y)}", "!x|y": f"{V(x)} or {K(y)}"}[form]

def build_prompt(n: int, shown: list[tuple], gloss: bool = False) -> str:
    if gloss:
        lines = [f"There are {n} persons, numbered 1 to {n}. Each is a knight (always tells the truth) or a knave (always lies)."]
        lines += [f"Person {s} says: {gloss_expr(f, x, y)}." for s, f, x, y in shown]
        lines.append(f"Answer with {n} digits (1 = knight, 0 = knave) for persons 1 to {n}, separated by single spaces.")
        return "\n".join(lines)
    lines = [f"persons 1..{n} (1 = knight, 0 = knave)"]
    lines += [f"s{i + 1}: {s} says {render_expr(f, x, y)}" for i, (s, f, x, y) in enumerate(shown)]
    lines.append(f"answer: {n} digits for persons 1..{n}")
    return "\n".join(lines)

# ---------------- answers ----------------
def normalize(ans: str) -> str | None:
    toks = ans.strip().split()
    if not toks or any(t not in ("0", "1") for t in toks):
        return None
    return " ".join(toks)

def is_valid(ans: str) -> bool:
    return normalize(ans) is not None

def check(pred: str, gold: str) -> bool:
    p = normalize(pred)
    return p is not None and p == normalize(gold)

def hamming(pred: str, gold: str) -> int | None:
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None or len(p) != len(g):
        return None
    return sum(a != b for a, b in zip(p.split(), g.split()))

_LENIENT = re.compile(r"(?<![0-9])([01](?:[ ,]+[01]){3,})(?![0-9])")
def lenient_extract(text: str) -> str | None:
    m = _LENIENT.findall(text)
    return " ".join(re.split(r"[ ,]+", m[-1])) if m else None

def chance(key: str) -> float:
    return 2.0 ** (-parse_key(key)["n"])

# ---------------- splits ----------------
def item_rng(seed: int, key: str, split: str, j: int) -> random.Random:
    """第 j 个候选的独立随机流：与进程调度无关，可并行生成且确定。"""
    return random.Random(f"cot-compress|kk|{seed}|{key}|{split}|{j}")

def generate_indexed(args):
    key, split, seed, j = args
    it = generate(item_rng(seed, key, split, j), key)
    it["_j"] = j
    return it

def build_splits(key: str, seed: int = 0, sizes: dict | None = None, log=print, workers: int | None = None) -> dict:
    """三个划分：候选流按 (seed, key, split, j) 独立播种（split 名互不相同 → 流互不相交）；
    规范形去重：先 stop / report，再 train；与已见规范形撞形的候选丢弃；最后断言零重叠。workers>1 用进程池。"""
    import multiprocessing as mp
    sizes = dict(SPLITS, **(sizes or {}))
    out, seen = {}, set()
    for split in ("stop", "report", "train"):
        items, forms, j = [], set(), 0
        while len(items) < sizes[split]:
            batch = [(key, split, seed, j + t) for t in range(max(8, int((sizes[split] - len(items)) * 1.05) + 1))]; j += len(batch)
            if workers and workers > 1:
                with mp.Pool(workers) as pool:
                    cands = pool.map(generate_indexed, batch, chunksize=4)
            else:
                cands = [generate_indexed(a) for a in batch]
            for it in sorted(cands, key=lambda x: x["_j"]):
                cf = it["meta"]["canonical"]
                if cf in seen or cf in forms or len(items) >= sizes[split]:
                    continue
                forms.add(cf); del it["_j"]; it["id"] = f"{key}/{split}/{seed}/{len(items)}"; items.append(it)
        seen |= forms; out[split] = items
        if log: log(f"[kk] {key} seed={seed} split={split}: {len(items)} items, {len(forms)} distinct canonical forms (candidates drawn: {j})")
    assert_disjoint(out)
    return out

def assert_disjoint(splits: dict):
    forms = {k: {it["meta"]["canonical"] for it in v} for k, v in splits.items()}
    for a, b in itertools.combinations(forms, 2):
        assert not (forms[a] & forms[b]), f"canonical-form overlap between {a} and {b}"
    prompts = {k: {it["prompt"] for it in v} for k, v in splits.items()}
    for a, b in itertools.combinations(prompts, 2):
        assert not (prompts[a] & prompts[b])

def split_path(key: str, seed: int) -> pathlib.Path:
    return DATA_DIR / f"{key}_seed{seed}.json"

_CACHE = {}
def load_splits(key: str, seed: int = 0, build: bool = False, workers: int | None = None) -> dict:
    """读 data/kk/<key>_seed<seed>.json；不存在且 build=False → 报错提示先跑 scripts/kk_build_splits.py。"""
    if (key, seed) in _CACHE:
        return _CACHE[(key, seed)]
    p = split_path(key, seed)
    if p.exists():
        sp = json.load(open(p))
    elif not build:
        raise FileNotFoundError(f"{p} missing: run `python scripts/kk_build_splits.py --key {key} --seed {seed}` first")
    else:
        sp = build_splits(key, seed, workers=workers)
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(sp, open(p, "w"))
    _CACHE[(key, seed)] = sp
    return sp

def split_report(splits: dict) -> dict:
    rep = {}
    for k, v in splits.items():
        forms = Counter(f for it in v for _, f, _, _ in it["meta"]["stmts"])
        ol = [it["meta"]["oracle_literals"] for it in v]
        rep[k] = dict(n=len(v), distinct_canonical=len({it["meta"]["canonical"] for it in v}),
                      S=dict(Counter(it["meta"]["S"] for it in v)), oracle_literals_mean=sum(ol) / max(1, len(ol)),
                      oracle_literals_hist=dict(sorted(Counter(ol).items())),
                      root_literals_hist=dict(sorted(Counter(it["meta"]["root_literals"] for it in v).items())),
                      form_frac={f: round(c / max(1, sum(forms.values())), 3) for f, c in sorted(forms.items())})
    return rep
