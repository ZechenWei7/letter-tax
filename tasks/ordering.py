"""v8 主任务：唯一线性序 + 析取先后约束（ordering）。难度键 ord_n{n}_h{h}_d{d}，候选格顺序 (8,4,5) (8,4,6) (9,5,6) (9,5,7)。
- n 个事件 0..n−1；硬约束 i<j（i 在 j 前）h 条；析取 (i<j)|(k<l) d 条，每条一边与真实顺序 σ 一致、另一边不一致。
- 生成：先抽 σ；硬约束在 σ 中成立的对里均匀抽 h 条；析取逐条加入；暴力枚举验证恰有一个线性序；拒绝其他。
  析取抽法（GEN_MODE，见 generation 段注释）："guided"（默认）：一致边只从 σ 的未被硬约束蕴含的覆盖边中抽，不一致边在所有与 σ 矛盾的对里均匀抽；
  "uniform"：都均匀（纯均匀拒绝在候选格上唯一解率 ≈ 0，(8,4,5) 10 万次 1 个）。
- 决策深度 S：DPLL，传播器 = 传递闭包（硬边 ∪ 已选边）+ 单元规则（一边被闭包否定 → 另一边被迫）；failed-disjunct probing（单独试每一边，出环 → 另一边被迫）；
  叶 = 闭包为全序 或 冲突；S = 最小决策树深度（截断 MAX_S）。只保留 S ≥ 1。
- 下限：决策下限 = d + n 个元素；Kahn 下限 = 就绪集轨迹（状态 = 前缀 + 剩余集合，就绪集经 failed-element probing 剪枝）沿 σ 的就绪集元素数之和；各 × c ∈ {2,2.5,3}。
- 每格统计（cell_stats / cell_decision）：#LE(hard) 中位 ≥20；闭包传播下第一次非单元素就绪集的位置中位 ≤ n/2（单元素步占比只报）；硬偏序最大反链中位 ≥3；无向被提及图哈密顿路径数中位 ≥3；
  probing 后存活析取数中位 ≥3；S≥1 保留率 ≥1%；全结构同构类 eval ⊄ train；S 分布；析取冗余率（去掉该条仍唯一的比例）。
- 划分：train 2000 / stop 500 / report 500 / direct 2000（direct-check），seed 流互不相交；每题事件随机重编号、约束与析取两边顺序随机。规范形去重（同构）。
- IR 题面（所有臂相同）：
    n=8
    hard: 0<3 2<5 ...
    disj: (1<6)|(6<2) ...
    reason inside <think>, then give the order
    answer: n digits, first to last
- 答案 <answer>3 0 5 ...</answer>：n 个空格分隔的事件号（每个 0..n−1 恰一次），exact match，≤32 token；chance = 1/#LE(hard)。lenient_extract 报宽松抽取。
- 每题存：sigma、S、decision_lb / kahn_lb（元素数与 token）、canonical_path（最低编号未决析取优先、先第一边的 DPLL 路径）、steps[t]（t=0..n：就绪集、下一事件、已定析取集）。
"""
from __future__ import annotations
import itertools, json, pathlib, random, re
from collections import Counter

KEY_RE = re.compile(r"^ord_n(\d+)_h(\d+)_d(\d+)$")
CELLS = ("ord_n8_h4_d5", "ord_n8_h4_d6", "ord_n9_h5_d6", "ord_n9_h5_d7")     # (10,6,7) 已删（用户 2026-09-18）
GEN_MODE = "guided"
MAX_S = 4
MAX_ANSWER_TOKENS = 32
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ordering"
SPLITS = {"train": 2000, "stop": 500, "report": 500, "direct": 2000}

SYSTEM_PROMPT = (
    "You are solving an ordering puzzle given in a compact notation. First think inside <think> and </think>, "
    "then give only the final answer inside <answer> and </answer>.\n"
    "Notation: there are n events numbered 0..n-1 that happen in some order, one at a time. 'a<b' means event a happens before event b. "
    "Lines after 'hard:' are constraints that all hold. Each item after 'disj:' is a disjunction: at least one of its two sides holds. "
    "Exactly one order satisfies everything. The answer is the n event numbers from first to last, separated by single spaces, e.g. <answer>3 0 5 1 2 4 7 6</answer>."
)
SYSTEM_PROMPT_GUIDED = (
    "You are solving an ordering puzzle given in a compact notation. Answer directly without working.\n"
    "Notation: there are n events numbered 0..n-1 that happen in some order. 'a<b' means event a happens before event b. "
    "Lines after 'hard:' are constraints that all hold. Each item after 'disj:' is a disjunction: at least one of its two sides holds. "
    "Exactly one order satisfies everything. The answer is the n event numbers from first to last, separated by single spaces, inside <answer> and </answer>."
)

def parse_key(key: str) -> dict:
    m = KEY_RE.match(key)
    if not m:
        raise ValueError(f"bad ordering difficulty key: {key}")
    return dict(n=int(m.group(1)), h=int(m.group(2)), d=int(m.group(3)))

# ---------------- linear extensions ----------------
def _pred_masks(n, pairs):
    pred = [0] * n
    for i, j in pairs: pred[j] |= 1 << i
    return pred

def linear_extensions(n: int, hard: list[tuple]) -> list[list[int]]:
    pred = _pred_masks(n, hard); out = []
    def rec(mask, seq):
        if mask == (1 << n) - 1: out.append(seq); return
        for e in range(n):
            if mask >> e & 1 or pred[e] & ~mask: continue
            rec(mask | 1 << e, seq + [e])
    rec(0, []); return out

def count_le_hard(n: int, hard: list[tuple]) -> int:
    """#LE(hard)：子集 DP。"""
    pred = _pred_masks(n, hard); full = (1 << n) - 1
    dp = [0] * (1 << n); dp[0] = 1
    for mask in range(1 << n):
        if not dp[mask]: continue
        for e in range(n):
            if mask >> e & 1 or pred[e] & ~mask: continue
            dp[mask | 1 << e] += dp[mask]
    return dp[full]

def count_solutions(n: int, hard: list[tuple], disj: list[tuple], limit: int = 2) -> int:
    """满足硬约束 + 全部析取的线性序个数（DFS + 子句剪枝，数到 limit 为止）。"""
    pred = _pred_masks(n, hard); cnt = 0
    def rec(mask, pos):
        nonlocal cnt
        if cnt >= limit: return
        if mask == (1 << n) - 1: cnt += 1; return
        for e in range(n):
            if mask >> e & 1 or pred[e] & ~mask: continue
            pos[e] = bin(mask).count("1")
            ok = True
            for (a, b), (c, dd) in disj:
                s1 = a in pos and b in pos and pos[a] > pos[b]
                s2 = c in pos and dd in pos and pos[c] > pos[dd]
                if s1 and s2: ok = False; break
            if ok: rec(mask | 1 << e, pos)
            del pos[e]
    rec(0, {}); return cnt

def solutions(n, hard, disj, limit=2):
    """返回满足全部约束的线性序列表（最多 limit 个）。"""
    pred = _pred_masks(n, hard); out = []
    def rec(mask, seq, pos):
        if len(out) >= limit: return
        if mask == (1 << n) - 1: out.append(list(seq)); return
        for e in range(n):
            if mask >> e & 1 or pred[e] & ~mask: continue
            pos[e] = len(seq)
            if all(not ((a in pos and b in pos and pos[a] > pos[b]) and (c in pos and dd in pos and pos[c] > pos[dd])) for (a, b), (c, dd) in disj):
                rec(mask | 1 << e, seq + [e], pos)
            del pos[e]
    rec(0, [], {}); return out

# ---------------- closure / propagation / probing / DPLL ----------------
def closure(n: int, pairs) -> list[int] | None:
    """after[i] = 必须在 i 之后的集合（bitmask）。出环 → None。"""
    after = [0] * n
    for i, j in pairs: after[i] |= 1 << j
    for k in range(n):
        bk = 1 << k
        for i in range(n):
            if after[i] & bk: after[i] |= after[k]
    for i in range(n):
        if after[i] >> i & 1: return None
    return after

def _implied(after, i, j): return bool(after[i] >> j & 1)

def propagate(n: int, hard, disj, chosen: dict) -> tuple[list[int] | None, dict]:
    """chosen: {析取下标: 0/1 已选边}。单元规则到不动点。返回 (after 或 None(冲突), chosen')。"""
    ch = dict(chosen)
    while True:
        pairs = list(hard) + [disj[k][s] for k, s in ch.items()]
        after = closure(n, pairs)
        if after is None: return None, ch
        changed = False
        for k, ((a, b), (c, d)) in enumerate(disj):
            if k in ch: continue
            f1 = _implied(after, b, a); f2 = _implied(after, d, c)          # 边被闭包否定
            t1 = _implied(after, a, b); t2 = _implied(after, c, d)
            if f1 and f2: return None, ch
            if t1 or t2:
                ch[k] = 0 if t1 else 1; changed = True; continue             # 已被蕴含 → 视为已定
            if f1: ch[k] = 1; changed = True
            elif f2: ch[k] = 0; changed = True
        if not changed: return after, ch

def probe(n: int, hard, disj, chosen: dict):
    """failed-disjunct probing 到不动点：对每条未决析取**单独**试每一边（只加这一边、算闭包，不做进一步单元传播）；出环 → 另一边被迫，然后传播。"""
    after, ch = propagate(n, hard, disj, chosen)
    if after is None: return None, ch
    changed = True
    while changed:
        changed = False
        base = list(hard) + [disj[k][s] for k, s in ch.items()]
        for k in range(len(disj)):
            if k in ch: continue
            for s in (0, 1):
                if closure(n, base + [disj[k][s]]) is None:
                    ch[k] = 1 - s
                    after, ch = propagate(n, hard, disj, ch)
                    if after is None: return None, ch
                    changed = True; break
            if changed: break
    return after, ch

def is_total(n: int, after) -> bool:
    return sum(bin(a).count("1") for a in after) == n * (n - 1) // 2

def _dpll_depth(n, hard, disj, after, ch, budget):
    if is_total(n, after): return 0
    if budget == 0: return None
    best = None
    for k in range(len(disj)):
        if k in ch: continue
        worst = 0
        for s in (0, 1):
            t = dict(ch); t[k] = s
            a2, c2 = probe(n, hard, disj, t)
            if a2 is None: continue
            r = _dpll_depth(n, hard, disj, a2, c2, budget - 1)
            if r is None: worst = None; break
            worst = max(worst, r)
        if worst is not None and (best is None or worst + 1 < best):
            best = worst + 1
            if best == 1: return 1
    return best

def decision_depth(n, hard, disj, max_depth: int = MAX_S) -> dict:
    after, ch = probe(n, hard, disj, {})
    if after is None: return dict(S=None, unsat=True)
    surviving = [k for k in range(len(disj)) if k not in ch]
    S = _dpll_depth(n, hard, disj, after, ch, max_depth)
    return dict(S=S, surviving_disj=len(surviving), root_chosen=len(ch), root_total=is_total(n, after))

def canonical_path(n, hard, disj, sigma) -> list[dict]:
    """DPLL 规范路径：最低编号未决析取优先、先第一边；记录每次决策与结果（conflict / ok），直到全序。"""
    pos = {e: i for i, e in enumerate(sigma)}
    path = []
    after, ch = probe(n, hard, disj, {})
    while after is not None and not is_total(n, after):
        k = min(i for i in range(len(disj)) if i not in ch)
        for s in (0, 1):
            t = dict(ch); t[k] = s
            a2, c2 = probe(n, hard, disj, t)
            if a2 is None:
                path.append(dict(disj=k, side=s, outcome="conflict")); continue
            (a, b) = disj[k][s]
            if pos[a] < pos[b]:                       # 与 σ 一致的边才是正解路径
                path.append(dict(disj=k, side=s, outcome="ok")); after, ch = a2, c2; break
            path.append(dict(disj=k, side=s, outcome="refuted_deeper"))       # 这一边在更深处才被否定（S≥2）；沿正解继续
        else:
            break
    return path

# ---------------- Kahn trajectory ----------------
def kahn_trajectory(n, hard, disj, sigma) -> list[dict]:
    """沿 σ 的就绪集轨迹：每步就绪集（硬边 ∪ 已定边闭包下无未放前驱；再经 failed-element probing（单独试、只算闭包）剪枝）、下一事件、已定析取集。"""
    steps = []; ch = {}
    for t in range(n):
        placed = sigma[:t]; remaining = [e for e in range(n) if e not in placed]
        prefix_pairs = [(p, r) for p in placed for r in remaining]
        after, ch = probe(n, list(hard) + prefix_pairs, disj, ch)
        assert after is not None
        ready = [e for e in remaining if not any(_implied(after, r, e) for r in remaining if r != e)]
        base = list(hard) + prefix_pairs + [disj[k][s_] for k, s_ in ch.items()]
        pruned = []
        for e in ready:                                   # failed-element probing：单独试"e 排下一个"（只算闭包），出环 → 剪掉
            others = [r for r in remaining if r != e]
            if closure(n, base + [(e, r) for r in others]) is not None: pruned.append(e)
        assert sigma[t] in pruned, (sigma, t, pruned)
        steps.append(dict(t=t, ready=pruned, next=sigma[t], decided=sorted(ch)))
    after, ch = probe(n, list(hard) + [(a, b) for i, a in enumerate(sigma) for b in sigma[i + 1:]], disj, ch)
    steps.append(dict(t=n, ready=[], next=None, decided=sorted(ch)))
    return steps

def first_split_position(n, hard, disj, sigma) -> int:
    """闭包传播（硬边 ∪ 单元规则定下的析取边，不做 probing）下，沿 σ 第一次出现非单元素就绪集的步位置 t（0 基）；始终单元素 → n。
    格规则（用户 2026-09-18）：中位 ≤ n/2（替代已作废的"单元素步占比 ≤0.8"）。"""
    ch = {}
    for t in range(n):
        placed = sigma[:t]; remaining = [e for e in range(n) if e not in placed]
        prefix = [(p_, r) for p_ in placed for r in remaining]
        after, ch = propagate(n, list(hard) + prefix, disj, ch)
        assert after is not None
        ready = [e for e in remaining if not any(_implied(after, r, e) for r in remaining if r != e)]
        if len(ready) >= 2: return t
    return n

def pairwise_accuracy(pred: list[int], sigma: list[int]) -> float | None:
    """逐对准确率：pred 中先后关系与 σ 一致的对的比例（pred 须是 0..n−1 的排列）。"""
    n = len(sigma)
    if sorted(pred) != list(range(n)): return None
    pp = {e: i for i, e in enumerate(pred)}; ps = {e: i for i, e in enumerate(sigma)}
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return sum(1 for a, b in pairs if (pp[a] < pp[b]) == (ps[a] < ps[b])) / len(pairs)

def pairwise_baseline(n, hard, sigma) -> float:
    """硬偏序均匀随机线性扩展的期望逐对准确率（诊断基线）。"""
    les = linear_extensions(n, hard); ps = {e: i for i, e in enumerate(sigma)}
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]; tot = 0.0
    for le in les:
        pp = {e: i for i, e in enumerate(le)}
        tot += sum(1 for a, b in pairs if (pp[a] < pp[b]) == (ps[a] < ps[b])) / len(pairs)
    return tot / len(les)

def lower_bounds(n, d, steps) -> dict:
    kahn = sum(len(s["ready"]) for s in steps[:n]); single = sum(1 for s in steps[:n] if len(s["ready"]) == 1) / n
    dec = d + n
    return dict(decision_lb=dec, kahn_lb=kahn, single_ready_frac=single,
                decision_lb_tokens={str(c): dec * c for c in (2, 2.5, 3)}, kahn_lb_tokens={str(c): kahn * c for c in (2, 2.5, 3)})

# ---------------- structure statistics ----------------
def max_antichain(n, hard) -> int:
    after = closure(n, hard)
    best = 0
    for mask in range(1, 1 << n):
        els = [i for i in range(n) if mask >> i & 1]
        if len(els) <= best: continue
        if all(not _implied(after, a, b) and not _implied(after, b, a) for a, b in itertools.combinations(els, 2)):
            best = len(els)
    return best

def mention_graph_edges(hard, disj) -> set:
    E = set()
    for i, j in hard: E.add(frozenset((i, j)))
    for s1, s2 in disj:
        E.add(frozenset(s1)); E.add(frozenset(s2))
    return E

def hamiltonian_path_count(n, edges: set) -> int:
    adj = [0] * n
    for e in edges:
        a, b = tuple(e); adj[a] |= 1 << b; adj[b] |= 1 << a
    dp = [[0] * n for _ in range(1 << n)]
    for v in range(n): dp[1 << v][v] = 1
    for mask in range(1 << n):
        for v in range(n):
            c = dp[mask][v]
            if not c: continue
            nxt = adj[v] & ~mask
            while nxt:
                u = (nxt & -nxt).bit_length() - 1; nxt &= nxt - 1
                dp[mask | 1 << u][u] += c
    return sum(dp[(1 << n) - 1]) // 2

def redundancy_rate(n, hard, disj) -> float:
    if not disj: return 0.0
    red = sum(1 for k in range(len(disj)) if count_solutions(n, hard, disj[:k] + disj[k + 1:]) == 1)
    return red / len(disj)

def _h(obj) -> str:
    import hashlib
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def canonical_form(n, hard, disj, max_tie_perms: int = 5000) -> str:
    """事件重编号不变的规范形（硬边有向 + 析取两边无序）：颜色细化 + 同色类内枚举置换取最小。"""
    hard = [tuple(p) for p in hard]; disj = [tuple(map(tuple, dj)) for dj in disj]
    color = {v: _h("v") for v in range(n)}
    for _ in range(n + 1):
        new = {}
        for v in range(n):
            ho = sorted(color[j] for i, j in hard if i == v); hi = sorted(color[i] for i, j in hard if j == v)
            dj = []
            for s1, s2 in disj:
                for me, other in ((s1, s2), (s2, s1)):
                    if v in me:
                        role = "src" if me[0] == v else "dst"
                        dj.append((role, color[me[1] if role == "src" else me[0]], tuple(sorted((color[other[0]], color[other[1]])))))
            new[v] = _h([ho, hi, sorted(dj)])
        stable = len(set(new.values())) == len(set(color.values())); color = new
        if stable: break
    classes = {}
    for v in range(n): classes.setdefault(color[v], []).append(v)
    order = [classes[c] for c in sorted(classes)]
    n_perms = 1
    for ps in order:
        for i in range(2, len(ps) + 1): n_perms *= i
    if n_perms > max_tie_perms:
        return "~" + json.dumps(sorted(color.values()))
    best = None
    for perm in itertools.product(*[itertools.permutations(ps) for ps in order]):
        flat = [v for g in perm for v in g]; rl = {old: new for new, old in enumerate(flat)}
        H = sorted((rl[i], rl[j]) for i, j in hard)
        D = sorted(tuple(sorted(((rl[a], rl[b]), (rl[c], rl[e])))) for (a, b), (c, e) in disj)
        s = json.dumps([H, D])
        if best is None or s < best: best = s
    return best

# ---------------- generation ----------------
# 析取抽法（用户 2026-09-18 定，写进 README）：
#   "guided"（默认）：一致边只从 σ 的**覆盖边**（σ 中相邻对）里、且**未被硬约束闭包蕴含**的那些中抽（先无放回；不够 d 条时再有放回）；
#                     不一致边在**所有与 σ 矛盾的对**里均匀抽（不按"最能缩减线性序数"挑，避免不一致边带可学信号）。
#   "uniform"        ：一致边、不一致边都均匀抽（只用于报告引导前的接受率）。
#   两种模式都：一边一致、一边不一致；不与硬约束重合；最终暴力验证恰有一个线性序。
# 记录（meta.gen）：一致边集合是否恰等于"覆盖边 − 硬约束蕴含边"（cover_exact）、重复使用的覆盖边数、不一致边在 σ 中的跨越距离。
def _true_pairs(sigma):
    pos = {e: i for i, e in enumerate(sigma)}
    return [(a, b) for a in range(len(sigma)) for b in range(len(sigma)) if a != b and pos[a] < pos[b]]

def _pick_false_side(rng, tp, hs, t):
    while True:
        f = rng.choice(tp); f = (f[1], f[0])                       # 反转的真对 = 与 σ 矛盾的对，均匀
        if (f[1], f[0]) in hs or t == (f[1], f[0]): continue
        return f

def sample_raw(rng: random.Random, n: int, h: int, d: int, mode: str = GEN_MODE):
    """一次构造：返回 (sigma, hard, disj, gen_info, unique: bool)。"""
    sigma = list(range(n)); rng.shuffle(sigma); pos = {e: i for i, e in enumerate(sigma)}
    tp = _true_pairs(sigma); hard = sorted(rng.sample(tp, h)); hs = set(hard)
    disj = []
    if mode == "uniform":
        for _ in range(d):
            while True:
                t = rng.choice(tp)
                if t in hs: continue
                f = _pick_false_side(rng, tp, hs, t)
                disj.append((t, f) if rng.random() < 0.5 else (f, t)); break
        info = dict(mode=mode)
    else:
        after = closure(n, hard)
        cover = [(sigma[i], sigma[i + 1]) for i in range(n - 1)]
        unimplied = [c for c in cover if not _implied(after, c[0], c[1])]
        pool = list(unimplied); rng.shuffle(pool); used = []
        for k in range(d):
            t = pool.pop() if pool else rng.choice(unimplied)            # 先无放回，用完再有放回
            used.append(t)
            f = _pick_false_side(rng, tp, hs, t)
            disj.append((t, f) if rng.random() < 0.5 else (f, t))
        info = dict(mode=mode, n_cover_unimplied=len(unimplied), cover_exact=(set(used) == set(unimplied)), n_repeated_cover=(len(used) - len(set(used))))
    info["false_side_span"] = [abs(pos[a] - pos[b]) for s1, s2 in disj for (a, b) in (s1, s2) if pos[a] > pos[b]]
    return sigma, hard, disj, info, count_solutions(n, hard, disj) == 1

def generate(rng: random.Random, key: str, mode: str = GEN_MODE, stats: dict | None = None) -> dict:
    p = parse_key(key); n, h, d = p["n"], p["h"], p["d"]
    while True:
        sigma, hard, disj, ginfo, unique = sample_raw(rng, n, h, d, mode)
        if stats is not None: stats["tries"] = stats.get("tries", 0) + 1
        if not unique: continue
        if stats is not None: stats["unique"] = stats.get("unique", 0) + 1
        info = decision_depth(n, hard, disj)
        if info.get("S") is None and not info.get("unsat"):
            info["S"] = MAX_S + 1                      # 截断：记为 >MAX_S
        if not info.get("S"):
            continue
        if stats is not None: stats["S_ge_1"] = stats.get("S_ge_1", 0) + 1
        return make_item(rng, key, sigma, hard, disj, info, ginfo)

def make_item(rng, key, sigma, hard, disj, info, ginfo=None) -> dict:
    n = len(sigma); d = len(disj)
    # 随机重编号 + 顺序随机
    perm = list(range(n)); rng.shuffle(perm); rl = {old: new for new, old in enumerate(perm)}
    sigma2 = [rl[e] for e in sigma]
    hard2 = [(rl[i], rl[j]) for i, j in hard]; rng.shuffle(hard2)
    disj2 = []
    for s1, s2 in disj:
        s1, s2 = (rl[s1[0]], rl[s1[1]]), (rl[s2[0]], rl[s2[1]])
        disj2.append((s2, s1) if rng.random() < 0.5 else (s1, s2))
    rng.shuffle(disj2)
    steps = kahn_trajectory(n, hard2, disj2, sigma2)
    lb = lower_bounds(n, d, steps)
    hedges = mention_graph_edges(hard2, disj2)
    stats = dict(n_le_hard=count_le_hard(n, hard2), max_antichain=max_antichain(n, hard2), ham_paths=hamiltonian_path_count(n, hedges),
                 surviving_disj=info["surviving_disj"], redundancy=redundancy_rate(n, hard2, disj2), single_ready_frac=lb["single_ready_frac"],
                 first_split_pos=first_split_position(n, hard2, disj2, sigma2), pairwise_baseline=pairwise_baseline(n, hard2, sigma2))
    return dict(prompt=build_prompt(n, hard2, disj2), answer=" ".join(map(str, sigma2)),
                meta=dict(task="ordering", key=key, n=n, h=len(hard2), d=d, hard=[list(p) for p in hard2], disj=[[list(a), list(b)] for a, b in disj2],
                          sigma=sigma2, S=info["S"], **lb, canonical_path=canonical_path(n, hard2, disj2, sigma2), steps=steps,
                          stats=stats, gen=(ginfo or {}), canonical=canonical_form(n, hard2, disj2)))

# ---------------- rendering / answers ----------------
def build_prompt(n, hard, disj) -> str:
    return "\n".join([f"n={n}", "hard: " + " ".join(f"{i}<{j}" for i, j in hard),
                      "disj: " + " ".join(f"({a}<{b})|({c}<{d})" for (a, b), (c, d) in disj),
                      "reason inside <think>, then give the order", f"answer: {n} digits, first to last"])

def normalize(ans: str) -> str | None:
    toks = ans.strip().replace(",", " ").split()
    if not toks or not all(t.isdigit() for t in toks): return None
    vals = [int(t) for t in toks]
    if sorted(vals) != list(range(len(vals))): return None
    return " ".join(map(str, vals))

def is_valid(ans: str) -> bool:
    return normalize(ans) is not None

def check(pred: str, gold: str) -> bool:
    p = normalize(pred)
    return p is not None and p == normalize(gold)

def hamming(pred: str, gold: str) -> int | None:
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None or len(p.split()) != len(g.split()): return None
    return sum(a != b for a, b in zip(p.split(), g.split()))

def lenient_extract(text: str, n: int | None = None) -> str | None:
    """宽松抽取：文本中最后一段由 n 个互异事件号（0..n−1）组成的序列（分隔符：空格 / 逗号 / < / → / -）。n 未知时取最长的置换段。"""
    runs = re.findall(r"\d+(?:\s*(?:[ ,<→>\-]|->)\s*\d+)+", text)
    best = None
    for r in runs:
        vals = [int(x) for x in re.findall(r"\d+", r)]
        if len(set(vals)) == len(vals) and sorted(vals) == list(range(len(vals))) and (n is None or len(vals) == n):
            best = " ".join(map(str, vals))
    return best

def chance(key: str, item: dict | None = None) -> float:
    """1/#LE(hard)。无题目时用格的中位 #LE(hard)（需 data 划分）。"""
    if item is not None: return 1.0 / item["meta"]["stats"]["n_le_hard"]
    try:
        import statistics
        sp = load_splits(key, 0); return 1.0 / statistics.median(it["meta"]["stats"]["n_le_hard"] for it in sp["stop"])
    except Exception:
        p = parse_key(key); return 1.0 / (2 ** p["h"] and 20)

# ---------------- splits ----------------
def item_rng(seed, key, split, j): return random.Random(f"cot-compress|ordering|{seed}|{key}|{split}|{j}")

def generate_indexed(args):
    key, split, seed, j = args
    st = {}; it = generate(item_rng(seed, key, split, j), key, stats=st); it["_j"] = j; it["_stats"] = st
    return it

def build_splits(key, seed=0, sizes=None, log=print, workers=None) -> dict:
    import multiprocessing as mp
    sizes = dict(SPLITS, **(sizes or {})); out, seen = {}, set(); gen_stats = Counter()
    for split in ("stop", "report", "direct", "train"):
        items, forms, j = [], set(), 0
        while len(items) < sizes[split]:
            batch = [(key, split, seed, j + t) for t in range(max(8, int((sizes[split] - len(items)) * 1.05) + 1))]; j += len(batch)
            if workers and workers > 1:
                with mp.Pool(workers) as pool: cands = pool.map(generate_indexed, batch, chunksize=2)
            else:
                cands = [generate_indexed(a) for a in batch]
            for it in sorted(cands, key=lambda x: x["_j"]):
                gen_stats.update(it.pop("_stats"))
                cf = it["meta"]["canonical"]
                if cf in seen or cf in forms or len(items) >= sizes[split]: continue
                forms.add(cf); del it["_j"]; it["id"] = f"{key}/{split}/{seed}/{len(items)}"; items.append(it)
        seen |= forms; out[split] = items
        if log: log(f"[ordering] {key} seed={seed} split={split}: {len(items)} items, {len(forms)} distinct isomorphism classes (candidates drawn: {j})")
    assert_disjoint(out)
    out["_gen_stats"] = dict(gen_stats)
    out["_uniform_rate"] = uniform_acceptance_rate(key, seed)
    return out

def uniform_acceptance_rate(key, seed=0, tries=20000) -> float:
    """引导前（纯均匀）的唯一解接受率，报告用。"""
    p = parse_key(key); rng = random.Random(f"uniform-rate|{key}|{seed}"); u = 0
    for _ in range(tries):
        if sample_raw(rng, p["n"], p["h"], p["d"], "uniform")[4]: u += 1
    return u / tries

def assert_disjoint(splits: dict):
    names = [k for k in splits if not k.startswith("_")]
    forms = {k: {it["meta"]["canonical"] for it in splits[k]} for k in names}
    for a, b in itertools.combinations(names, 2):
        assert not (forms[a] & forms[b]), f"isomorphism-class overlap between {a} and {b}"
    prompts = {k: {it["prompt"] for it in splits[k]} for k in names}
    for a, b in itertools.combinations(names, 2):
        assert not (prompts[a] & prompts[b])

def split_path(key, seed): return DATA_DIR / f"{key}_seed{seed}.json"

_CACHE = {}
def load_splits(key, seed=0, build=False, workers=None) -> dict:
    if (key, seed) in _CACHE: return _CACHE[(key, seed)]
    p = split_path(key, seed)
    if p.exists(): sp = json.load(open(p))
    elif not build: raise FileNotFoundError(f"{p} missing: run `python scripts/ord_build_splits.py --key {key} --seed {seed}` first")
    else:
        sp = build_splits(key, seed, workers=workers); p.parent.mkdir(parents=True, exist_ok=True); json.dump(sp, open(p, "w"))
    _CACHE[(key, seed)] = sp; return sp

# ---------------- cell statistics / decision ----------------
import statistics as _st
RULES = dict(n_le_hard_median_ge=20, first_split_pos_median_le_frac_n=0.5, max_antichain_median_ge=3, ham_paths_median_ge=3, surviving_disj_median_ge=3, retention_S_ge1_ge=0.01)
# "单元素步占比中位 ≤0.8" 已作废（与 S=1 天然冲突，用户 2026-09-18）；single_ready_frac 只报。

def cell_stats(splits: dict) -> dict:
    names = [k for k in splits if not k.startswith("_")]
    rep = {}
    for k in names:
        v = splits[k]; st = [it["meta"]["stats"] for it in v]
        n_ = v[0]["meta"]["n"]
        rep[k] = dict(n=len(v), n_events=n_, distinct_classes=len({it["meta"]["canonical"] for it in v}),
                      n_le_hard_median=_st.median(s["n_le_hard"] for s in st), single_ready_frac_median=_st.median(s["single_ready_frac"] for s in st),
                      first_split_pos_median=_st.median(s.get("first_split_pos", n_) for s in st),
                      first_split_pos_hist=dict(sorted(Counter(s.get("first_split_pos", n_) for s in st).items())),
                      pairwise_baseline_mean=_st.mean(s.get("pairwise_baseline", 0.5) for s in st),
                      max_antichain_median=_st.median(s["max_antichain"] for s in st), ham_paths_median=_st.median(s["ham_paths"] for s in st),
                      surviving_disj_median=_st.median(s["surviving_disj"] for s in st), redundancy_mean=_st.mean(s["redundancy"] for s in st),
                      S_dist=dict(sorted(Counter(it["meta"]["S"] for it in v).items())),
                      decision_lb=v[0]["meta"]["decision_lb"], kahn_lb_median=_st.median(it["meta"]["kahn_lb"] for it in v),
                      kahn_lb_tokens_median={c: _st.median(it["meta"]["kahn_lb_tokens"][c] for it in v) for c in ("2", "2.5", "3")},
                      decision_lb_tokens=v[0]["meta"]["decision_lb_tokens"])
    gs = splits.get("_gen_stats") or {}
    allit = [it for k in names for it in splits[k]]
    spans = Counter(x for it in allit for x in it["meta"].get("gen", {}).get("false_side_span", []))
    rep["_generation"] = dict(gs, retention_S_ge1=(gs.get("S_ge_1", 0) / gs["unique"]) if gs.get("unique") else None,
                              unique_rate_guided=(gs.get("unique", 0) / gs["tries"]) if gs.get("tries") else None,
                              unique_rate_uniform=splits.get("_uniform_rate"),
                              cover_exact_frac=(sum(1 for it in allit if it["meta"].get("gen", {}).get("cover_exact")) / len(allit)) if allit else None,
                              repeated_cover_mean=(_st.mean(it["meta"].get("gen", {}).get("n_repeated_cover", 0) for it in allit)) if allit else None,
                              false_side_span_hist={str(k): v / max(1, sum(spans.values())) for k, v in sorted(spans.items())})
    tr = {it["meta"]["canonical"] for it in splits.get("train", [])}
    for ev in ("stop", "report"):
        if ev in splits:
            ef = {it["meta"]["canonical"] for it in splits[ev]}
            rep[ev]["iso_classes_subset_of_train"] = ef <= tr; rep[ev]["iso_overlap_with_train"] = len(ef & tr)
    return rep

def cell_decision(rep: dict, split: str = "stop") -> dict:
    s = rep[split]; g = rep.get("_generation", {})
    checks = {
        "n_le_hard_median_ge_20": s["n_le_hard_median"] >= RULES["n_le_hard_median_ge"],
        "first_split_pos_median_le_n/2": s["first_split_pos_median"] <= RULES["first_split_pos_median_le_frac_n"] * s["n_events"],
        "max_antichain_median_ge_3": s["max_antichain_median"] >= RULES["max_antichain_median_ge"],
        "ham_paths_median_ge_3": s["ham_paths_median"] >= RULES["ham_paths_median_ge"],
        "surviving_disj_median_ge_3": s["surviving_disj_median"] >= RULES["surviving_disj_median_ge"],
        "retention_S_ge1_ge_1pct": (g.get("retention_S_ge1") is None) or g["retention_S_ge1"] >= RULES["retention_S_ge1_ge"],
        "eval_iso_classes_not_subset_of_train": not s.get("iso_classes_subset_of_train", False),
    }
    return dict(ok=all(checks.values()), checks=checks, S_dist=s["S_dist"], redundancy_mean=s["redundancy_mean"])
