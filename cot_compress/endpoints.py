"""端点计算（v7 §8），纯函数，可单测。analyze.py 组装。
E1 匹配准确率与税：
  每臂每 seed 的强制预算曲线点 (x_f, y_f)，f ∈ 0.1…1.0 × 自身收敛长度；y = 外化准确率 = acc_f − 训练后 direct；x = 该预算下正确轨迹的 token 数均值。
  每 seed 曲线做 isotonic（PAVA，y 随 x 非降）单调化。
  匹配准确率 y* = min_{臂 ∈ A, A″, B} (该臂所有 seed / 点的最高 y) − 5pp；任一臂没有点落在 y* 的 5pp 内 → E1 不计算。
  L_arm,seed(y*) = 单调化曲线上首次达到 y* 的 x（线性插值）。税 = (L_A − L_B)/L_A（按 seed 配对），报 per-seed、范围、seed 级精确单边 Mann-Whitney（H1: L_B < L_A）。
  三种单位：token；code point（每臂 seed 的正确 think 段 code point/token 比换算）；zstd -19 共享字典比特（字典在所有臂正确 think 段并集的一半上训练，另一半算 bits/token）。
  阈值：≥67% 强、≥25% 中、<10% 无税；≤ −10% 负税单独报告。
"""
from __future__ import annotations
import itertools, math, random, statistics
from collections import Counter

# ---------- isotonic (PAVA) ----------
def isotonic_nondecreasing(y: list[float], w: list[float] | None = None) -> list[float]:
    w = list(w) if w else [1.0] * len(y)
    blocks = [[float(v), float(wt), 1] for v, wt in zip(y, w)]      # [mean, weight, count]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            m = (a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1])
            blocks[i] = [m, a[1] + b[1], a[2] + b[2]]; del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for m, _, c in blocks: out += [m] * c
    return out

def monotone_curve(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """points (x, y) → 按 x 排序并对 y 做非降单调化。"""
    pts = sorted(points); ys = isotonic_nondecreasing([p[1] for p in pts])
    return [(p[0], y) for p, y in zip(pts, ys)]

def length_at(curve: list[tuple[float, float]], y_star: float) -> float | None:
    """单调曲线上首次达到 y_star 的 x（线性插值）；达不到 → None。"""
    prev = None
    for x, y in curve:
        if y >= y_star:
            if prev is None or prev[1] == y: return x
            x0, y0 = prev
            return x0 + (y_star - y0) / (y - y0) * (x - x0)
        prev = (x, y)
    return None

# ---------- matched accuracy ----------
def matched_accuracy(curves: dict[str, dict[int, list[tuple[float, float]]]], arms=("A", "A2", "B"), margin: float = 0.05) -> dict:
    """curves[arm][seed] = 单调曲线。返回 y*、可计算性与各臂最高 y。"""
    tops = {}
    for a in arms:
        if a not in curves or not curves[a]:
            return dict(y_star=None, computable=False, reason=f"arm {a} missing", tops=tops)
        tops[a] = max(y for c in curves[a].values() for _, y in c)
    y_star = min(tops.values()) - margin
    near = {a: any(abs(y - y_star) <= margin + 1e-9 for c in curves[a].values() for _, y in c) for a in arms}
    ok = all(near.values())
    return dict(y_star=y_star, computable=ok, reason=(None if ok else f"no point within {margin:.0%} of y* for arms {[a for a, v in near.items() if not v]}"), tops=tops, near=near)

# ---------- Mann-Whitney exact one-sided ----------
def mann_whitney_exact(a: list[float], b: list[float], alternative: str = "less") -> dict:
    """精确单边 Mann-Whitney U（seed 级，小样本枚举）。alternative='less'：H1 a 的值倾向小于 b（U_a 小）。"""
    a, b = list(a), list(b); n, m = len(a), len(b)
    if n == 0 or m == 0: return dict(U=None, p=None, n=n, m=m)
    U = sum(1.0 if x < y else (0.5 if x == y else 0.0) for x in a for y in b)
    pool = a + b; idx = range(n + m); count = 0; total = 0
    for comb in itertools.combinations(idx, n):
        s = set(comb); aa = [pool[i] for i in comb]; bb = [pool[i] for i in idx if i not in s]
        u = sum(1.0 if x < y else (0.5 if x == y else 0.0) for x in aa for y in bb)
        total += 1
        if (alternative == "less" and u >= U) or (alternative == "greater" and u <= U): count += 1
    return dict(U=U, p=count / total, n=n, m=m, exact=True)

# ---------- spearman ----------
def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]: j += 1
        for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r

def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3: return None
    rx, ry = _rank(x), _rank(y); mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None

# ---------- zstd shared dictionary ----------
def zstd_dict_bits(segments_by_group: dict[str, list[str]], dict_size: int = 110 * 1024, level: int = 19, seed: int = 0) -> dict:
    """所有组正确 think 段并集随机对半：一半训字典，另一半按组算 bits（压缩后字节 × 8）与每段 bits。返回 {group: dict(bits_mean, n, bytes_mean)}。"""
    import zstandard as zstd
    rng = random.Random(seed)
    allseg = [(g, s) for g, segs in segments_by_group.items() for s in segs if s]
    rng.shuffle(allseg); half = len(allseg) // 2
    train = [s.encode() for _, s in allseg[:half]]; test = allseg[half:]
    if len(train) < 8 or not test:
        return {g: dict(bits_mean=None, n=0) for g in segments_by_group}
    d = zstd.train_dictionary(dict_size, train)
    cctx = zstd.ZstdCompressor(level=level, dict_data=d)
    out = {}
    for g in segments_by_group:
        segs = [s for gg, s in test if gg == g]
        if not segs: out[g] = dict(bits_mean=None, n=0); continue
        bits = [len(cctx.compress(s.encode())) * 8 for s in segs]
        out[g] = dict(bits_mean=statistics.mean(bits), n=len(segs), bits_per_char=statistics.mean(b / max(1, len(s)) for b, s in zip(bits, segs)))
    return out

# ---------- tax ----------
def tax_level(t: float | None) -> str | None:
    if t is None: return None
    if t <= -0.10: return "negative"
    if t < 0.10: return "none"
    if t < 0.25: return "small"
    if t < 0.67: return "moderate"
    return "strong"

def tax_by_seed(L_A: dict[int, float], L_B: dict[int, float]) -> dict:
    seeds = sorted(set(L_A) & set(L_B)); per = {s: (L_A[s] - L_B[s]) / L_A[s] for s in seeds if L_A[s]}
    vals = list(per.values())
    mw = mann_whitney_exact([L_B[s] for s in seeds], [L_A[s] for s in seeds], "less")
    return dict(per_seed={str(s): v for s, v in per.items()}, mean=(statistics.mean(vals) if vals else None),
                range=((min(vals), max(vals)) if vals else None), level=tax_level(statistics.mean(vals) if vals else None),
                negative_seeds=[str(s) for s, v in per.items() if v <= -0.10], mann_whitney_LB_lt_LA=mw)

# ---------- substitution-cipher check ----------
def cipher_check(a_items: dict[str, Counter], b_items: dict[str, Counter], top_k: int = 100, min_items: int = 20) -> dict:
    """B 的 token 与 A 的词是否构成替换密码：按题配对（同题 A 轨迹词计数 / B 轨迹 token 计数），
    对 B 的前 top_k 个 token，各取与之跨题计数相关最大的 A 词（贪心一对一），再算配对词/token 的总频率 Spearman；≥ 0.85 判为密码。"""
    items = sorted(set(a_items) & set(b_items))
    if len(items) < min_items: return dict(applicable=False, n_items=len(items))
    fa, fb = Counter(), Counter()
    for i in items: fa.update(a_items[i]); fb.update(b_items[i])
    A = [w for w, _ in fa.most_common(top_k * 3)]; B = [t for t, _ in fb.most_common(top_k)]
    def vec(c, k): return [c[i].get(k, 0) for i in items]
    va = {w: vec(a_items, w) for w in A}; used = set(); pairs = []
    for t in B:
        vb = vec(b_items, t); best, bw = None, None
        for w in A:
            if w in used: continue
            r = _pearson(vb, va[w])
            if r is not None and (best is None or r > best): best, bw = r, w
        if bw is not None: used.add(bw); pairs.append((t, bw, best))
    if len(pairs) < 5: return dict(applicable=False, n_items=len(items), n_pairs=len(pairs))
    rho = spearman([fb[t] for t, _, _ in pairs], [fa[w] for _, w, _ in pairs])
    return dict(applicable=True, n_items=len(items), n_pairs=len(pairs), spearman=rho, is_cipher=(rho is not None and rho >= 0.85),
                mean_pair_corr=statistics.mean(r for _, _, r in pairs))

def _pearson(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y)); den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else None

# ======================= v8：按题配对 bootstrap 的税（seed 等权分块） =======================
TIERS = ((0.67, "strong"), (0.25, "moderate"), (0.10, "small"))

def tier_v8(t: float | None) -> str | None:
    """档位：≥67 强 / ≥25 中 / 10–25 小 / <10 无税 / ≤−10 负税。"""
    if t is None: return None
    if t <= -0.10: return "negative"
    if t < 0.10: return "none"
    for thr, name in TIERS:
        if t >= thr: return name
    return "none"

def curve_from_items(per_item: dict, ids: list) -> list[tuple[float, float]]:
    """per_item[f] = {id: (correct: bool, tokens: int)} → 单调曲线 [(x=正确轨迹 token 均值, y=总准确率)]。"""
    pts = []
    for f, d in per_item.items():
        rows = [d[i] for i in ids if i in d]
        if not rows: continue
        corr = [t for c, t in rows if c]
        if not corr: continue
        pts.append((statistics.mean(corr), sum(1 for c, _ in rows if c) / len(rows)))
    return monotone_curve(pts) if pts else []

def tax_from_curves(curves: dict, arms=("A", "A2", "B"), margin: float = 0.05, scale: dict | None = None) -> dict | None:
    """curves[arm][seed] → 匹配 y* → 每 seed 的 L → 税（A vs B 按 seed 配对，seed 等权）。scale[arm][seed] = 单位换算系数（默认 1）。
    任一臂无 5pp 内的点 → None。"""
    ma = matched_accuracy(curves, arms, margin)
    if not ma["computable"]: return None
    y = ma["y_star"]
    L = {a: {s: length_at(c, y) for s, c in curves[a].items()} for a in curves}
    if scale:
        L = {a: {s: (v * scale[a][s] if v is not None else None) for s, v in sv.items()} for a, sv in L.items()}
    seeds = sorted(s for s in L.get("A", {}) if s in L.get("B", {}) and L["A"][s] and L["B"][s] is not None)
    if not seeds: return None
    per = {s: (L["A"][s] - L["B"][s]) / L["A"][s] for s in seeds}
    return dict(y_star=y, per_seed=per, mean=statistics.mean(per.values()), L=L)

def bootstrap_tax(per_item_by_run: dict, ids: list, n_boot: int = 200, seed: int = 0, scale: dict | None = None, arms=("A", "A2", "B")) -> dict:
    """per_item_by_run[arm][seed] = per_item（见 curve_from_items）。按题重采样（所有臂 / seed 用同一份重采样 → 配对），seed 等权。
    返回点估计、bootstrap 均值与 2.5/97.5 分位、不可计算的重采样比例。"""
    rng = random.Random(seed)
    def curves_for(sel):
        return {a: {s: curve_from_items(pi, sel) for s, pi in sv.items()} for a, sv in per_item_by_run.items()}
    point = tax_from_curves(curves_for(ids), arms, scale=scale)
    boots, skipped = [], 0
    for _ in range(n_boot):
        sel = [ids[rng.randrange(len(ids))] for _ in ids]
        t = tax_from_curves(curves_for(sel), arms, scale=scale)
        if t is None: skipped += 1; continue
        boots.append(t["mean"])
    boots.sort()
    q = lambda p: boots[min(len(boots) - 1, int(p * len(boots)))] if boots else None
    return dict(point=(point["mean"] if point else None), per_seed=(point["per_seed"] if point else None), y_star=(point["y_star"] if point else None),
                boot_mean=(statistics.mean(boots) if boots else None), lo=q(0.025), hi=q(0.975), n_boot=len(boots), skipped_frac=skipped / n_boot,
                tier=tier_v8(point["mean"] if point else None), L=(point["L"] if point else None))

def zstd_bits_per_token(segments_by_group: dict, tokens_by_group: dict, level: int = 19, seed: int = 0, per_group_dict: bool = False) -> dict:
    """每组的 zstd-19 bits/token：并集字典（per_group_dict=False）或各组自己的字典（True）；字典训练 / 测量对半分。"""
    import zstandard as zstd
    rng = random.Random(seed); out = {}
    groups = list(segments_by_group)
    def train_test(segs):
        idx = list(range(len(segs))); rng.shuffle(idx); h = len(idx) // 2
        return [segs[i] for i in idx[:h]], [i for i in idx[h:]]
    if not per_group_dict:
        allseg = [(g, i) for g in groups for i in range(len(segments_by_group[g]))]
        rng.shuffle(allseg); h = len(allseg) // 2
        train = [segments_by_group[g][i].encode() for g, i in allseg[:h]]; test = allseg[h:]
        if len(train) < 8: return {g: None for g in groups}
        cctx = zstd.ZstdCompressor(level=level, dict_data=zstd.train_dictionary(110 * 1024, train))
        for g in groups:
            te = [i for gg, i in test if gg == g]
            bits = sum(len(cctx.compress(segments_by_group[g][i].encode())) * 8 for i in te); toks = sum(tokens_by_group[g][i] for i in te)
            out[g] = (bits / toks) if toks else None
        return out
    for g in groups:
        segs = segments_by_group[g]; train, te = train_test(segs)
        if len(train) < 8: out[g] = None; continue
        cctx = zstd.ZstdCompressor(level=level, dict_data=zstd.train_dictionary(110 * 1024, [s.encode() for s in train]))
        bits = sum(len(cctx.compress(segs[i].encode())) * 8 for i in te); toks = sum(tokens_by_group[g][i] for i in te)
        out[g] = (bits / toks) if toks else None
    return out
