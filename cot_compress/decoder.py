"""线性解码器（v7 §5）：L2 多项逻辑回归，输入 = think 段 n-gram（n ≤ 3）词袋（哈希到 D 维；≥4 token 的 IR 复制片段先掩掉），
标签 = 金标准决策树深度 k∈{0..S} 处单元传播后的部分赋值（每人一个 3 类标签 {0,1,?}），按题划分 train / held-out（题 id 哈希）。
指标：acc_all（所有 (题, 人) 位置）、acc_assigned（标签 ∈ {0,1} 的位置）、majority 基线；仪器检查 = A 轨迹上 k=S（主格 S=1：分情况之后）的 acc_assigned ≥ 0.90。
基线：移植轨迹（think 换成他题的）、C_rand 轨迹；报告 B − 基线的差与配对 bootstrap 区间（按 held-out 题配对）。
纯 numpy（云上没有 sklearn 也能跑）。
"""
from __future__ import annotations
import hashlib, random
import numpy as np
from .strategy import copied_mask

CLASSES = ("0", "1", "?")

def _hash(s: str, D: int) -> int:
    return int(hashlib.blake2b(s.encode(), digest_size=4).hexdigest(), 16) % D

def featurize(think_tokens: list, prompt_tokens: list, D: int = 4096, max_n: int = 3, mask_copy: bool = True) -> np.ndarray:
    toks = [str(t) for t in think_tokens]
    if mask_copy and prompt_tokens:
        cm = copied_mask(think_tokens, prompt_tokens, 4)
        toks = [t for t, c in zip(toks, cm) if not c]
    x = np.zeros(D, dtype=np.float32)
    for n in range(1, max_n + 1):
        for i in range(len(toks) - n + 1):
            x[_hash(f"{n}|" + " ".join(toks[i:i + n]), D)] += 1.0
    x = np.log1p(x)
    nrm = np.linalg.norm(x)
    return x / nrm if nrm > 0 else x

def split_items(ids: list[str], held_frac: float = 0.3, seed: int = 0) -> tuple[set, set]:
    """按题划分：id 的哈希决定；同题所有轨迹同侧。"""
    uniq = sorted(set(ids)); rng = random.Random(seed); rng.shuffle(uniq)
    k = int(round(len(uniq) * held_frac))
    return set(uniq[k:]), set(uniq[:k])

class GroupedLogReg:
    """N 个人 × 3 类的联合 softmax 回归（每人一组 softmax），L2 正则，全批梯度下降。"""
    def __init__(self, n_persons: int, D: int, l2: float = 1e-3, lr: float = 0.5, iters: int = 300):
        self.n, self.D, self.l2, self.lr, self.iters = n_persons, D, l2, lr, iters
        self.W = np.zeros((D, n_persons, 3), dtype=np.float32); self.b = np.zeros((n_persons, 3), dtype=np.float32)

    def _logits(self, X):
        return np.einsum("md,dnc->mnc", X, self.W) + self.b

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """X [m, D]; Y [m, n] 类下标 0/1/2。"""
        m = X.shape[0]; onehot = np.zeros((m, self.n, 3), dtype=np.float32)
        for c in range(3): onehot[:, :, c] = (Y == c)
        for _ in range(self.iters):
            Z = self._logits(X); Z -= Z.max(-1, keepdims=True); P = np.exp(Z); P /= P.sum(-1, keepdims=True)
            G = (P - onehot) / m
            gW = np.einsum("md,mnc->dnc", X, G) + self.l2 * self.W; gb = G.sum(0)
            self.W -= self.lr * gW; self.b -= self.lr * gb
        return self

    def predict(self, X):
        return self._logits(X).argmax(-1)

def labels_to_idx(label: str) -> np.ndarray:
    return np.array([CLASSES.index(ch) for ch in label], dtype=np.int64)

def evaluate(pred: np.ndarray, Y: np.ndarray) -> dict:
    assigned = Y != 2
    return dict(acc_all=float((pred == Y).mean()), acc_assigned=(float((pred[assigned] == Y[assigned]).mean()) if assigned.any() else None),
                frac_assigned=float(assigned.mean()), exact_rows=float((pred == Y).all(1).mean()))

def majority_baseline(Ytr: np.ndarray, Yte: np.ndarray) -> dict:
    maj = np.array([np.bincount(Ytr[:, i], minlength=3).argmax() for i in range(Ytr.shape[1])])
    return evaluate(np.tile(maj, (Yte.shape[0], 1)), Yte)

def run_decoder(rows: list[dict], k: int, n_persons: int, D: int = 4096, held_frac: float = 0.3, seed: int = 0, **lr_kw) -> dict:
    """rows: dict(id, think_tokens, prompt_tokens, labels=[str]*3)。返回指标与 per-item held-out 正确率（配对 bootstrap 用）。"""
    tr_ids, te_ids = split_items([r["id"] for r in rows], held_frac, seed)
    assert not (tr_ids & te_ids)
    X = np.stack([featurize(r["think_tokens"], r["prompt_tokens"], D) for r in rows]); Y = np.stack([labels_to_idx(r["labels"][k]) for r in rows])
    tr = np.array([r["id"] in tr_ids for r in rows]); te = ~tr
    model = GroupedLogReg(n_persons, D, **lr_kw).fit(X[tr], Y[tr])
    pred = model.predict(X[te])
    res = evaluate(pred, Y[te]); res["majority"] = majority_baseline(Y[tr], Y[te]); res["n_train"] = int(tr.sum()); res["n_test"] = int(te.sum())
    ass = Y[te] != 2
    per_item = {}
    for j, r in enumerate([r for r, t in zip(rows, te) if t]):
        per_item.setdefault(r["id"], []).append(float((pred[j][ass[j]] == Y[te][j][ass[j]]).mean()) if ass[j].any() else None)
    res["per_item_acc_assigned"] = {i: (float(np.mean([v for v in vs if v is not None])) if any(v is not None for v in vs) else None) for i, vs in per_item.items()}
    return res

def paired_bootstrap(a: dict, b: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """a, b: per-item 指标（同一批 held-out 题）；返回 mean(a−b) 与 2.5/97.5 分位。"""
    common = [i for i in a if i in b and a[i] is not None and b[i] is not None]
    if not common:
        return dict(diff=None, lo=None, hi=None, n=0)
    d = np.array([a[i] - b[i] for i in common]); rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return dict(diff=float(d.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), n=len(common))

def transplant_rows(rows: list[dict], seed: int = 0) -> list[dict]:
    """移植基线：每条的 think 换成另一题的 think（标签不变）。"""
    rng = random.Random(seed); out = []
    for r in rows:
        others = [o for o in rows if o["id"] != r["id"]]
        o = rng.choice(others) if others else r
        out.append(dict(r, think_tokens=o["think_tokens"]))
    return out

# ======================= v8 ordering：前缀对齐解析器 + 真值标签 =======================
# 特征：仅前缀的 think 段 n-gram（n≤3）词袋，复制片段掩掉（featurize）。
# 对齐：发布的解析器 parse_step(prefix, n) 只读轨迹前缀 → 步数 t = 前缀里最长事件链（数字用 < / → / -> / , 连接，去重后长度，封顶 n）；
#       无链 → 不可解析（排除，报告解析率）。标签 labels_for(item, t) 只由题目真值决定（steps[t]：下一事件、真值就绪集、已定析取集），绝不读轨迹。
import re as _re
_CHAIN = _re.compile(r"\d+(?:\s*(?:<|→|->|,)\s*\d+)+")

def parse_step(prefix: str, n: int) -> int | None:
    best = 0
    for m in _CHAIN.finditer(prefix):
        vals = [int(x) for x in _re.findall(r"\d+", m.group(0))]
        if not all(0 <= v < n for v in vals): continue
        best = max(best, len(dict.fromkeys(vals)))
    return min(best, n) if best > 0 else None

def labels_for(item: dict, t: int) -> dict:
    """真值标签：steps[t]（t 个事件已放）。next：下一事件（n 类，t=n 时 '?'）；ready：n 位 0/1；decided：d 位 0/1。"""
    m = item["meta"]; n, d = m["n"], m["d"]; st = m["steps"][min(t, n)]
    ready = "".join("1" if e in st["ready"] else "0" for e in range(n))
    decided = "".join("1" if k in st["decided"] else "0" for k in range(d))
    return dict(next=("?" if st["next"] is None else str(st["next"])), ready=ready, decided=decided)

def prefix_rows(rows: list[dict], items: dict, fracs=(0.25, 0.5, 0.75, 1.0), tok=None) -> tuple[list[dict], dict]:
    """rows: dict(id, think, prompt)。按前缀比例切，解析 t，配真值标签。返回 (行, 解析率)。"""
    from .strategy import tokens_of
    out, tried, parsed = [], 0, 0
    for r in rows:
        it = items.get(r["id"])
        if it is None: continue
        n = it["meta"]["n"]
        for f in fracs:
            pre = r["think"][:max(1, int(len(r["think"]) * f))]; tried += 1
            t = parse_step(pre, n)
            if t is None: continue
            parsed += 1
            lab = labels_for(it, t)
            out.append(dict(id=r["id"], frac=f, t=t, think_tokens=tokens_of(pre, tok), prompt_tokens=tokens_of(it["prompt"], tok), labels=lab))
    return out, dict(parse_rate=(parsed / tried) if tried else None, n_prefixes=tried, n_parsed=parsed)

def run_ordering_decoder(prow: list[dict], n: int, target: str = "next", D: int = 4096, held_frac: float = 0.3, seed: int = 0, **lr_kw) -> dict:
    """target: next（n 类 + '?'）| ready（n 个二元）| decided（d 个二元）。返回 held-out 准确率、majority、per-item 准确率（配对 bootstrap 用）。"""
    if not prow: return dict(acc=None, n_train=0, n_test=0)
    tr_ids, te_ids = split_items([r["id"] for r in prow], held_frac, seed)
    X = np.stack([featurize(r["think_tokens"], r["prompt_tokens"], D) for r in prow])
    if target == "next":
        classes = [str(i) for i in range(n)] + ["?"]
        Y = np.array([[classes.index(r["labels"]["next"])] for r in prow]); K = len(classes); G = 1
    else:
        Y = np.stack([labels_to_idx(r["labels"][target]) for r in prow]); K = 3; G = Y.shape[1]
    tr = np.array([r["id"] in tr_ids for r in prow]); te = ~tr
    model = _GroupedK(G, K, D, **lr_kw).fit(X[tr], Y[tr]); pred = model.predict(X[te])
    acc = float((pred == Y[te]).mean()); maj = np.array([np.bincount(Y[tr][:, g], minlength=K).argmax() for g in range(G)])
    maj_acc = float((np.tile(maj, (te.sum(), 1)) == Y[te]).mean())
    per_item = {}
    for j, r in enumerate([r for r, t in zip(prow, te) if t]):
        per_item.setdefault(r["id"], []).append(float((pred[j] == Y[te][j]).mean()))
    return dict(acc=acc, majority=maj_acc, n_train=int(tr.sum()), n_test=int(te.sum()), per_item_acc={i: float(np.mean(v)) for i, v in per_item.items()})

class _GroupedK(GroupedLogReg):
    """G 组 × K 类的联合 softmax（GroupedLogReg 的 K 类推广）。"""
    def __init__(self, G, K, D, l2=1e-3, lr=0.5, iters=300):
        self.n, self.K, self.D, self.l2, self.lr, self.iters = G, K, D, l2, lr, iters
        self.W = np.zeros((D, G, K), dtype=np.float32); self.b = np.zeros((G, K), dtype=np.float32)
    def fit(self, X, Y):
        m = X.shape[0]; onehot = np.zeros((m, self.n, self.K), dtype=np.float32)
        for c in range(self.K): onehot[:, :, c] = (Y == c)
        for _ in range(self.iters):
            Z = self._logits(X); Z -= Z.max(-1, keepdims=True); P = np.exp(Z); P /= P.sum(-1, keepdims=True)
            Gd = (P - onehot) / m
            self.W -= self.lr * (np.einsum("md,mnc->dnc", X, Gd) + self.l2 * self.W); self.b -= self.lr * Gd.sum(0)
        return self

def soundness_completeness(prefix: str, item: dict) -> dict:
    """描述量：前缀断言的 a<b 对（括号外）相对真值的 soundness；相对"自身承诺闭包"（硬边 ∪ 前缀断言到的析取边 的传递闭包）的 completeness。"""
    from tasks.ordering import closure as _closure
    from .shortcuts import asserted_pairs
    m = item["meta"]; n = m["n"]; pos = {e: i for i, e in enumerate(m["sigma"])}
    ap = [(a, b) for a, b in asserted_pairs(prefix) if 0 <= a < n and 0 <= b < n and a != b]
    if not ap: return dict(soundness=None, completeness=None, n_asserted=0)
    sound = sum(1 for a, b in ap if pos[a] < pos[b]) / len(ap)
    sides = {tuple(s) for dj in m["disj"] for s in dj}
    committed = [tuple(h) for h in m["hard"]] + [p for p in ap if p in sides]
    after = _closure(n, committed)
    if after is None: return dict(soundness=sound, completeness=None, n_asserted=len(ap), cyclic=True)
    cl = {(i, j) for i in range(n) for j in range(n) if after[i] >> j & 1}
    return dict(soundness=sound, completeness=(len(set(ap) & cl) / len(cl)) if cl else None, n_asserted=len(ap))

def cipher_spearman_warm(a_thinks: list[str], b_thinks: list[str]) -> dict:
    """描述量：用 B_warm 映射把 A 的控制词配到 B 的符号，比较配对后的频率 Spearman。"""
    from .warm import WARM_MAP
    from .endpoints import spearman
    from collections import Counter as _C
    fa = _C(w for t in a_thinks for w in _re.findall(r"[a-zA-Z]+", t.lower())); fb = _C(x for t in b_thinks for x in t.split())
    by_sym = _C()
    for w, sym in WARM_MAP.items(): by_sym[sym] += fa[w]                      # 同一符号的多个控制词合并
    syms = [sym for sym in by_sym if by_sym[sym] > 0 or fb[sym] > 0]
    rho = spearman([by_sym[sym] for sym in syms], [fb[sym] for sym in syms])
    return dict(n_pairs=len(syms), spearman=rho)
