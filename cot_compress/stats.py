"""seed 层面的置换检验（v4 第 6 条）：两组 seed 均值的差，标签置换求 p。不做分层 bootstrap。"""
from __future__ import annotations
import itertools, random, statistics

def permutation_test(a: list[float], b: list[float], n_perm: int = 10000, seed: int = 0, alternative: str = "two-sided") -> dict:
    a, b = list(a), list(b)
    obs = statistics.mean(a) - statistics.mean(b)
    pool = a + b; na = len(a)
    if na == 0 or len(b) == 0:
        return dict(diff=obs, p=None, n_perm=0, exact=False)
    total = len(pool)
    from math import comb
    exact = comb(total, na) <= n_perm
    diffs = []
    if exact:
        for idx in itertools.combinations(range(total), na):
            s = set(idx); aa = [pool[i] for i in idx]; bb = [pool[i] for i in range(total) if i not in s]
            diffs.append(statistics.mean(aa) - statistics.mean(bb))
    else:
        rng = random.Random(seed)
        for _ in range(n_perm):
            rng.shuffle(pool); diffs.append(statistics.mean(pool[:na]) - statistics.mean(pool[na:]))
    if alternative == "greater":
        p = sum(d >= obs for d in diffs) / len(diffs)
    elif alternative == "less":
        p = sum(d <= obs for d in diffs) / len(diffs)
    else:
        p = sum(abs(d) >= abs(obs) for d in diffs) / len(diffs)
    return dict(diff=obs, p=p, n_perm=len(diffs), exact=exact)

def seed_ci(values: list[float]) -> dict:
    """seed 均值与简单区间（均值 ± 1.96·se；n<2 时 se=None）。"""
    n = len(values)
    if n == 0: return dict(mean=None, se=None, lo=None, hi=None, n=0)
    m = statistics.mean(values); se = (statistics.stdev(values) / n ** 0.5) if n > 1 else None
    return dict(mean=m, se=se, lo=(m - 1.96 * se) if se is not None else None, hi=(m + 1.96 * se) if se is not None else None, n=n)
