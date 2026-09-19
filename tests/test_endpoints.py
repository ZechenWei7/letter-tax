"""v7 §8 端点纯函数：isotonic、匹配准确率、税与 Mann-Whitney、zstd 字典比特、密码检查。"""
import random
from collections import Counter
import pytest
from cot_compress import endpoints as E

def test_isotonic_pava():
    assert E.isotonic_nondecreasing([1, 3, 2, 4]) == [1, 2.5, 2.5, 4]
    assert E.isotonic_nondecreasing([3, 2, 1]) == [2, 2, 2]
    assert E.isotonic_nondecreasing([1, 2, 3]) == [1, 2, 3]
    c = E.monotone_curve([(300, 0.5), (100, 0.1), (200, 0.4), (250, 0.3)])
    assert [x for x, _ in c] == [100, 200, 250, 300] and c[1][1] == c[2][1] == pytest.approx(0.35)

def test_length_at_interpolates():
    c = [(100, 0.1), (200, 0.5), (300, 0.5)]
    assert E.length_at(c, 0.3) == pytest.approx(150) and E.length_at(c, 0.5) == 200 and E.length_at(c, 0.6) is None and E.length_at(c, 0.05) == 100

def test_matched_accuracy_r3_own_budget_mean_and_crossing():
    cur = {"A": {1: [(100, 0.2), (400, 0.7)], 2: [(100, 0.3), (420, 0.8)]}, "A2": {1: [(100, 0.1), (400, 0.6)]}, "B": {1: [(50, 0.3), (200, 0.65)], 2: [(60, 0.2), (210, 0.50)]}}
    m = E.matched_accuracy(cur)
    assert m["tops"] == {"A": pytest.approx(0.75), "A2": pytest.approx(0.6), "B": pytest.approx(0.575)}     # 各 seed 自身预算（x 最大点）处 acc 的等权均值
    assert m["y_star"] == pytest.approx(0.525) and m["computable"] and m["non_crossing"]["B"] == [2]          # B seed 2 最高 0.50 < 0.525：不穿过，记下、不进税
    assert set(E.tax_from_curves(cur)["per_seed"]) == {1}
    none_cross = {"A": {1: [(100, 0.9)]}, "A2": {1: [(100, 0.9)]}, "B": {1: [(50, 0.1), (200, 0.2)], 2: [(50, 0.9), (200, 0.2)]}}
    # 注意：曲线是 isotonic 后的；这里直接给单调曲线。B 均值 0.2 → y* = 0.15，B 两个 seed 都达到 → 可计算
    assert E.matched_accuracy({k: ({s: E.monotone_curve(c) for s, c in v.items()}) for k, v in none_cross.items()})["computable"]
    low = {"A": {1: [(100, 0.05), (400, 0.10)]}, "A2": {1: [(100, 0.9)]}, "B": {1: [(50, 0.9)]}}
    m = E.matched_accuracy(low); assert m["y_star"] == pytest.approx(0.05) and m["computable"]                  # 曲线起点已 ≥ y* 也算达到
    assert not E.matched_accuracy({"A": {}, "A2": {1: []}, "B": {1: []}})["computable"]

def test_mann_whitney_exact_and_tax():
    mw = E.mann_whitney_exact([1, 2, 3], [4, 5, 6], "less")
    assert mw["U"] == 9 and mw["p"] == pytest.approx(1 / 20)                # U = #(a<b) 对；H1 a 小 → U 大
    assert E.mann_whitney_exact([4, 5, 6], [1, 2, 3], "less")["p"] == 1.0
    t = E.tax_by_seed({1: 1000, 2: 1200, 3: 900}, {1: 500, 2: 300, 3: 450})
    assert t["per_seed"]["1"] == 0.5 and t["level"] == "moderate" and t["mann_whitney_LB_lt_LA"]["p"] == pytest.approx(0.05)
    assert E.tax_level(0.7) == "strong" and E.tax_level(0.05) == "none" and E.tax_level(-0.2) == "negative" and E.tax_level(0.15) == "small"
    assert E.tax_by_seed({1: 100}, {1: 120})["negative_seeds"] == ["1"]

def test_spearman():
    assert E.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0 and E.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert E.spearman([1, 2], [1, 2]) is None

def test_zstd_dict_bits_splits_and_measures():
    rng = random.Random(0)
    segs = {"A": [" ".join(rng.choice(["assume", "person", "knight", "then", "so"]) for _ in range(80)) for _ in range(40)],
            "B": [" ".join(rng.choice(["»", "3", "1", "→", "⇒"]) for _ in range(40)) for _ in range(40)]}
    r = E.zstd_dict_bits(segs, dict_size=4096)
    assert r["A"]["n"] + r["B"]["n"] == 40 and r["A"]["bits_mean"] > r["B"]["bits_mean"] > 0

def test_cipher_check_detects_substitution():
    rng = random.Random(1); words = [f"w{i}" for i in range(30)]; sym = {w: f"#{i}" for i, w in enumerate(words)}
    a, b = {}, {}
    for j in range(40):
        c = Counter(rng.choices(words, weights=[1 / (i + 1) for i in range(30)], k=60)); a[f"it{j}"] = c; b[f"it{j}"] = Counter({sym[w]: n for w, n in c.items()})
    r = E.cipher_check(a, b, top_k=20)
    assert r["applicable"] and r["is_cipher"] and r["spearman"] > 0.95
    b2 = {k: Counter(rng.choices([f"#{i}" for i in range(30)], k=60)) for k in a}
    r2 = E.cipher_check(a, b2, top_k=20)
    assert r2["applicable"] and not r2["is_cipher"]

# ---- v8 ----
def _per_item(n_items, acc_top, L, rng, offset=0):
    """合成每题预算结果：预算 f 下正确概率 = acc_top·min(1, f/0.6)，token = f·L·(1±0.2)。"""
    per = {}
    for i in range(1, 11):
        f = round(0.1 * i, 1); p = acc_top * min(1.0, f / 0.6); d = {}
        for j in range(n_items):
            d[f"it{j}"] = (rng.random() < p, int(f * L * rng.uniform(0.8, 1.2)))
        per[str(f)] = d
    return per

def test_bootstrap_tax_paired_seed_blocks():
    rng = random.Random(0); ids = [f"it{j}" for j in range(300)]
    runs = {"A": {1: _per_item(300, 0.8, 1500, rng), 2: _per_item(300, 0.78, 1550, rng), 3: _per_item(300, 0.79, 1500, rng)},
            "A2": {1: _per_item(300, 0.76, 1200, rng), 2: _per_item(300, 0.77, 1250, rng)},
            "B": {1: _per_item(300, 0.75, 600, rng), 2: _per_item(300, 0.74, 620, rng), 3: _per_item(300, 0.76, 610, rng), 4: _per_item(300, 0.75, 600, rng)}}
    r = E.bootstrap_tax(runs, ids, n_boot=60)
    assert r["point"] is not None and 0.4 < r["point"] < 0.8 and r["lo"] < r["point"] < r["hi"] and set(r["per_seed"]) == {1, 2, 3}
    assert r["tier_point"] in ("moderate", "strong") and r["skipped_frac"] < 0.5 and (r["tier"] is None or r["tier"] == r["tier_point"]) and r["tier_span"] is not None
    sc = {a: {s: 3.0 for s in sv} for a, sv in runs.items()}
    r2 = E.bootstrap_tax(runs, ids, n_boot=10, scale=sc)                      # 同比例换算不改税
    assert abs(r2["point"] - r["point"]) < 1e-9
    assert E.tax_from_curves({"A": {1: [(100, 0.9)]}, "A2": {1: [(100, 0.9)]}}) is None                     # 缺臂 → 不可计算

def test_tier_v8_and_zstd_units():
    assert [E.tier_v8(x) for x in (0.7, 0.67, 0.3, 0.25, 0.15, 0.10, 0.05, -0.05, -0.10, -0.2)] == ["strong", "strong", "moderate", "moderate", "small", "small", "none", "none", "negative", "negative"]
    assert E.tier_interval(0.30, 0.60) == dict(tier="moderate", span="moderate", lo_tier="moderate", hi_tier="moderate")
    assert E.tier_interval(0.20, 0.40)["tier"] is None and E.tier_interval(0.20, 0.40)["span"] == "small–moderate" and E.tier_interval(None, 0.4)["tier"] is None
    rng = random.Random(1)
    segs = {"A_1": [" ".join(rng.choice(["assume", "then", "so", "before", "after"]) for _ in range(60)) for _ in range(40)],
            "B_1": [" ".join(rng.choice(["»", "→", "⇒", "<", ">"]) for _ in range(30)) for _ in range(40)]}
    toks = {g: [len(s.split()) for s in v] for g, v in segs.items()}
    u = E.zstd_bits_per_token(segs, toks); p = E.zstd_bits_per_token(segs, toks, per_group_dict=True)
    assert u["A_1"] > 0 and u["B_1"] > 0 and p["A_1"] > 0 and p["B_1"] > 0
