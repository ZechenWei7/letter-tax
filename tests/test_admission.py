"""§4.2 准入八条（用户 2026-09-18 定稿；纯逻辑）。"""
import random
from cot_compress.admission import admission_decision, post_training_check, next_cell, prescreen, derangement, CELL_ORDER

def base():
    return dict(chance=1 / 1680, d=5, direct=0.03, direct_n=2000, direct_lenient=0.035, native=0.70, native_lenient=0.72, M=3000,
                budget_curve={"0.1": 0.05, "0.25": 0.15, "0.5": 0.40, "0.75": 0.60, "1.0": 0.68},
                filler_curve={"0.1": 0.02, "0.25": 0.03, "0.5": 0.05, "0.75": 0.05, "1.0": 0.06},
                transplant_acc=0.10, template_hit_rate=0.01, masked_frozen_acc=0.03,
                kahn_lb={"2": 24.0, "2.5": 30.0, "3": 36.0}, decision_lb={"2": 26.0, "2.5": 32.5, "3": 39.0}, S_dist={"1": 450, "2": 50},
                pairwise_direct=0.66, pairwise_baseline=0.67)

def test_all_pass():
    d = admission_decision(base())
    assert d["admitted"] and not d["pending"] and len([k for k in d["checks"] if k[0] != "0"]) == 8
    c8 = d["checks"]["8_M_ge_5x_kahn_c2.5"]; assert c8["ratio_kahn_c2_5"] == 100 and c8["ratio_decision_c2_5"] is not None
    c1 = d["checks"]["1_direct_le_2x_2^-d"]
    assert c1["threshold"] == 2 * 2 ** -5 == 0.0625 and c1["pairwise_baseline"] == 0.67 and c1["chance_1_over_LE_reported_only"] == 1 / 1680
    assert d["checks"]["2_native_in_window"]["native_lenient"] == 0.72

def test_each_criterion_fails():
    for k, v in [("direct", 0.07), ("native", 0.85), ("transplant_acc", 0.40), ("template_hit_rate", 0.05), ("masked_frozen_acc", 0.30), ("M", 100.0)]:
        m = base(); m[k] = v
        assert not admission_decision(m)["admitted"], k
    m = base(); m["transplant_acc"] = 0.365                                  # native 0.70, direct 0.03 → 需掉 ≥ 0.335：掉 0.335 恰好通过
    assert admission_decision(m)["admitted"]
    m = base(); m["d"] = 7                                                   # 2×2^−7 = 0.0156 < direct 0.03
    assert not admission_decision(m)["admitted"]
    m = base(); m["budget_curve"]["0.25"] = 0.66
    assert not admission_decision(m)["admitted"]
    m = base(); m["filler_curve"]["1.0"] = 0.60
    assert not admission_decision(m)["admitted"]
    m = base(); m["filler_curve"]["0.25"] = 0.15
    assert admission_decision(m)["admitted"]                                 # <0.5M 不比较
    m = base(); m["S_dist"] = {"0": 1, "1": 499}
    assert not admission_decision(m)["admitted"]

def test_pending_when_shortcut_or_kahn_or_d_missing():
    m = base(); m["template_hit_rate"] = None; m["kahn_lb"] = None; m["d"] = None
    d = admission_decision(m)
    assert not d["admitted"] and set(d["pending"]) == {"1_direct_le_2x_2^-d", "7_shortcut_hits_lt_5pct", "8_M_ge_5x_kahn_c2.5"}

def test_derangement_has_no_fixed_points_and_is_a_permutation():
    rng = random.Random(0)
    for n in (2, 5, 500):
        p = derangement(n, rng); assert sorted(p) == list(range(n)) and all(p[i] != i for i in range(n))
    assert derangement(500, random.Random(1)) == derangement(500, random.Random(1))

def test_filler_only_compared_where_native_above_chance_plus_10pp():
    """D10：native 在 0.5M 处 ≈ chance 时该点不比较（否则填充≈0 与 native≈0 "10pp 内" 空成立）。"""
    m = base(); m["budget_curve"] = {"0.5": 0.05, "1.0": 0.68}; m["filler_curve"] = {"0.5": 0.05, "1.0": 0.06}
    d = admission_decision(m)
    assert d["checks"]["4_filler_not_within_10pp"]["compared_fracs"] == [1.0] and d["admitted"]
    m["filler_curve"]["1.0"] = 0.60
    assert not admission_decision(m)["admitted"]
    m = base(); m["budget_curve"] = {"0.5": 0.1005, "1.0": 0.68}; m["filler_curve"] = {"0.5": 0.05, "1.0": 0.06}     # chance 1/1680：0.1005 ≤ chance+0.10 → 不比
    assert admission_decision(m)["checks"]["4_filler_not_within_10pp"]["compared_fracs"] == [1.0]

def test_post_training_and_cell_order():
    assert post_training_check(0.11, 0.02)["ok"] and not post_training_check(0.13, 0.02)["ok"]
    assert CELL_ORDER == ("ord_n8_h4_d5", "ord_n8_h4_d6", "ord_n9_h5_d6", "ord_n9_h5_d7")
    assert next_cell({"ord_n8_h4_d5": dict(admitted=False), "ord_n8_h4_d6": dict(admitted=True)}) == "ord_n8_h4_d6"
    assert next_cell({"ord_n8_h4_d5": dict(admitted=True)}, cell_ok={"ord_n8_h4_d5": False}) is None
    assert prescreen({"a": {"_decision": {"ok": True}}, "b": {}}) == {"a": True, "b": False}
    assert next_cell({}) is None
