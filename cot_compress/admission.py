"""准入判定（§4.2，用户 2026-09-18 定稿；冻结 Qwen3-4B，n=500，全部通过才准入），纯逻辑，可单测。
metrics 由 01_calibrate.py --admission 在 stopping-eval 500 题（split="stop"）上测得：
  direct（严格）, direct_lenient, native（严格，S≥1 总体）, native_lenient, chance = 1/#LE(hard)（格中位）, M（中位原生长度）,
  budget_curve {0.1,0.25,0.5,0.75,1.0}×M, filler_curve（白名单 unigram 填充）, transplant_acc（原生轨迹 i→j 预填只生成答案）,
  masked_frozen_acc（冻结 letter-ban acc_clean）, template_hit_rate（捷径检测器）, kahn_lb {c: 中位 token}, decision_lb {c: …}（只报）,
  pairwise_direct / pairwise_baseline（直接作答逐对准确率 vs 硬偏序均匀随机扩展，诊断）
八条：
  1 直接作答 exact ≤ 2 × chance（准入用严格抽取；宽松并报）
  2 原生 exact ∈ [60%, 80%]（S≥1 总体）
  3 强制预算曲线到 native−5pp 的最小预算 ≥ 0.5·M（没有点到 → 通过）
  4 白名单 unigram 填充在 ≥0.5M 任一预算不进 native 曲线 10pp 内
  5 原生轨迹移植 i→j 使准确率向 direct 掉一半以上：native − transplant ≥ (native − direct)/2
  6 冻结 letter-ban ≤ direct + 10pp
  7 捷径检测器命中 < 5%
  8 中位原生长度 M ≥ 5 × Kahn 下限（c=2.5）；决策下限与 c=2/3 只报
先按格统计规则筛（tasks/ordering.cell_decision），通过的格再上 GPU 做八项；候选格顺序 = CELL_ORDER。
训练后残留检查（train.py）：每 50 步 direct-check 2000 题，(带轨迹 − 直接) < 冻结差距/2 连续两次 → 停 run 标不可解释；E1 要求臂间训练后直接作答差 ≤3pp（analyze.py）。
"""
from __future__ import annotations

FRACS = (0.1, 0.25, 0.5, 0.75, 1.0)
CELL_ORDER = ("ord_n8_h4_d5", "ord_n8_h4_d6", "ord_n9_h5_d6", "ord_n9_h5_d7")     # v8；kk 格已退出主线
TEMPLATE_MAX = 0.05
ORACLE_MULT = 5.0
ORACLE_C = "2.5"
DIRECT_CHANCE_MULT = 2.0

def admission_decision(m: dict, native_lo: float = 0.60, native_hi: float = 0.80) -> dict:
    checks = {}
    chance = m["chance"]; native = m["native"]; direct = m["direct"]
    checks["1_direct_le_2x_chance"] = dict(ok=direct <= DIRECT_CHANCE_MULT * chance, direct=direct, direct_lenient=m.get("direct_lenient"), chance=chance,
                                           pairwise_direct=m.get("pairwise_direct"), pairwise_baseline=m.get("pairwise_baseline"))
    checks["2_native_in_window"] = dict(ok=native_lo <= native <= native_hi, native=native, native_lenient=m.get("native_lenient"))
    bc = {float(k): v for k, v in m["budget_curve"].items()}
    thr = native - 0.05
    reached = [f for f in sorted(bc) if bc[f] >= thr]
    min_budget = reached[0] if reached else None
    checks["3_min_budget_ge_0.5M"] = dict(ok=(min_budget is None or min_budget >= 0.5), min_budget_frac=(min_budget if min_budget is not None else ">1.0"), target=thr)
    fc = {float(k): v for k, v in m.get("filler_curve", {}).items()}
    comparable = sorted(f for f in fc if f >= 0.5 and f in bc)
    bad = [f for f in comparable if fc[f] >= bc[f] - 0.10]
    checks["4_filler_not_within_10pp"] = dict(ok=not bad, offending_fracs=bad, compared_fracs=comparable)
    drop = native - m["transplant_acc"]; need = (native - direct) * 0.5
    checks["5_transplant_drop_ge_half_to_direct"] = dict(ok=drop >= need, drop=drop, required=need)
    checks["6_masked_frozen_le_direct+10pp"] = dict(ok=m["masked_frozen_acc"] <= direct + 0.10, masked_frozen_acc=m["masked_frozen_acc"])
    t = m.get("template_hit_rate")
    checks["7_shortcut_hits_lt_5pct"] = dict(ok=(None if t is None else t < TEMPLATE_MAX), rate=t)
    kl = m.get("kahn_lb") or {}; dl = m.get("decision_lb") or {}
    lb = kl.get(ORACLE_C); M = m.get("M")
    checks["8_M_ge_5x_kahn_c2.5"] = dict(ok=(None if (lb is None or M is None) else M >= ORACLE_MULT * lb), M=M, kahn_lb_c2_5=lb,
                                         ratio_kahn_c2_5=((M / lb) if (lb and M) else None), kahn_lb=kl, decision_lb=dl,
                                         ratio_decision_c2_5=((M / dl["2.5"]) if (dl.get("2.5") and M) else None))
    info = dict(native_gloss=m.get("native_gloss"))
    sd = m.get("S_dist")
    if sd is not None and m.get("target_S") is not None:
        target = int(m["target_S"])
        checks["0_generator_S_eq_target"] = dict(ok=(set(map(int, sd.keys())) == {target}), S_dist=sd, target=target)
    elif sd is not None:
        checks["0_generator_S_ge_1"] = dict(ok=all(int(k) >= 1 for k in sd), S_dist=sd)
    pending = [k for k, v in checks.items() if v["ok"] is None]
    return dict(admitted=(all(v["ok"] for v in checks.values() if v["ok"] is not None) and not pending), pending=pending, checks=checks, info=info)

def post_training_check(direct_post: float, frozen_direct: float) -> dict:
    """训练后逐臂：direct ≤ 冻结 direct + 10pp。"""
    return dict(ok=direct_post <= frozen_direct + 0.10, direct_post=direct_post, frozen_direct=frozen_direct)

def next_cell(results: dict, cell_ok: dict | None = None) -> str | None:
    """按 CELL_ORDER 取第一个（格统计通过且）八项通过的格。"""
    for k in CELL_ORDER:
        if cell_ok is not None and not cell_ok.get(k, False): continue
        d = results.get(k)
        if d and d.get("admitted"):
            return k
    return None

def prescreen(cell_reports: dict) -> dict:
    """格统计预筛：{key: cell_report(含 _decision)} → {key: ok}。"""
    return {k: bool((r.get("_decision") or {}).get("ok")) for k, r in cell_reports.items()}
