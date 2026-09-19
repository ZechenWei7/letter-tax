"""analyze.py v7（E1/E2/E3、冷启动、direct 检查）与 run_matrix.py（裁剪、门、重调）的纯逻辑，用合成 run 目录。"""
import importlib.util, pathlib, json, random, pytest

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def _points(L, top, direct):
    pts = {}
    for i in range(1, 11):
        f = round(0.1 * i, 1); acc = min(top, top * (f / 0.6)) if f < 0.6 else top
        pts[str(f)] = dict(acc=acc, externalized=acc - direct, mean_tokens_correct=f * L, mean_tokens_all=f * L, n_correct=50)
    return pts

def _budget_files(d, L, top, n_items, rng):
    """diag_budget<f>.jsonl.zst：每题在各预算下的 correct / token（按题配对 bootstrap 用）。"""
    from cot_compress.archive import append_jsonl_zst
    for i in range(1, 11):
        f = round(0.1 * i, 1); pacc = min(top, top * (f / 0.6)) if f < 0.6 else top
        append_jsonl_zst(d / f"diag_budget{f}.jsonl.zst", [dict(id=f"kk/report/0/{j}", correct=(rng.random() < pacc), think_tokens=int(f * L * rng.uniform(0.8, 1.2))) for j in range(n_items)])

def _think(kind, rng, n=10):
    if kind == "A":
        return " ".join(f"assume person {rng.randint(1, n)} is a knight then person {rng.randint(1, n)} is a knave so" for _ in range(6))
    return " ".join(f"» {rng.randint(1, n)} 1 → {rng.randint(1, n)} 0 ⇒" for _ in range(6))

def _write_run(root, arm, seed, key, acc, L, direct=0.05, transplant=0.10, unigram=0.12, copy=0.1, kind="A", eval0=None, letter=0.8, n_rows=60):
    d = root / (f"{arm}_0.5_{key}_s{seed}" if arm != "Bwarm_sft" else f"evalonly_Bwarm_{key}_s{seed}_sft"); d.mkdir(parents=True)
    diag = dict(acc=acc, L_median=L, direct_acc=direct, budget_points=_points(L, acc, direct), diag2_transplant_acc=transplant, diag3_unigram_resample_acc=unigram,
                descriptives=dict(copy_rate_mean=copy, letter_frac=letter, strategy_class={"case_split": 0.8, "other": 0.2}),
                decision_lb_tokens_median=32.5, kahn_lb_tokens_median=30.0)
    json.dump(diag, open(d / "diagnostics.json", "w"))
    rng = random.Random(seed); _budget_files(d, L, acc, 120, rng)
    with open(d / "diagnostics_rows.jsonl", "w") as f:
        for i in range(n_rows):
            corr = rng.random() < acc
            f.write(json.dumps(dict(id=f"kk/report/0/{i}", correct=corr, think_tokens=int(L * rng.uniform(0.7, 1.3)),
                                    completion=_think(kind, rng) + "\n</think>\n\n<answer>1 0</answer>", strategy=dict(strategy_class="case_split"))) + "\n")
    if eval0 is not None:
        with open(d / "eval.jsonl", "w") as f:
            f.write(json.dumps(dict(step=0, L_median=eval0, L_mean=eval0, acc=0.7, letter_frac=0.85)) + "\n")
            f.write(json.dumps(dict(step=100, L_median=L, L_mean=L, acc=acc, letter_frac=letter)) + "\n")
    return d

@pytest.fixture
def synth_runs(tmp_path):
    key = "kk_n10_s1"
    for s, (acc, L) in {1: (0.80, 1500), 2: (0.78, 1550), 3: (0.79, 1520)}.items(): _write_run(tmp_path, "A", s, key, acc, L, kind="A", eval0=3000)
    for s, (acc, L) in {1: (0.76, 1200), 2: (0.77, 1250)}.items(): _write_run(tmp_path, "A2", s, key, acc, L, kind="A")
    for s, (acc, L) in {1: (0.75, 600), 2: (0.74, 620), 3: (0.76, 610)}.items(): _write_run(tmp_path, "B", s, key, acc, L, kind="B")
    for s, (acc, L) in {1: (0.30, 600), 2: (0.32, 620)}.items(): _write_run(tmp_path, "Crand", s, key, acc, L, kind="B")
    _write_run(tmp_path, "Bwarm_sft", 1, key, 0.77, 700, kind="B")
    return tmp_path, key

def test_analyze_e1_e2_e3(synth_runs):
    az = _load("az", "scripts/analyze.py"); root, key = synth_runs
    runs = az.load_runs(root, key)
    assert set(runs) == {"A", "A2", "B", "Crand", "Bwarm_sft"} and set(runs["A"]) == {1, 2, 3}
    dec = dict(instrument_gate=dict(ok=True, per_arm={"A": dict(ok=True)}), arms={"A": dict(parse=dict(parse_rate=0.9), soundness_mean=0.95), "B": dict(parse=dict(parse_rate=0.6), soundness_mean=0.8)}, cipher_spearman_warm=dict(spearman=0.3))
    R = az.analyze(runs, key, decoder=dec, frozen_direct=0.02)
    E1 = R["E1"]
    assert E1["matched"]["computable"] and E1["matched"]["y_star"] == pytest.approx(min((0.80 + 0.78 + 0.79) / 3, (0.76 + 0.77) / 2, (0.75 + 0.74 + 0.76) / 3) - 0.05)   # r3：各臂 seed 均值取 min − 5pp
    tk = E1["tax"]["tokens"]
    assert set(tk["per_seed"]) <= {1, 2, 3} and tk["tier_point"] in ("moderate", "strong") and 0.4 < tk["point"] < 0.8 and tk["lo"] <= tk["point"] <= tk["hi"]
    assert set(E1["tier_spans"]) == set(E1["tax"]) == set(E1["tier_points"]) and E1["tier_points"]["tokens"] == tk["tier_point"] and len(E1["claim"]["point_tiers"]) == 3
    assert E1["mann_whitney_sensitivity"]["p"] == pytest.approx(0.05)
    for u in ("code_points", "zstd_bits_union_dict", "zstd_bits_own_dict"): assert E1["tax"][u]["point"] is not None
    assert E1["claim"]["same_tier_all_units"] in (True, False) and set(E1["no_match_table"]) == {"A", "A2", "B"}
    assert R["E1_extra"]["ratio_to_lower_bounds"]["A_1"]["over_decision_lb"] == pytest.approx(1500 / 32.5)
    assert R["direct_comparable_le_3pp"]["ok"] and R["direct_comparable_le_3pp"]["label"] == "comparable"
    assert R["E1_extra"]["frozen_to_A_compression"]["1"]["compression"] == pytest.approx(0.5)
    assert R["E1_extra"]["tax_within_A_dominant_class"]["dominant_class"] == "case_split"
    assert "1" in R["E1_extra"]["per_item_matched_length"] and R["E1_extra"]["A_letter_frac_low"]["1"] is False
    assert R["E2"]["1"]["valid"] and R["E2"]["1"]["length_ratio"] == pytest.approx(700 / 1500) and R["E2"]["1"]["verdict"] == "encoding tax"
    assert R["cold_start_failure_converged"]["failure"] is False
    E3 = R["E3"]
    assert E3["B_gt_Crand_paired"]["ok"] and E3["B_gt_Crand_paired"]["lo"] > 0 and set(E3["B_gt_Crand_paired"]["per_seed"]) == {"1", "2"}
    assert E3["transplant_and_unigram_close_gt_half"]["ok"] and E3["copy_rate_lt_30pct"]["ok"]
    assert E3["decoder"]["instrument_ok"] and E3["decoder"]["B_clause_active"] and E3["verdict"]
    assert R["uninterpretable_arms"] == []

def test_analyze_uncomputable_and_flags(synth_runs):
    az = _load("az", "scripts/analyze.py"); root, key = synth_runs
    runs = az.load_runs(root, key)
    for v in runs["B"].values(): v["diag"]["direct_acc"] = 0.20                                   # B 的 direct 冒到 20%
    runs2 = {k: v for k, v in runs.items() if k != "A2"}                              # 缺 A″ → E1 不可计算，输出无匹配表
    R = az.analyze(runs2, key, frozen_direct=0.02)
    assert not R["E1"]["matched"]["computable"] and "tax" not in R["E1"] and R["E1"]["verdict"].startswith("E1 不可计算") and "own_budget_acc" in R["E1"]["no_match_table"]["A"]["1"]
    assert R["uninterpretable_arms"] == ["B"] and R["direct_comparable_le_3pp"]["label"] == "not comparable" and R["E1"]["comparability"] == "not comparable"

def test_run_matrix_prune_gate_retune(tmp_path):
    rm = _load("rm", "scripts/run_matrix.py")
    runs = [dict(arm="A", seed=1, priority=99), dict(arm="B", seed=1, priority=99), dict(arm="Crand", seed=1, priority=1), dict(arm="A2", seed=3, priority=2), dict(arm="B", seed=5, priority=3)]
    assert [r["arm"] for r in rm.prune(runs, 3)] == ["A", "B", "B"] and [r["arm"] for r in rm.prune(runs, 2)] == ["A", "B"]
    with open(tmp_path / "eval.jsonl", "w") as f:
        f.write(json.dumps(dict(step=0, L_mean=3000, L_median=3000, acc=0.70)) + "\n"); f.write(json.dumps(dict(step=200, L_mean=1800, L_median=1700, acc=0.68)) + "\n")
    g = rm.first_run_gate(tmp_path, oracle_lb=30.0, kill_lb=32.5)
    assert g["ok"] and g["token_reduction"] == pytest.approx(0.4) and g["above_2x_kahn"] and g["kill"] is False
    assert not rm.first_run_gate(tmp_path, oracle_lb=900.0)["ok"]                                  # 收敛 L 1700 < 2×900（Kahn）
    k = rm.first_run_gate(tmp_path, oracle_lb=30.0, kill_lb=1200.0); assert k["kill"] and not k["ok"]  # 1700 ≤ 1.5×1200（决策下限）
    assert rm.first_run_gate(tmp_path, oracle_lb=30.0, kill_lb=1700 / 1.5)["kill"] and not rm.first_run_gate(tmp_path, oracle_lb=30.0, kill_lb=1133.0)["kill"]   # 边界取等
    with open(tmp_path / "eval.jsonl", "a") as f: f.write(json.dumps(dict(step=250, L_mean=2500, L_median=2400, acc=0.69)) + "\n")
    assert not rm.first_run_gate(tmp_path, oracle_lb=27.5)["ok"]
    import yaml
    mx = yaml.safe_load(open("configs/matrix_core.yaml"))["runs"]; order = [(r["arm"], int(r["seed"])) for r in mx]
    assert order.index(("Bwarm_sft", 1)) == order.index(("A", 1)) + 1 and order.index(("Bwarm_sft", 2)) == order.index(("A", 2)) + 1 and order.index(("Bwarm_sft", 3)) == order.index(("A", 3)) + 1
    assert all(r.get("set") == ["train.crand_source=auto"] for r in mx if r["arm"] == "Crand")                  # C_rand seed i 绑 B seed i
    bp = rm.budget_projection("configs/cloud_4b.yaml", 2); assert bp["budget_usd"] == 300 and bp["n_extra_runs"] == 2 and bp["ok"] in (None, True)   # 本地无 dry-run 投影 → None
    assert [x["tag"] for x in rm.RETUNE] == ["_lam0.3", "_lam1.0", "_G32"] and rm.run_name(dict(arm="Bwarm_sft", key="kk_n10_s1", seed=1)) == "Bwarm_sft_kk_n10_s1_s1"
