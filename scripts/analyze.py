"""汇总分析 v7（§8 端点）：读各 run 的 diagnostics.json / diagnostics_rows.jsonl（reporting-eval），输出 results/analysis_<key>.json + .md。
E1  匹配准确率下的税：每臂每 seed 的强制预算曲线（0.1…1.0 × 自身长度；y = 外化准确率 = acc − 训练后 direct；x = 正确轨迹 token 数），
    每 seed isotonic 单调化；y* = A/A″/B 全部达到的最高 y − 5pp；任一臂在 y* 的 5pp 内无点 → E1 不计算。
    税 = (L_A − L_B)/L_A 按 seed 配对 + 范围 + seed 级精确单边 Mann-Whitney；三种单位：token / code point / zstd-19 共享字典 bits。
    阈值 25% / 67%；<10% 无税；≤ −10% 负税单独报告。
    附加：冻结→A 的压缩量（A 各 seed step-0 eval 的 L vs 收敛 L）；A 主流策略类内的税；按题匹配的次要长度（同题 A/B 都答对的 L 比）；A 的字母占比曲线（"低" = <30%）。
E2  B_warm-SFT 的外化准确率在源 A 的 5pp 内才有效；长度比 L_Bwarm / L_A = 编码税。
E3  B > C_rand（每 seed 外化准确率）；移植与 unigram 重采样各使 B 外化准确率向训练后 direct 掉一半以上；复制率 < 30%；
    解码器（k=S，主格 k=1）在 B 上比移植 / C_rand 基线高 ≥ 10pp（results/decoder_<key>.json）；替换密码检查（B token 频率与 A 词频 Spearman ≥ 0.85 判为密码 → 阴性）。
其他：训练后逐臂 direct ≤ 冻结 direct + 10pp（不满足 → 该臂长度结果标记不可解释）；冷启动失败判定（B 各 seed 最佳外化准确率 < A 最佳 − 15pp）。
用法: python scripts/analyze.py --key kk_n10_s1 [--runs-dir runs] [--copy-thr 0.3]
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import argparse, json, re, statistics
from collections import Counter, defaultdict
from cot_compress.endpoints import (monotone_curve, length_at, matched_accuracy, tax_by_seed, zstd_dict_bits, cipher_check, mann_whitney_exact)
from cot_compress.traj_stats import think_text

ARMS_E1 = ("A", "A2", "B")

def load_runs(runs_dir, key):
    """返回 {arm: {seed: dict(diag=..., rows=[...], eval=[...], dir=...)}}；evalonly_Bwarm_<key>_s<seed>_sft → arm 'Bwarm_sft'。"""
    out = defaultdict(dict)
    for d in sorted(pathlib.Path(runs_dir).glob(f"*{key}_s*")):
        f = d / "diagnostics.json"
        if not f.exists(): continue
        m = re.match(rf"([A-Za-z0-9]+)_([0-9.]+)_{re.escape(key)}_s(\d+)$", d.name)
        m2 = re.match(rf"evalonly_Bwarm_{re.escape(key)}_s(\d+)_sft$", d.name)
        if m: arm, seed = m.group(1), int(m.group(3))
        elif m2: arm, seed = "Bwarm_sft", int(m2.group(1))
        else: continue
        rows = [json.loads(l) for l in open(d / "diagnostics_rows.jsonl")] if (d / "diagnostics_rows.jsonl").exists() else []
        ev = [json.loads(l) for l in open(d / "eval.jsonl")] if (d / "eval.jsonl").exists() else []
        out[arm][seed] = dict(diag=json.load(open(f)), rows=rows, eval=ev, dir=str(d), budget_items=load_budget_items(d))
    return out

def load_budget_items(d: pathlib.Path) -> dict | None:
    """diag_budget<f>.jsonl.zst → {f: {id: (correct, tokens)}}（按题配对 bootstrap 用）；没有则 None。"""
    from cot_compress.archive import read_jsonl_zst
    out = {}
    for f in sorted(d.glob("diag_budget*.jsonl.zst")):
        frac = f.name[len("diag_budget"):-len(".jsonl.zst")]
        out[frac] = {r["id"]: (bool(r.get("correct")), int(r.get("think_tokens") or 0)) for r in read_jsonl_zst(f)}
    return out or None

def curve_of(diag, ykey: str = "acc") -> list[tuple[float, float]]:
    """v8：y = 总准确率（ykey="acc"；"externalized" 并列），x = 正确轨迹 token 均值。"""
    pts = [(p["mean_tokens_correct"], p[ykey]) for p in diag.get("budget_points", {}).values() if p.get("mean_tokens_correct") is not None and p.get(ykey) is not None]
    return monotone_curve(pts) if pts else []

def unit_ratios(rows) -> dict:
    """正确轨迹的 code point / token 比。"""
    corr = [r for r in rows if r.get("correct") and r.get("think_tokens")]
    if not corr: return dict(cp_per_token=None)
    return dict(cp_per_token=statistics.mean(len(think_text(r["completion"])) / r["think_tokens"] for r in corr))

def frozen_direct_for(key):
    f = ROOT / f"results/admission_{key}.json"
    return json.load(open(f))["direct"] if f.exists() else None

def analyze(runs, key, copy_thr=0.3, decoder=None, frozen_direct=None):
    R = dict(key=key)
    R["endpoints"] = {a: {str(s): dict(L=v["diag"]["L_median"], acc=v["diag"]["acc"], direct=v["diag"]["direct_acc"], externalized=v["diag"]["acc"] - v["diag"]["direct_acc"])
                          for s, v in sv.items()} for a, sv in runs.items()}
    # 训练后逐臂 direct 检查
    R["frozen_direct"] = frozen_direct
    R["direct_check"] = {a: {str(s): dict(direct_post=v["diag"]["direct_acc"], uninterpretable=(frozen_direct is not None and v["diag"]["direct_acc"] > frozen_direct + 0.10))
                             for s, v in sv.items()} for a, sv in runs.items()}
    R["uninterpretable_arms"] = sorted({a for a, sv in R["direct_check"].items() if any(x["uninterpretable"] for x in sv.values())})
    for a, sv in runs.items():                                                               # v8：训练中残留检查触发的 run
        for s, v in sv.items():
            if (pathlib.Path(v["dir"]) / "uninterpretable.json").exists(): R["direct_check"][a][str(s)]["residual_stop"] = True
    dmeans = {a: statistics.mean(v["diag"]["direct_acc"] for v in sv.values()) for a, sv in runs.items() if a in ARMS_E1 and sv}
    R["direct_comparable_le_3pp"] = dict(ok=(bool(dmeans) and max(dmeans.values()) - min(dmeans.values()) <= 0.03), direct_means=dmeans)
    # ---- E1 ----
    from cot_compress.endpoints import bootstrap_tax, zstd_bits_per_token, tier_v8
    curves = {a: {s: curve_of(v["diag"]) for s, v in runs.get(a, {}).items()} for a in ARMS_E1 if a in runs}
    curves = {a: {s: c for s, c in sv.items() if c} for a, sv in curves.items()}
    ma = matched_accuracy(curves)
    E1 = dict(matched=ma, curves={a: {str(s): c for s, c in sv.items()} for a, sv in curves.items()},
              curves_externalized={a: {str(s): curve_of(v["diag"], "externalized") for s, v in runs.get(a, {}).items()} for a in ARMS_E1 if a in runs},
              unconditional_tokens={a: {str(s): {f: p.get("mean_tokens_all") for f, p in v["diag"].get("budget_points", {}).items()} for s, v in runs.get(a, {}).items()} for a in ARMS_E1 if a in runs})
    E1["no_match_table"] = {a: {str(s): dict(top_y=max(y for _, y in c), L_at_top=max(c, key=lambda p: p[1])[0], L_median=runs[a][s]["diag"]["L_median"], acc=runs[a][s]["diag"]["acc"])
                                for s, c in sv.items()} for a, sv in curves.items()}
    if ma["computable"]:
        y = ma["y_star"]
        # 单位换算系数（每臂 seed）：code point / token；zstd bits / token（并集字典、各臂字典）
        segs = {}; toks = {}
        for a in ARMS_E1:
            for s, v in runs.get(a, {}).items():
                corr = [r for r in v["rows"] if r.get("correct") and r.get("think_tokens")]
                segs[f"{a}_{s}"] = [think_text(r["completion"]) for r in corr]; toks[f"{a}_{s}"] = [r["think_tokens"] for r in corr]
        zb_union = zstd_bits_per_token(segs, toks) if any(segs.values()) else {}
        zb_own = zstd_bits_per_token(segs, toks, per_group_dict=True) if any(segs.values()) else {}
        cp = {g: (statistics.mean(len(t) for t in segs[g]) / statistics.mean(toks[g])) if segs[g] else None for g in segs}
        scales = {"tokens": None,
                  "code_points": {a: {s: cp.get(f"{a}_{s}") or 1.0 for s in runs[a]} for a in ARMS_E1 if a in runs},
                  "zstd_bits_union_dict": {a: {s: zb_union.get(f"{a}_{s}") or 1.0 for s in runs[a]} for a in ARMS_E1 if a in runs},
                  "zstd_bits_own_dict": {a: {s: zb_own.get(f"{a}_{s}") or 1.0 for s in runs[a]} for a in ARMS_E1 if a in runs}}
        per_item = {a: {s: v["budget_items"] for s, v in runs[a].items() if v.get("budget_items")} for a in ARMS_E1 if a in runs}
        ids = sorted(set.intersection(*[set(next(iter(pi.values())).keys()) for sv in per_item.values() for pi in sv.values() if pi])) if all(per_item.get(a) for a in ARMS_E1) else []
        tax = {}
        if ids:
            for unit, sc in scales.items():
                tax[unit] = bootstrap_tax(per_item, ids, n_boot=200, scale=sc)
        else:
            E1["note"] = "no per-item budget files (diag_budget*.jsonl.zst); tax from aggregated curves only"
            L = {a: {s: length_at(c, y) for s, c in sv.items()} for a, sv in curves.items()}
            for unit, sc in scales.items():
                LA = {s: L["A"][s] * (sc["A"][s] if sc else 1) for s in L.get("A", {}) if L["A"][s]}; LB = {s: L["B"][s] * (sc["B"][s] if sc else 1) for s in L.get("B", {}) if L["B"][s]}
                t = tax_by_seed(LA, LB) if LA and LB else None
                tax[unit] = dict(point=(t["mean"] if t else None), per_seed=(t["per_seed"] if t else None), tier=tier_v8(t["mean"] if t else None), mann_whitney=(t["mann_whitney_LB_lt_LA"] if t else None), L=L)
        # seed 级 Mann-Whitney 敏感性（token）
        Lt = (tax.get("tokens") or {}).get("L") or {}
        if Lt.get("A") and Lt.get("B"):
            E1["mann_whitney_sensitivity"] = mann_whitney_exact([v for v in Lt["B"].values() if v], [v for v in Lt["A"].values() if v], "less")
        tiers = {u: t.get("tier") for u, t in tax.items()}
        E1["tax"] = tax; E1["tiers"] = tiers
        E1["claim"] = dict(same_tier_all_units=(len({tiers.get(u) for u in ("tokens", "code_points", "zstd_bits_union_dict")}) == 1), tier=(tiers.get("tokens") if len({tiers.get(u) for u in ("tokens", "code_points", "zstd_bits_union_dict")}) == 1 else None))
    else:
        E1["verdict"] = "E1 不可计算：" + str(ma.get("reason"))
    # 附加
    extra = {}
    if "A" in runs:
        comp = {}
        for s, v in runs["A"].items():
            e0 = next((e for e in v["eval"] if e["step"] == 0), None)
            if e0: comp[str(s)] = dict(L_frozen=e0["L_median"], L_converged=v["diag"]["L_median"], compression=1 - v["diag"]["L_median"] / max(e0["L_median"], 1))
        extra["frozen_to_A_compression"] = comp
        extra["A_letter_frac_curve"] = {str(s): [(e["step"], e.get("letter_frac")) for e in v["eval"] if e.get("letter_frac") is not None] for s, v in runs["A"].items()}
        extra["A_letter_frac_low"] = {str(s): (v["diag"]["descriptives"].get("letter_frac") is not None and v["diag"]["descriptives"]["letter_frac"] < 0.30) for s, v in runs["A"].items()}
        cls = Counter()
        for v in runs["A"].values(): cls.update(r["strategy"]["strategy_class"] for r in v["rows"] if r.get("strategy") and r.get("correct"))
        if cls and "B" in runs:
            dom = cls.most_common(1)[0][0]
            def medL(v):
                xs = [r["think_tokens"] for r in v["rows"] if r.get("correct") and r.get("strategy", {}).get("strategy_class") == dom]
                return statistics.median(xs) if xs else None
            LA = {s: x for s, x in ((s, medL(v)) for s, v in runs["A"].items()) if x}; LB = {s: x for s, x in ((s, medL(v)) for s, v in runs["B"].items()) if x}
            extra["tax_within_A_dominant_class"] = dict(dominant_class=dom, tax=(tax_by_seed(LA, LB) if LA and LB else None))
        if "B" in runs:
            per = {}
            for s in set(runs["A"]) & set(runs["B"]):
                la = {r["id"]: r["think_tokens"] for r in runs["A"][s]["rows"] if r.get("correct")}; lb = {r["id"]: r["think_tokens"] for r in runs["B"][s]["rows"] if r.get("correct")}
                common = [i for i in la if i in lb and la[i]]
                if common: per[str(s)] = dict(n_items=len(common), median_ratio_B_over_A=statistics.median(lb[i] / la[i] for i in common))
            extra["per_item_matched_length"] = per
        # 对两个下限的比（c=2.5）
        lbs = {}
        for a, sv in runs.items():
            for s, v in sv.items():
                d = v["diag"]
                if d.get("decision_lb_tokens_median") and d.get("kahn_lb_tokens_median"):
                    lbs[f"{a}_{s}"] = dict(L_median=d["L_median"], over_decision_lb=d["L_median"] / d["decision_lb_tokens_median"], over_kahn_lb=d["L_median"] / d["kahn_lb_tokens_median"])
        extra["ratio_to_lower_bounds"] = lbs
    R["E1"] = E1; R["E1_extra"] = extra
    # ---- E2（§4.5）：B_warm-SFT 总准确率在源 A 5pp 内才有效，长度比 = 编码税，按 A seed 报；否则"词承载结构或映射有损" ----
    E2 = {}
    if "Bwarm_sft" in runs and "A" in runs:
        for s, v in runs["Bwarm_sft"].items():
            src = runs["A"].get(s)
            if src is None: E2[str(s)] = dict(valid=None, note="no A run with this seed"); continue
            aa, bb = src["diag"]["acc"], v["diag"]["acc"]; valid = abs(aa - bb) <= 0.05
            E2[str(s)] = dict(A_seed=s, acc_A=aa, acc_Bwarm_sft=bb, valid=valid, length_ratio=v["diag"]["L_median"] / max(src["diag"]["L_median"], 1),
                              encoding_tax=(1 - v["diag"]["L_median"] / max(src["diag"]["L_median"], 1)) if valid else None,
                              verdict=("encoding tax" if valid else "词承载结构或映射有损（B_warm-SFT 准确率不在源 A 5pp 内）"))
    R["E2"] = E2
    # 冷启动失败（run_matrix 的 200 步判定另存 results/warm_decision_<key>.json；这里按收敛值报）
    if "A" in runs and "B" in runs:
        bA = max(v["diag"]["acc"] for v in runs["A"].values()); bB = max(v["diag"]["acc"] for v in runs["B"].values())
        R["cold_start_failure_converged"] = dict(best_A=bA, best_B=bB, failure=(bB < bA - 0.15))
    # ---- E3（§4.5，对 B 必需）----
    E3 = {}
    if "B" in runs:
        from cot_compress.endpoints import bootstrap_tax
        # (i) B > C_rand：匹配长度下的准确率，按题配对、seed 分块（C_rand 的 think 长度取自 B 的分布 → 同题配对比较正确率）
        if "Crand" in runs:
            per_seed, diffs = {}, []
            for s in sorted(set(runs["B"]) & set(runs["Crand"])):
                cb = {r["id"]: bool(r["correct"]) for r in runs["B"][s]["rows"]}; cc = {r["id"]: bool(r["correct"]) for r in runs["Crand"][s]["rows"]}
                ids = sorted(set(cb) & set(cc))
                if not ids: continue
                d = [cb[i] - cc[i] for i in ids]; per_seed[str(s)] = dict(n=len(ids), diff=statistics.mean(d)); diffs.append((ids, cb, cc))
            if diffs:
                import random as _r
                rng = _r.Random(0); boots = []
                for _ in range(500):
                    vals = []
                    for ids, cb, cc in diffs:
                        sel = [ids[rng.randrange(len(ids))] for _ in ids]; vals.append(statistics.mean(cb[i] - cc[i] for i in sel))
                    boots.append(statistics.mean(vals))
                boots.sort(); lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
                E3["B_gt_Crand_paired"] = dict(ok=lo > 0, diff=statistics.mean(x["diff"] for x in per_seed.values()), lo=lo, hi=hi, per_seed=per_seed,
                                               matched_length_note="C_rand 的 think 长度从 B 收敛分布抽取（同题优先），因此按题配对即匹配长度")
        # (ii) 移植与 unigram 重采样各使 B 准确率向训练后 direct 掉一半以上
        drops = {}
        for s, v in runs["B"].items():
            d = v["diag"]; gap = d["acc"] - d["direct_acc"]
            drops[str(s)] = {k: dict(acc=d[k], closed_frac=((d["acc"] - d[k]) / gap if gap > 0 else None)) for k in ("diag2_transplant_acc", "diag3_unigram_resample_acc") if k in d}
        E3["transplant_and_unigram_close_gt_half"] = dict(ok=all(x["closed_frac"] is not None and x["closed_frac"] > 0.5 for sv in drops.values() for x in sv.values()), per_seed=drops)
        # (iii) 复制率 < 30%
        cr = [v["diag"]["descriptives"].get("copy_rate_mean", v["diag"]["descriptives"].get("prompt_copy_rate_4gram")) for v in runs["B"].values()]
        cr = [x for x in cr if x is not None]
        E3["copy_rate_lt_30pct"] = dict(ok=(bool(cr) and max(cr) < copy_thr), per_seed=cr, thr=copy_thr)
        # 解码器（results/decoder_<key>.json，scripts/08_decoder.py）：仪器门 + B 条款（解析率 ≥50% 才生效）
        if decoder:
            g = decoder.get("instrument_gate") or {}
            E3["decoder"] = dict(instrument_ok=g.get("ok"), per_arm=g.get("per_arm"), B_clause_active=((decoder.get("arms", {}).get("B", {}).get("parse", {}) or {}).get("parse_rate") or 0) >= 0.5,
                                 parse_rates={a: (v.get("parse") or {}).get("parse_rate") for a, v in (decoder.get("arms") or {}).items()},
                                 descriptive=dict(cipher_spearman_warm=decoder.get("cipher_spearman_warm"),
                                                  soundness={a: v.get("soundness_mean") for a, v in (decoder.get("arms") or {}).items()},
                                                  completeness={a: v.get("completeness_mean") for a, v in (decoder.get("arms") or {}).items()}))
        E3["verdict"] = all(v["ok"] for k, v in E3.items() if isinstance(v, dict) and "ok" in v)
    R["E3"] = E3
    return R

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key", required=True); ap.add_argument("--runs-dir", default="runs"); ap.add_argument("--copy-thr", type=float, default=0.3)
    args = ap.parse_args()
    runs = load_runs(ROOT / args.runs_dir, args.key)
    dec = ROOT / f"results/decoder_{args.key}.json"; decoder = json.load(open(dec)) if dec.exists() else None
    R = analyze(runs, args.key, args.copy_thr, decoder=decoder, frozen_direct=frozen_direct_for(args.key))
    out = ROOT / f"results/analysis_{args.key}.json"; json.dump(R, open(out, "w"), indent=1, default=str)
    md = ROOT / f"results/analysis_{args.key}.md"
    with open(md, "w") as f:
        tk = (R['E1'].get('tax') or {}).get('tokens') or {}
        f.write(f"# analysis {args.key}\n\nE1 computable={R['E1']['matched'].get('computable')} y*={R['E1']['matched'].get('y_star')} "
                f"tax_tokens={tk.get('point')} [{tk.get('lo')}, {tk.get('hi')}] tiers={json.dumps(R['E1'].get('tiers'), default=str)} claim={json.dumps(R['E1'].get('claim'), default=str)}\n"
                f"{R['E1'].get('verdict', '')}\nno_match_table={json.dumps(R['E1'].get('no_match_table'), default=str)}\nE2={json.dumps(R['E2'], default=str)}\nE3 verdict={R['E3'].get('verdict')}\n\n"
                f"| arm | seed | L_med | acc | direct | externalized |\n|---|---|---|---|---|---|\n")
        for a, v in R["endpoints"].items():
            for s, d in v.items(): f.write(f"| {a} | {s} | {d['L']} | {d['acc']:.3f} | {d['direct']:.3f} | {d['externalized']:.3f} |\n")
    print(json.dumps(R, indent=1, default=str)); print("wrote", out, md)

if __name__ == "__main__":
    main()
