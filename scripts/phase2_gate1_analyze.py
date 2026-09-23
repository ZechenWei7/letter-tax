"""第二阶段闸门 1 判读（CPU）。读 6 个 run 的逐题记录，按 configs/phase2_gate1.yaml 的 analysis 段给出读法。
判读逻辑在 cot_compress.gate1.classify（每个分支在 tests/test_gate1.py 里都有构造样例）；本脚本只负责组装数据与报告。
配对键 = (题 id, seed)，两个 seed 合并。N3 主条件 = native_letterfree；N3u = native_none。
A1 的 91.2% 只作次要参照，不进判读。
用法：python scripts/phase2_gate1_analyze.py [--root runs] [--out results/phase2_gate1_verdict.json]"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.environ.setdefault("COT_NO_UNSLOTH", "1")
import argparse, json
import yaml
from cot_compress import gate1 as G
from cot_compress.archive import read_jsonl_zst

A1_REFERENCE_ACC = 0.912       # stage-1 A1 收敛（step 350，停止集）——只作参照


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(ROOT / "runs"))
    ap.add_argument("--gate-config", default="configs/phase2_gate1.yaml"); ap.add_argument("--out", default=str(ROOT / "results/phase2_gate1_verdict.json"))
    a = ap.parse_args()
    g = yaml.safe_load(open(ROOT / a.gate_config))
    rows, summ = {}, {}
    for N in g["notations"]:
        for s in g["seeds"]:
            d = pathlib.Path(a.root) / f"gate1_{N}_s{s}"
            summ[f"{N}_s{s}"] = json.load(open(d / "gate1_eval.json"))
            for r in read_jsonl_zst(d / "gate1_rows.jsonl.zst"):
                rows.setdefault((N, r["cond"]), {})[(r["id"], r["seed"])] = r
    def acc(N, cond): return {k: int(r["correct"]) for k, r in rows[(N, cond)].items()}
    def rate(N, cond, field):
        v = rows[(N, cond)].values(); return sum(bool(r[field]) for r in v) / len(v)
    X = dict(N2=acc("N2", "native_none"), N2s=acc("N2s", "native_none"), N3m=acc("N3", "native_letterfree"), N3u=acc("N3", "native_none"))
    premise = dict(N2_acc=rate("N2", "native_none", "correct"), N2_exact=rate("N2", "native_none", "exact"),
                   N2_transplant_acc=rate("N2", "transplant_gold", "correct"))
    v = G.classify(X, premise)
    table = {}
    for N, conds in (("N2", ["native_none"]), ("N2s", ["native_none"]), ("N3", ["native_letterfree", "native_none"])):
        for c in conds:
            rs = rows[(N, c)].values(); n = len(rs)
            table[f"{N}:{c}"] = {k: sum(bool(r[k]) for r in rs) / n for k in ("correct", "exact", "parsed", "sound", "loop", "capped")} | dict(n=n)
        for c in ("direct", "transplant_gold", "transplant_own", "tamper_control", "tamper"):
            rs = list(rows.get((N, c), {}).values())
            if rs: table[f"{N}:{c}"] = dict(n=len(rs), acc=sum(r["correct"] for r in rs) / len(rs))
        tt = list(rows.get((N, "tamper"), {}).values())
        if tt:
            ment = [r for r in tt if r["uses_flip"] or r["uses_orig"]]
            table[f"{N}:tamper"] |= dict(follows_flip_among_mentioning=(sum(r["follows_flip"] for r in ment) / len(ment) if ment else None),
                                         n_mentioning=len(ment), answer_order_flip=sum(r["answer_order"] == "flip" for r in tt) / len(tt),
                                         delta_vs_control=table[f"{N}:tamper"]["acc"] - table[f"{N}:tamper_control"]["acc"])
            cc = list(rows.get((N, "tamper_control"), {}).values()); mc = [r for r in cc if r["uses_flip"] or r["uses_orig"]]
            table[f"{N}:tamper_control"] |= dict(follows_flip_among_mentioning=(sum(r["follows_flip"] for r in mc) / len(mc) if mc else None),
                                                 n_mentioning=len(mc), answer_order_flip=sum(r["answer_order"] == "flip" for r in cc) / len(cc))
    out = dict(reading=v["reading"], text=v["text"], premise=premise, premise_checks=v["premise_checks"], premise_ok=v["premise_ok"],
               subreadings=v["subreadings"], comparisons=v["comparisons"], per_condition_pooled=table, per_run=summ,
               reference_only=dict(A1_converged_acc_stop_split=A1_REFERENCE_ACC, note="次要参照，不进判读"),
               rules=g["analysis"])
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False, default=list)
    print(f"读法：{v['reading']} —— {v['text']}")
    print("前提 P：", {k: round(x, 3) for k, x in premise.items()}, v["premise_checks"])
    for k, c in v["comparisons"].items():
        print(f"  {k:12} n={c['n']} diff={c['diff']:+.3f} 90%CI=({c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}) p={c['p_mcnemar']:.3g} 等效={c['equivalent']} 显著更差={c['sig_worse']} 显著更好={c['sig_better']}")
    if v["subreadings"]: print("  R2 子读法：", v["subreadings"])
    print("各条件（两 seed 合并）："); [print(f"  {k:28} {json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in t.items()}, ensure_ascii=False)}") for k, t in table.items()]
    print(f"参照（不进判读）：A1 收敛准确率 {A1_REFERENCE_ACC}")


if __name__ == "__main__":
    main()
