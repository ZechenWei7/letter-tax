"""策略类检测器人工抽样审计（v8 §2）：每臂抽 50 条正确轨迹，输出 CSV（检测器标签 + 空白人工标签列）；填好后 --labels 计算逐类精确率 / 召回率表。
用法: python scripts/14_strategy_audit.py --key ord_n8_h4_d5 --runs runs/A_0.5_ord_n8_h4_d5_s1 runs/B_0.5_ord_n8_h4_d5_s1 [--out results/strategy_audit_<key>.csv]
      python scripts/14_strategy_audit.py --labels results/strategy_audit_<key>.csv     # 读回人工标签列 human_label
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.environ.setdefault("COT_NO_UNSLOTH", "1")
import argparse, csv, json, random
from collections import Counter

def precision_recall(rows):
    classes = sorted({r["detector_label"] for r in rows} | {r["human_label"] for r in rows if r["human_label"]})
    out = {}
    for c in classes:
        tp = sum(1 for r in rows if r["detector_label"] == c and r["human_label"] == c)
        fp = sum(1 for r in rows if r["detector_label"] == c and r["human_label"] and r["human_label"] != c)
        fn = sum(1 for r in rows if r["human_label"] == c and r["detector_label"] != c)
        out[c] = dict(precision=(tp / (tp + fp)) if tp + fp else None, recall=(tp / (tp + fn)) if tp + fn else None, tp=tp, fp=fp, fn=fn)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key"); ap.add_argument("--runs", nargs="*", default=[]); ap.add_argument("--per-arm", type=int, default=50)
    ap.add_argument("--out"); ap.add_argument("--labels"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.labels:
        rows = [r for r in csv.DictReader(open(args.labels)) if r.get("human_label")]
        pr = precision_recall(rows); print(json.dumps(pr, indent=1)); json.dump(pr, open(pathlib.Path(args.labels).with_suffix(".pr.json"), "w"), indent=1); return
    import tasks
    from cot_compress.shortcuts import strategy_class, detect_shortcuts
    from cot_compress.traj_stats import think_text
    items = {it["id"]: it for sp in tasks.ordering.load_splits(args.key, 0).values() if isinstance(sp, list) for it in sp}
    p = tasks.ordering.parse_key(args.key); rng = random.Random(args.seed); out_rows = []
    for run in args.runs:
        rows = [json.loads(l) for l in open(ROOT / run / "diagnostics_rows.jsonl")]
        rows = [r for r in rows if r.get("correct")]; rng.shuffle(rows)
        for r in rows[:args.per_arm]:
            it = items[r["id"]]; th = think_text(r["completion"])
            sc = strategy_class(th, it["prompt"], p["n"], it["meta"]["S"])
            out_rows.append(dict(run=run, id=r["id"], detector_label=sc["strategy_class"], shortcuts="+".join(detect_shortcuts(th, it["prompt"], p["n"], p["d"])),
                                 branches=sc["branches"], human_label="", think=th.replace("\n", " ⏎ ")[:4000]))
    out = ROOT / (args.out or f"results/strategy_audit_{args.key}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    print(Counter(r["detector_label"] for r in out_rows)); print("wrote", out)

if __name__ == "__main__":
    main()
