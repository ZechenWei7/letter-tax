"""生成 kk 三个划分（train 2000 / stop 500 / report 500）到 data/kk/<key>_seed<seed>.json，并打印划分报告（规范形数、S、形式分布）。
用法: python scripts/kk_build_splits.py --key kk_n10_s1 [--seed 0] [--workers 12] [--train 2000]
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import argparse, json, os, time
from tasks import kk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--train", type=int, default=kk.SPLITS["train"]); ap.add_argument("--stop", type=int, default=kk.SPLITS["stop"]); ap.add_argument("--report", type=int, default=kk.SPLITS["report"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    p = kk.split_path(args.key, args.seed)
    if p.exists() and not args.force:
        sp = json.load(open(p)); print(f"[exists] {p}")
    else:
        t0 = time.time()
        sp = kk.build_splits(args.key, args.seed, sizes=dict(train=args.train, stop=args.stop, report=args.report), workers=args.workers)
        p.parent.mkdir(parents=True, exist_ok=True); json.dump(sp, open(p, "w"))
        print(f"[built] {p} in {(time.time() - t0) / 60:.1f} min")
    rep = kk.split_report(sp)
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(p.with_suffix(".report.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
