"""生成 ordering 四个划分（train 2000 / stop 500 / report 500 / direct 2000）到 data/ordering/<key>_seed<seed>.json，并打印格统计与拒格判定。
用法: python scripts/ord_build_splits.py --key ord_n8_h4_d5 [--seed 0] [--workers 11] [--train 2000 --stop 500 --report 500 --direct 2000]
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import argparse, json, os, time
from tasks import ordering as O

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    for k in ("train", "stop", "report", "direct"): ap.add_argument(f"--{k}", type=int, default=O.SPLITS[k])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    p = O.split_path(args.key, args.seed)
    if p.exists() and not args.force:
        sp = json.load(open(p)); print(f"[exists] {p}")
    else:
        t0 = time.time()
        sp = O.build_splits(args.key, args.seed, sizes=dict(train=args.train, stop=args.stop, report=args.report, direct=args.direct), workers=args.workers)
        p.parent.mkdir(parents=True, exist_ok=True); json.dump(sp, open(p, "w"))
        print(f"[built] {p} in {(time.time() - t0) / 60:.1f} min")
    rep = O.cell_stats(sp); rep["_decision"] = O.cell_decision(rep)
    print(json.dumps(rep, indent=1)); json.dump(rep, open(p.with_suffix(".report.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
