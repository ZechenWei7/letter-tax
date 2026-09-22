#!/usr/bin/env python3
"""看板用聚合（描述量，不进判定）：从一个 run 的 rollouts.jsonl.zst 按每 BIN 步一个点算
   hit_exact（= c 均值）、hit_hamming2（tasks.ordering.hamming ≤ 2，与 eval.jsonl 的 hit_hamming2 同口径）、
   hard 违规率、零奖励差（无梯度）组占比、全错组占比、组内奖励差中位、L 均值。
   写到 runs/<run>/agg10.json；pod 上循环跑，watcher 拉回，ops/dashboard.py 画。
用法：python ops/analysis/b_agg.py runs/B_0.5_ord_n8_h4_d5_s1 [--bin 10] [--loop 600]"""
import json, zstandard, io, collections, statistics as st, sys, os, time, argparse, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
os.environ.setdefault("COT_NO_UNSLOTH", "1")
from tasks.ordering import hamming
def rd(p):
    with open(p, "rb") as f:
        r = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
        for l in io.TextIOWrapper(r, encoding="utf-8"):
            if l.strip(): yield json.loads(l)
def agg(run, binw):
    p = ROOT / run / "rollouts.jsonl.zst"
    if not p.exists(): return None
    G = collections.defaultdict(list)
    for r in rd(p): G[(r["step"], r["prompt_id"])].append(r)
    if not G: return None
    bins = collections.defaultdict(list)
    for (s, _), g in G.items(): bins[(s - 1) // binw].append(g)        # step 1..10 → bin 0
    out = []
    for b in sorted(bins):
        gs = bins[b]; rows = [x for g in gs for x in g]
        hd = [hamming(x.get("pred") or "", x["gold"]) for x in rows]
        sp = [max(x["r"] for x in g) - min(x["r"] for x in g) for g in gs]
        out.append(dict(step_lo=b * binw + 1, step_hi=(b + 1) * binw, step=(b + 1) * binw, n_groups=len(gs), n=len(rows),
                        hit_exact=sum(1 for x in rows if x.get("c")) / len(rows),
                        hit_hamming2=sum(1 for h in hd if h is not None and h <= 2) / len(rows),
                        hard_rate=sum(1 for x in rows if x.get("hard")) / len(rows),
                        zero_grad_frac=sum(1 for s_ in sp if s_ < 1e-9) / len(gs),
                        all_wrong_frac=sum(1 for g in gs if not any(x["c"] for x in g)) / len(gs),
                        reward_spread_median=st.median(sp), L_mean=st.mean(x["L"] for x in rows)))
    return dict(run=run, bin=binw, generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), bins=out)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--bin", type=int, default=10); ap.add_argument("--loop", type=int, default=0)
    a = ap.parse_args()
    while True:
        d = agg(a.run, a.bin)
        if d:
            (ROOT / a.run / "agg10.json").write_text(json.dumps(d, indent=1)); print(time.strftime("%H:%M:%S"), "wrote", a.run, "bins:", len(d["bins"]), flush=True)
        if not a.loop: break
        time.sleep(a.loop)
