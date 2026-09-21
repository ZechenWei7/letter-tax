import json, zstandard, io, collections, statistics as st
def rd(p):
    with open(p, "rb") as f:
        r = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
        for l in io.TextIOWrapper(r, encoding="utf-8"):
            if l.strip(): yield json.loads(l)
G = collections.defaultdict(list)
for r in rd("/workspace/cot-compress/runs/A_0.5_ord_n8_h4_d5_s1/rollouts.jsonl.zst"):
    G[(r["step"], r["prompt_id"])].append(r)
sizes = collections.Counter(len(v) for v in G.values())
print("组大小分布:", dict(sizes), "  组数:", len(G))
BINS = [(1,25),(26,50),(51,75),(76,10**9)]
print(f"\n{'训练步':>10} {'组数':>5} {'全对%':>7} {'全错%':>7} {'零奖励差%':>10} {'奖励差中位':>11} {'奖励差p25':>10} {'全对组内奖励差中位':>18} {'组内L差中位':>12}")
for a, b in BINS:
    gs = [v for (s, _), v in G.items() if a <= s <= b]
    if not gs: continue
    n = len(gs)
    allc = sum(1 for v in gs if all(x["c"] for x in v))
    allw = sum(1 for v in gs if not any(x["c"] for x in v))
    spread = [max(x["r"] for x in v) - min(x["r"] for x in v) for v in gs]
    zero = sum(1 for s_ in spread if s_ < 1e-9)
    ac = [max(x["r"] for x in v) - min(x["r"] for x in v) for v in gs if all(x["c"] for x in v)]
    ls = [max(x["L"] for x in v) - min(x["L"] for x in v) for v in gs]
    lbl = f"{a}-{'' if b > 10**8 else b}"
    print(f"{lbl:>10} {n:>5} {100*allc/n:>7.1f} {100*allw/n:>7.1f} {100*zero/n:>10.1f} "
          f"{st.median(spread):>11.2f} {st.quantiles(spread, n=4)[0]:>10.2f} "
          f"{(st.median(ac) if ac else float('nan')):>18.2f} {int(st.median(ls)):>12}")
print("\n参考：奖励区间约 [0, 10]（r = 10·[c·(1 − min(λL/M, 0.9)) − …]）；Dr. GRPO + scale_rewards=none →")
print("组内优势 = r − 组均值，**组内奖励差为 0 的组梯度为 0**。全对组仍有长度带来的奖励差（这正是长度信号的来源）。")
