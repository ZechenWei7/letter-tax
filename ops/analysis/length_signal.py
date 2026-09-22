import json, zstandard, io, collections, statistics as st, sys
R = sys.argv[2] if len(sys.argv) > 2 else "/workspace/cot-compress/runs/A_0.5_ord_n8_h4_d5_s1"
def rd(p):
    with open(p, "rb") as f:
        r = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
        for l in io.TextIOWrapper(r, encoding="utf-8"):
            if l.strip(): yield json.loads(l)
G = collections.defaultdict(list)
for r in rd(f"{R}/rollouts.jsonl.zst"): G[(r["step"], r["prompt_id"])].append(r)
BINS = [(int(a), int(b)) for a, b in (x.split("-") for x in sys.argv[1].split(","))]
print(f"{'训练步':>10} {'组数':>5} {'组内答对数中位':>14} {'全对组%':>8} {'全错组%':>8} {'答对数分布(0/1-4/5-8/9-12/13-15/16)':>36}")
D = {}
for a, b in BINS:
    gs = [v for (s, _), v in G.items() if a <= s <= b]
    if not gs: continue
    nc = [sum(1 for x in v if x["c"]) for v in gs]
    h = [sum(1 for k in nc if lo <= k <= hi) for lo, hi in [(0,0),(1,4),(5,8),(9,12),(13,15),(16,16)]]
    D[(a,b)] = gs
    print(f"{f'{a}-{b}':>10} {len(gs):>5} {st.median(nc):>14.1f} {100*h[5]/len(gs):>8.1f} {100*h[0]/len(gs):>8.1f} {str(h):>36}")
print(f"\n{'训练步':>10} {'组内总奖励差中位':>16} {'答对者之间的奖励差中位':>22} {'长度信号占比中位':>16} {'该占比 p25':>11}")
for (a, b), gs in D.items():
    tot, cor, frac = [], [], []
    for v in gs:
        t = max(x["r"] for x in v) - min(x["r"] for x in v)
        cs = [x["r"] for x in v if x["c"]]
        c = (max(cs) - min(cs)) if len(cs) >= 2 else 0.0
        tot.append(t); cor.append(c)
        if t > 1e-9: frac.append(c / t)
    print(f"{f'{a}-{b}':>10} {st.median(tot):>16.2f} {st.median(cor):>22.2f} {st.median(frac):>16.3f} {st.quantiles(frac,n=4)[0]:>11.3f}")
print("\n口径：组内所有 rollout 的 r 极差 = 总奖励差（Dr.GRPO 优势 = r − 组均值）；")
print("     只在答对者之间取极差 = 纯由长度造成的那部分（c=1,v=0 时 r = 10(1−λL/M)，单调于 L）；占比 = 后者 / 前者。")
