import json, zstandard, io, os, sys, statistics as st
R = sys.argv[2] if len(sys.argv) > 2 else "/workspace/cot-compress/runs/A_0.5_ord_n8_h4_d5_s1"
def rd(p):
    with open(p, "rb") as f:
        r = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
        for l in io.TextIOWrapper(r, encoding="utf-8"):
            if l.strip(): yield json.loads(l)
def q(v, p): 
    v = sorted(v); 
    if not v: return float("nan")
    k = (len(v)-1)*p; lo, hi = int(k), min(int(k)+1, len(v)-1)
    return v[lo] + (v[hi]-v[lo])*(k-lo)
steps = [int(x) for x in sys.argv[1].split(",")]
print("== 全部 500 条（含被截断者，其长度钉在 cap 10240 附近）==")
print(f"{'step':>5} {'p10':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p90':>7} {'mean':>7} {'到顶率':>7}")
rows_by_step = {}
for s in steps:
    f = f"{R}/eval_step{s:04d}.jsonl.zst"
    if not os.path.exists(f): continue
    rows = list(rd(f)); rows_by_step[s] = rows
    L = [r["think_tokens"] for r in rows]
    print(f"{s:>5} {q(L,.1):>7.0f} {q(L,.25):>7.0f} {q(L,.5):>7.0f} {q(L,.75):>7.0f} {q(L,.9):>7.0f} {st.mean(L):>7.0f} {sum(1 for r in rows if r['capped'])/len(rows):>7.3f}")
print("\n== 仅自然结束者（未被 cap 截断）==")
print(f"{'step':>5} {'n':>5} {'p10':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p90':>7} {'mean':>7}")
for s, rows in rows_by_step.items():
    L = [r["think_tokens"] for r in rows if not r["capped"]]
    print(f"{s:>5} {len(L):>5} {q(L,.1):>7.0f} {q(L,.25):>7.0f} {q(L,.5):>7.0f} {q(L,.75):>7.0f} {q(L,.9):>7.0f} {st.mean(L):>7.0f}")
b = rows_by_step[steps[0]]; e = rows_by_step[steps[-1]]
print("\n== 相对首个点的降幅（全部 / 仅自然结束）==")
for p in (.1,.25,.5,.75,.9):
    a1 = q([r["think_tokens"] for r in b], p); a2 = q([r["think_tokens"] for r in e], p)
    n1 = q([r["think_tokens"] for r in b if not r["capped"]], p); n2 = q([r["think_tokens"] for r in e if not r["capped"]], p)
    print(f"  p{int(p*100):<3} {a1:6.0f} -> {a2:6.0f}  ({100*(1-a2/a1):5.1f}%)   |  自然结束 {n1:6.0f} -> {n2:6.0f}  ({100*(1-n2/n1):5.1f}%)")
