import json, zstandard, io, os, sys
R = "/workspace/cot-compress/runs/A_0.5_ord_n8_h4_d5_s1"
def rd(p):
    with open(p, "rb") as f:
        r = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
        for l in io.TextIOWrapper(r, encoding="utf-8"):
            if l.strip(): yield json.loads(l)
steps = [int(x) for x in sys.argv[1].split(",")]
P = {}
for s in steps:
    f = f"{R}/eval_step{s:04d}.jsonl.zst"
    if not os.path.exists(f): print(f"(step {s} 归档不存在，跳过)"); continue
    rows = list(rd(f))
    cap = [r for r in rows if r["capped"]]; nat = [r for r in rows if not r["capped"]]
    acc = lambda g: (sum(1 for r in g if r["correct"]) / len(g)) if g else float("nan")
    P[s] = dict(n=len(rows), p_cap=len(cap)/len(rows), acc_cap=acc(cap), acc_nat=acc(nat), acc=acc(rows),
                n_cap=len(cap), n_nat=len(nat))
print(f"{'step':>5} {'n':>5} {'到顶率':>8} {'自然结束 acc':>13} {'(n)':>6} {'被截断 acc':>12} {'(n)':>6} {'总 acc':>8} {'核对':>8}")
for s in steps:
    if s not in P: continue
    d = P[s]; chk = d["p_cap"]*d["acc_cap"] + (1-d["p_cap"])*d["acc_nat"]
    print(f"{s:>5} {d['n']:>5} {d['p_cap']:>8.3f} {d['acc_nat']:>13.3f} {d['n_nat']:>6} {d['acc_cap']:>12.3f} {d['n_cap']:>6} {d['acc']:>8.3f} {chk:>8.3f}")

print("\n=== 分解（对称/精确：Δ(p·a) = Δp·ā + p̄·Δa，两组求和）===")
print(f"{'区间':>12} {'Δ总acc(pp)':>11} {'到顶率下降贡献':>15} {'组内acc变化贡献':>16} {'  其中自然结束':>14} {'  其中被截断':>13}")
ks = [s for s in steps if s in P]
for i in range(1, len(ks)):
    a, b = P[ks[i-1]], P[ks[i]]
    pn0, pn1 = 1-a["p_cap"], 1-b["p_cap"]; pc0, pc1 = a["p_cap"], b["p_cap"]
    comp = (pn1-pn0)*(a["acc_nat"]+b["acc_nat"])/2 + (pc1-pc0)*(a["acc_cap"]+b["acc_cap"])/2
    rate_n = (pn0+pn1)/2*(b["acc_nat"]-a["acc_nat"]); rate_c = (pc0+pc1)/2*(b["acc_cap"]-a["acc_cap"])
    tot = b["acc"]-a["acc"]
    print(f"{str(ks[i-1])+'→'+str(ks[i]):>12} {100*tot:>11.1f} {100*comp:>15.1f} {100*(rate_n+rate_c):>16.1f} {100*rate_n:>14.1f} {100*rate_c:>13.1f}")
a, b = P[ks[0]], P[ks[-1]]
pn0, pn1 = 1-a["p_cap"], 1-b["p_cap"]; pc0, pc1 = a["p_cap"], b["p_cap"]
comp = (pn1-pn0)*(a["acc_nat"]+b["acc_nat"])/2 + (pc1-pc0)*(a["acc_cap"]+b["acc_cap"])/2
rate_n = (pn0+pn1)/2*(b["acc_nat"]-a["acc_nat"]); rate_c = (pc0+pc1)/2*(b["acc_cap"]-a["acc_cap"])
print(f"{str(ks[0])+'→'+str(ks[-1])+' 合计':>12} {100*(b['acc']-a['acc']):>11.1f} {100*comp:>15.1f} {100*(rate_n+rate_c):>16.1f} {100*rate_n:>14.1f} {100*rate_c:>13.1f}")
