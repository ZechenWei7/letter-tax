import json, zstandard, io, sys, statistics as st
sys.path.insert(0, "/workspace/cot-compress")
from cot_compress.traj_stats import think_text
def rd(p):
    with open(p, "rb") as f:
        r = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
        for l in io.TextIOWrapper(r, encoding="utf-8"):
            if l.strip(): yield json.loads(l)
WANT = {int(x) for x in sys.argv[1].split(",")}
best = {}
for r in rd("/workspace/cot-compress/runs/A_0.5_ord_n8_h4_d5_s1/rollouts.jsonl.zst"):
    if r["step"] in WANT and r.get("c"):
        t = think_text(r["text"])
        # 取该步中位长度附近的一条
        best.setdefault(r["step"], []).append((len(t), r["prompt_id"], t))
for s in sorted(best):
    xs = sorted(best[s]); m = xs[len(xs)//2]
    print(f"\n{'='*100}\nSTEP {s}  (该步正确轨迹的中位长度条; {len(xs)} 条正确)  prompt={m[1]}  chars={m[0]}")
    t = m[2]
    print(f"--- 开头 1000 字符 ---\n{t[:1000]}")
    print(f"--- 结尾 1200 字符 ---\n{t[-1200:]}")
