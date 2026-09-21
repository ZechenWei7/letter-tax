import json, zstandard, io, sys, re, os, statistics as st
sys.path.insert(0, "/workspace/cot-compress")
from cot_compress.lengths import letter_fraction
from cot_compress.traj_stats import think_text
R = "/workspace/cot-compress/runs/A_0.5_ord_n8_h4_d5_s1"
OUT = "/tmp/cot_samples"; os.makedirs(OUT, exist_ok=True)

def rd(p):
    with open(p, "rb") as f:
        r = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
        for l in io.TextIOWrapper(r, encoding="utf-8"):
            if l.strip(): yield json.loads(l)

A = {r["id"]: r for r in rd(f"{R}/eval_step0000.jsonl.zst")}
B = {r["id"]: r for r in rd(f"{R}/eval_step0100.jsonl.zst")}
both = [i for i in A if i in B and A[i]["correct"] and B[i]["correct"]]
print(f"step0 正确 {sum(1 for r in A.values() if r['correct'])}/{len(A)}; "
      f"step100 正确 {sum(1 for r in B.values() if r['correct'])}/{len(B)}; 两者都对 {len(both)}")
red = sorted(((1 - B[i]["think_tokens"] / A[i]["think_tokens"]), i) for i in both)
print("两者都对的题上，step100 相对 step0 的 token 缩减：",
      f"中位 {100*st.median([r for r, _ in red]):.1f}%  p25 {100*red[len(red)//4][0]:.1f}%  p75 {100*red[3*len(red)//4][0]:.1f}%")
picks = [("q1_p25", red[len(red)//4]), ("q2_p50", red[len(red)//2]), ("q3_p75", red[3*len(red)//4])]

def stats(t):
    n = lambda p: len(re.findall(p, t, re.I))
    return dict(chars=len(t), letter=round(100*letter_fraction(t), 1), sent=n(r"[.!?](?:\s|$)"),
                wait=n(r"\bwait\b"), check=n(r"\b(?:check|verif|confirm|recheck)"),
                arrow=n(r"→|->|<"), orders=n(r"\d(?:\s*(?:[<,→]|->)\s*|\s+)\d(?:(?:\s*(?:[<,→]|->)\s*|\s+)\d){6,}"))

rows = []
for tag, (r, i) in picks:
    for step, src in (("000", A), ("100", B)):
        rec = src[i]; th = think_text(rec["completion"]); s = stats(th)
        path = f"{OUT}/{tag}_step{step}.txt"
        with open(path, "w") as f:
            f.write(f"# {rec['id']}   checkpoint = step {int(step)}\n"
                    f"# think_tokens={rec['think_tokens']}  chars={s['chars']}  字母占比={s['letter']}%  "
                    f"句子={s['sent']}  wait={s['wait']}  check/verify={s['check']}  箭头+<={s['arrow']}  完整顺序={s['orders']}\n"
                    f"# capped={rec['capped']}  forced={rec['forced']}  correct={rec['correct']}\n"
                    f"# gold={rec['gold']}   pred={rec['pred']}\n"
                    f"{'='*100}\n[题面]\n{rec['prompt']}\n{'='*100}\n[think 段]\n{th}\n")
        rows.append((tag, step, rec["id"], rec["think_tokens"], s))
    print(f"{tag}: {i}  缩减 {100*r:.1f}%  {A[i]['think_tokens']} → {B[i]['think_tokens']} token")

with open(f"{OUT}/README.md", "w") as f:
    f.write("# A1 同题对照：step-0 vs step-100（eval 输出，stopping-eval 500 题，T=0.6，每题 1 条）\n\n"
            "**挑法（中性规则，非挑选性）**：两个 checkpoint 都答对的题共 %d 道；按 step-100 相对 step-0 的 token 缩减排序，\n"
            "取 **p25 / p50 / p75** 三道，分别为 q1 / q2 / q3。缩减中位 %.1f%%。\n\n"
            "| 文件 | 题 | checkpoint | think token | 字符 | 字母占比 | 句子 | wait | check/verify | 箭头+`<` | 完整顺序 |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
            % (len(both), 100*st.median([r for r, _ in red])))
    for tag, step, i, tk, s in rows:
        f.write(f"| `{tag}_step{step}.txt` | {i.split('/')[-1]} | {int(step)} | {tk} | {s['chars']} | {s['letter']}% | "
                f"{s['sent']} | {s['wait']} | {s['check']} | {s['arrow']} | {s['orders']} |\n")
    f.write("\n并排读法：`diff -y --width=200 q2_p50_step000.txt q2_p50_step100.txt | less` 或在编辑器里左右开两栏。\n")
print("\n写入", OUT)
