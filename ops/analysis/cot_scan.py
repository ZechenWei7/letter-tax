import json, zstandard, io, collections, statistics as st, re, sys
sys.path.insert(0, "/workspace/cot-compress")
from cot_compress.lengths import letter_fraction
from cot_compress.traj_stats import think_text

def rd(p):
    with open(p, "rb") as f:
        r = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
        for l in io.TextIOWrapper(r, encoding="utf-8"):
            if l.strip(): yield json.loads(l)

WANT = {int(x) for x in sys.argv[1].split(",")}
buck = collections.defaultdict(list)
for r in rd("/workspace/cot-compress/runs/A_0.5_ord_n8_h4_d5_s1/rollouts.jsonl.zst"):
    if r["step"] in WANT: buck[r["step"]].append(r)
print("每步条数:", {k: len(v) for k, v in sorted(buck.items())})

PATS = [("wait", r"\bwait\b"), ("but/however", r"\b(?:but|however)\b"),
        ("check/verify", r"\b(?:check|verif|confirm|recheck|re-check)"),
        ("let's/now", r"\b(?:let['’]s|let us|now)\b"),
        ("so/thus", r"\b(?:so|thus|therefore|hence)\b"),
        ("if/suppose", r"\b(?:if|suppose|assume|case)\b"),
        ("句子数", r"[.!?](?:\s|$)"), ("箭头/<", r"→|->|<"),
        ("完整顺序(>=6数)", r"\d(?:\s*(?:[<,→]|->)\s*|\s+)\d(?:(?:\s*(?:[<,→]|->)\s*|\s+)\d){4,}")]
cols = ["step", "n", "corr", "L_med", "chars", "letter%"] + [p[0] for p in PATS]
print(" | ".join(f"{c:>13}" if i > 5 else f"{c:>8}" for i, c in enumerate(cols)))
for s in sorted(buck):
    rows = [r for r in buck[s] if r.get("c")]
    if not rows:
        print(f"{s:>8} | {len(buck[s]):>8} | 0 correct"); continue
    th = [think_text(r["text"]) for r in rows]
    med = lambda vals: st.median(vals)
    vals = [f"{s:>8}", f"{len(buck[s]):>8}", f"{len(rows):>8}",
            f"{int(med([r['L'] for r in rows])):>8}", f"{int(med([len(t) for t in th])):>8}",
            f"{100*med([letter_fraction(t) for t in th]):>8.1f}"]
    for name, pat in PATS:
        vals.append(f"{med([len(re.findall(pat, t, re.I)) for t in th]):>13.1f}")
    print(" | ".join(vals))
