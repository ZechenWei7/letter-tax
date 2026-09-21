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

ORD = re.compile(r"\d(?:\s*(?:[<,→]|->)\s*|\s+)\d(?:(?:\s*(?:[<,→]|->)\s*|\s+)\d){6,}")
PATS = [("wait", r"\bwait\b"), ("but/however", r"\b(?:but|however)\b"),
        ("check/verify", r"\b(?:check|verif|confirm|recheck)"),
        ("so/thus", r"\b(?:so|thus|therefore|hence)\b"),
        ("if/suppose/case", r"\b(?:if|suppose|assume|case)\b"),
        ("sentences", r"[.!?](?:\s|$)"), ("arrows+<", r"→|->|<")]
print(f"{'step':>5} {'corr':>5} {'chars':>6} | 每千字符频次: " + " ".join(f"{p[0]:>16}" for p in PATS))
for s in sorted(buck):
    rows = [r for r in buck[s] if r.get("c")]
    if not rows: continue
    th = [think_text(r["text"]) for r in rows]
    out = []
    for name, pat in PATS:
        out.append(f"{st.median([1000*len(re.findall(pat, t, re.I))/max(len(t),1) for t in th]):>16.2f}")
    print(f"{s:>5} {len(rows):>5} {int(st.median([len(t) for t in th])):>6} | " + " ".join(out))

print("\n=== 复核段落：写出第一个完整顺序之后还剩多少文本 ===")
print(f"{'step':>5} {'首个完整顺序位置%':>18} {'其后文本占比%':>14} {'完整顺序个数':>12} {'其后 wait 次数':>14}")
for s in sorted(buck):
    rows = [r for r in buck[s] if r.get("c")]
    if not rows: continue
    th = [think_text(r["text"]) for r in rows]
    pos, tail, cnt, wtail = [], [], [], []
    for t in th:
        m = ORD.search(t)
        if not m: continue
        pos.append(100*m.start()/len(t)); tail.append(100*(len(t)-m.end())/len(t))
        cnt.append(len(ORD.findall(t))); wtail.append(len(re.findall(r"\bwait\b", t[m.end():], re.I)))
    if pos:
        print(f"{s:>5} {st.median(pos):>18.1f} {st.median(tail):>14.1f} {st.median(cnt):>12.1f} {st.median(wtail):>14.1f}  (n={len(pos)}/{len(th)})")
