"""白名单审计（v4 第 2 条）：导出 letterfree 白名单全部 token 与解码串，自动标记：
  (a) 带圈 / 括号 / 方框 / 负圈拉丁字母等 So 类字母替身（Unicode 名含 CIRCLED/PARENTHESIZED/SQUARED/NEGATIVE ... LATIN 或 DIGIT 不算）；
  (b) 区域指示符（U+1F1E6–1F1FF）；
  (c) NFKC 归一化后含字母的 token（全角、数学字母等）；
  (d) 能被 leet 反解码成英文单词的 token（4→a 3→e 1→i/l 0→o 5→s 7→t @→a $→s |→l !→i 8→b 6→g 2→z 9→g）。
标记的 id 写 results/whitelist_extra_banned.json（vocab_mask 自动读取并从白名单剔除）；报告 results/whitelist_audit.md。
用法: python scripts/07_audit_whitelist.py [--tokenizer unsloth/Qwen3-1.7B] [--words words.txt]
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import argparse, json, itertools, unicodedata, re
from transformers import AutoTokenizer
from cot_compress.vocab_mask import think_allowed_ids, lift_id

COMMON_WORDS = set("""the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would
there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than
then now look only come its over think also back after use two how our work first well way even new want because any these give day most us is are was were been has had
did does done said says made get got going gone let lets yes ok okay sum add mul times plus minus equal equals total result answer step next last first second third
digit number carry product divide half double zero one two three four five six seven eight nine ten hundred thousand million cup cups position swap flip up down left
right top bottom start end begin final check again wait let see try note thus hence so therefore since because thing things each every both same different big small
long short high low true false yes no set list count""".split())
LEET = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s", "|": "l", "!": "i", "8": "b", "6": "g", "2": "z", "9": "g"}
LEET_ALT = {"1": "l"}

def leet_decodings(s: str):
    s = s.strip()
    if not (2 <= len(s) <= 8):
        return []
    opts = []
    for ch in s:
        alts = set()
        if ch in LEET: alts.add(LEET[ch])
        if ch in LEET_ALT: alts.add(LEET_ALT[ch])
        if ch.isalpha(): alts.add(ch.lower())
        if not alts:
            return []
        opts.append(sorted(alts))
    return ["".join(p) for p in itertools.product(*opts)][:64]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="unsloth/Qwen3-1.7B"); ap.add_argument("--words", help="额外英文词表文件（每行一个）")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    words = set(COMMON_WORDS)
    if args.words:
        words |= {w.strip().lower() for w in open(args.words) if w.strip()}
    allowed = think_allowed_ids(tok, "letterfree", extra_banned=set())   # 审计前的原始白名单
    lid = lift_id(tok)
    flagged = {}
    rows = []
    for tid in sorted(allowed):
        s = tok.decode([tid]); reasons = []
        if tid == lid:
            rows.append((tid, s, [])); continue
        for ch in s:
            name = unicodedata.name(ch, "")
            cat = unicodedata.category(ch)
            if cat == "So" and "LATIN" in name and any(k in name for k in ("CIRCLED", "PARENTHESIZED", "SQUARED", "NEGATIVE")):
                reasons.append(f"circled_latin:{name}")
            if 0x1F1E6 <= ord(ch) <= 0x1F1FF:
                reasons.append("regional_indicator")
        nf = unicodedata.normalize("NFKC", s)
        if any(unicodedata.category(c)[0] == "L" for c in nf):
            reasons.append(f"nfkc_letters:{nf.strip()!r}")
        hits = [d for d in leet_decodings(s) if d in words]
        if hits:
            reasons.append(f"leet:{hits[0]}")
        if reasons:
            flagged[tid] = (s, reasons)
        rows.append((tid, s, reasons))
    out = ROOT / "results/whitelist_extra_banned.json"
    json.dump(dict(tokenizer=args.tokenizer, ids=sorted(flagged), n=len(flagged),
                   reasons={str(t): r for t, (_, r) in flagged.items()}), open(out, "w"), indent=1)
    md = ROOT / "results/whitelist_audit.md"
    with open(md, "w") as f:
        f.write(f"# Whitelist audit — {args.tokenizer}\n\n原始白名单 {len(allowed)} 个 token；标记并追加禁集 {len(flagged)} 个（写入 {out.name}）。\n\n")
        f.write("## 标记明细\n\n| id | token | 原因 |\n|---|---|---|\n")
        for tid, (s, r) in sorted(flagged.items()):
            f.write(f"| {tid} | `{s!r}` | {'; '.join(r)} |\n")
        f.write("\n## 审计后白名单全量（id, repr）\n\n")
        for tid, s, r in rows:
            if tid not in flagged:
                f.write(f"{tid}\t{s!r}\n")
    print(f"allowed(before)={len(allowed)} flagged={len(flagged)} -> {out}, report {md}")
    for tid, (s, r) in list(sorted(flagged.items()))[:30]:
        print(f"  {tid} {s!r} {r}")

if __name__ == "__main__":
    main()
