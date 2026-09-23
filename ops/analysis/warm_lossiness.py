#!/usr/bin/env python3
"""E2 失败归因（纯 CPU、只读）：B_warm 的 SFT 训练数据在数据层面是否已经有损 / 重复，还是训练数据正常而生成塌缩。
三列对照：A1 原始轨迹（SFT 源）｜warm 变换后的训练数据（本地确定性重建，与 pod 上 report 核对）｜E2 实际生成（eval 归档）。
用法：COT_NO_UNSLOTH=1 .venv/bin/python ops/analysis/warm_lossiness.py
"""
import os, sys, re, json, zlib, random, statistics as st, collections, pathlib
os.environ.setdefault("COT_NO_UNSLOTH", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import tasks
from cot_compress.warm import warm_text, make_sft_rows, WARM_MAP, _WORD
from cot_compress.traj_stats import think_text
from cot_compress.archive import read_jsonl_zst

A1 = ROOT / "cloud_pull/runs/A_0.5_ord_n8_h4_d5_s1/rollouts.jsonl.zst"
E2 = ROOT / "cloud_pull/runs/evalonly_Bwarm_ord_n8_h4_d5_s1_sft/eval_step0000.jsonl.zst"
KEY = "ord_n8_h4_d5"

# ---------- 1. 重建 SFT 数据（逐字照搬 scripts/11_sft_warm.py::collect_rows，last_steps=50） ----------
rows0 = list(read_jsonl_zst(A1))
steps = sorted({r["step"] for r in rows0 if r.get("step") is not None}); keep = set(steps[-50:])
items = {it["id"]: it for sp in tasks.task_for_key(KEY).load_splits(KEY, 0).values() if isinstance(sp, list) for it in sp}
src = []
for r in rows0:
    if r.get("step") not in keep or r.get("c") != 1.0 or r.get("violations"): continue
    it = items.get(r.get("prompt_id"))
    if it is None: continue
    src.append(dict(id=r["prompt_id"], prompt=it["prompt"], think=think_text(r["text"]), gold=r["gold"], correct=True))
sft = make_sft_rows(src)
ratio = st.mean(r["warm_len_chars"] / max(1, r["src_len_chars"]) for r in sft)
print(f"重建：n_src={len(src)} n_sft={len(sft)} len_ratio_chars={ratio:.14f}  (pod report: 1435 / 0.33758948738716477)")
gen = [think_text(r["completion"]) for r in read_jsonl_zst(E2)]

# ---------- 指标 ----------
# 推理标记：只用映射符号与约束语法不冲突的那部分（before→<, after→>, or→|, false→-, true→+, knight/knave/lies/truthful→数字 与约束式同形，排除）
AMBIG = {"before", "after", "or", "true", "false", "knight", "knave", "lies", "truthful"}
MARK_WORDS = sorted(w for w in WARM_MAP if w not in AMBIG)
MARK_SYMS = sorted({WARM_MAP[w] for w in MARK_WORDS})
INF_WORDS = ["if", "then", "so", "therefore", "thus", "hence", "assume", "suppose", "case", "contradiction", "contradicts", "impossible", "cycle"]
INF_SYMS = sorted({WARM_MAP[w] for w in INF_WORDS})
def rx_words(ws): return re.compile(r"\b(?:" + "|".join(map(re.escape, ws)) + r")\b", re.I)
def rx_syms(ss): return re.compile("|".join(map(re.escape, sorted(ss, key=len, reverse=True))))
M_EN, M_W = rx_words(MARK_WORDS), rx_syms(MARK_SYMS)
I_EN, I_W = rx_words(INF_WORDS), rx_syms(INF_SYMS)
ATOM = re.compile(r"\(?\s*\d\s*[<>]\s*\d\s*\)?(?:\s*\|\s*\(?\s*\d\s*[<>]\s*\d\s*\)?)?")
SCAF = re.compile(r"[\s\-\*\.,:;()\d|<>`'\"]")
PARA = re.compile(r"\n\s*\n")

def is_list(p, mark):
    if mark.search(p): return False
    if len(ATOM.findall(p)) < 3: return False
    ns = re.sub(r"\s", "", p)
    if not ns: return False
    resid = SCAF.sub("", ATOM.sub("", p))
    return len(resid) <= 0.2 * len(ns)

def list_stats(t, mark):
    ps = [p for p in PARA.split(t) if p.strip()]
    li = [i for i, p in enumerate(ps) if is_list(p, mark)]
    pairs = list(zip(li, li[1:]))
    nomark = sum(1 for a, b in pairs if not any(mark.search(ps[k]) for k in range(a + 1, b)))
    ch = sum(len(ps[i]) for i in li)
    return dict(n_para=len(ps), n_list=len(li), pairs=len(pairs), pairs_nomark=nomark, list_chars=ch, chars=max(1, sum(len(p) for p in ps)))

def rep4(t):
    w = t.split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0
def blk20(t, k=20):
    if len(t) < 5 * k: return 0.0
    c = collections.Counter(t[i:i + k] for i in range(0, len(t) - k, k)); return c.most_common(1)[0][1] / (len(t) // k)
def zr(t): return len(zlib.compress(t.encode(), 9)) / max(1, len(t.encode()))

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
    ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
except Exception as e:
    tok = None; ntok = None; print("（本地无 Qwen3 分词器，只报字符：", repr(e)[:80], "）")

def summarize(name, texts, en):
    mark, inf = (M_EN, I_EN) if en else (M_W, I_W)
    L = [len(t) for t in texts]
    ls = [list_stats(t, mark) for t in texts]; li = [list_stats(t, inf) for t in texts]
    d = dict(name=name, n=len(texts), chars_mean=st.mean(L), chars_median=st.median(L),
             tokens_mean=(st.mean(ntok(t) for t in texts[:300]) if ntok else None),
             rep4_median=st.median(rep4(t) for t in texts), rep4_mean=st.mean(rep4(t) for t in texts),
             blk20_median=st.median(blk20(t) for t in texts), blk20_gt05=sum(1 for t in texts if blk20(t) > 0.5) / len(texts),
             zlib_median=st.median(zr(t) for t in texts),
             list_paras_mean=st.mean(s["n_list"] for s in ls), list_char_share=sum(s["list_chars"] for s in ls) / sum(s["chars"] for s in ls),
             adj_pairs=sum(s["pairs"] for s in ls), adj_nomark=sum(s["pairs_nomark"] for s in ls),
             adj_nomark_frac=(sum(s["pairs_nomark"] for s in ls) / max(1, sum(s["pairs"] for s in ls))),
             adj_noinf_frac=(sum(s["pairs_nomark"] for s in li) / max(1, sum(s["pairs"] for s in li))),
             marker_per_kchar=st.mean(1000 * len(mark.findall(t)) / max(1, len(t)) for t in texts),
             inf_per_kchar=st.mean(1000 * len(inf.findall(t)) / max(1, len(t)) for t in texts))
    return d

S = [summarize("A1 原文（SFT 源）", [r["think"] for r in src], True),
     summarize("warm 训练数据", [r["think_warm"] for r in sft], False),
     summarize("E2 实际生成", gen, False)]

# ---------- 段落级对齐：A1 的推理段落（非清单段）经 warm 后变成什么 ----------
seg = collections.Counter(); segc = collections.Counter()
for r in src:
    for p in PARA.split(r["think"]):
        if not p.strip() or is_list(p, M_EN): continue
        w = warm_text(p); has_d = bool(re.search(r"\d", w)); has_m = bool(M_W.search(w)); has_i = bool(I_W.search(w))
        k = ("含数字+推理标记" if has_d and has_m else "只剩数字（无标记）" if has_d else "只剩标记（无数字）" if has_m else "只剩标点碎屑 / 空")
        seg[k] += 1; segc[k] += len(p)

# ---------- 3. 被删掉的高频词 ----------
cnt = collections.Counter(); df = collections.Counter(); mapped = 0; total = 0
for r in src:
    ws = [m.group(0).lower().replace("’", "'") for m in _WORD.finditer(r["think"])]
    total += len(ws); seen = set()
    for w in ws:
        if w in WARM_MAP: mapped += 1; continue
        cnt[w] += 1; seen.add(w)
    for w in seen: df[w] += 1
deleted = sum(cnt.values())

out = dict(rebuild=dict(n_src=len(src), n_sft=len(sft), len_ratio_chars=ratio), metrics=S,
           reasoning_paragraph_fate=dict(counts=dict(seg), chars=dict(segc)),
           words=dict(total=total, mapped=mapped, deleted=deleted, top=[(w, c, df[w]) for w, c in cnt.most_common(60)]),
           markers=dict(all_words=MARK_WORDS, all_syms=MARK_SYMS, inf_words=INF_WORDS, inf_syms=INF_SYMS))
(ROOT / "ops/analysis/warm_lossiness.out.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

# ---------- 2. 三条样本 ----------
od = ROOT / "ops/warm_samples"; od.mkdir(exist_ok=True)
idx = sorted(random.Random(0).sample(range(len(sft)), 3))
for k, i in enumerate(idx, 1):
    a, w = src[i], sft[i]
    hdr = f"# {a['id']}   gold={a['gold']}   A1 原文 {len(a['think'])} 字符 → warm {len(w['think_warm'])} 字符（{len(w['think_warm'])/len(a['think']):.3f}）\n# 抽样：seed=0，从 {len(sft)} 条 SFT 行里随机取 3 条（索引 {idx}）\n"
    (od / f"s{k}_A1.txt").write_text(hdr + "# ---- A1 原文（SFT 源；题面见文末）----\n" + a["think"] + "\n\n# ---- 题面 ----\n" + a["prompt"] + "\n")
    (od / f"s{k}_warm.txt").write_text(hdr + "# ---- warm 变换后（= B_warm-SFT 的训练目标 think 段）----\n" + w["think_warm"] + "\n\n# ---- 题面（SFT 时原样给出）----\n" + w["prompt"] + "\n")
print(json.dumps(dict(samples=idx), ensure_ascii=False))
