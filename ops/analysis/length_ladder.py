#!/usr/bin/env python3
"""第二阶段第 0 步（纯 CPU、只读）：长度阶梯 A1 原生 → N1 → N2 → N3（N2s 对照），与 A1 的强制预算曲线放在一起。
主块 = 报告集（stage-1 report 划分，500 题）：A1 收敛轨迹取自 06_diagnose 的 diag_native_test（与强制预算曲线同一批题、同一 checkpoint）。
参考块 = 停止集：冻结模型（A1 step-0 eval）与 A1 收敛（step-350 eval）。
单位：token（Qwen3-4B 分词器，对 think 文本重新分词）、code point、zstd 比特。
zstd：与 E1（cot_compress.endpoints.zstd_dict_bits）同设置——110 KB 并集字典、level 19、seed 0——但按"题"对半：一半题的全部版本训练字典，
另一半题在所有版本上计算比特（E1 按片段对半、只报均值；这里需要逐题配对，故改为按题）。另报无字典 level-19 比特（全部题）。
A1 只取答对的轨迹；N 版本是求解器推导，按构造正确。配对统计在"A1 答对"的题上做。
用法：HF_HOME=~/hf COT_NO_UNSLOTH=1 .venv/bin/python ops/analysis/length_ladder.py"""
import os, sys, json, random, statistics as st, pathlib
os.environ.setdefault("COT_NO_UNSLOTH", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import zstandard as zstd
from transformers import AutoTokenizer
from tasks import ordering as O
from cot_compress import derivation as D
from cot_compress.archive import read_jsonl_zst
from cot_compress.traj_stats import think_text

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
RUN = ROOT / "cloud_pull/runs/A_0.5_ord_n8_h4_d5_s1"
diag = json.load(open(RUN / "diagnostics.json"))
splits = O.load_splits("ord_n8_h4_d5", 0)

def block(split, a1_sources):
    items = {it["id"]: it for it in splits[split]}
    # 以 prompt 对齐（eval 行里有 prompt；id 字段也有，优先用 id）
    lv = {}
    for name, path in a1_sources.items():
        rows = list(read_jsonl_zst(path)); m = {}
        for r in rows:
            it = items.get(r.get("id")) or next((x for x in items.values() if x["prompt"] == r.get("prompt")), None)
            if it is not None and r.get("correct"): m[it["id"]] = think_text(r["completion"])
        lv[name] = m
    for v in D.VERSIONS:
        lv[v] = {i: D.render(D.derive(it), it, v) for i, it in items.items()}
    return items, lv

def stats(items, lv, a1_key):
    ids_all = sorted(items); ids_c = sorted(lv[a1_key])               # A1 答对的题
    rng = random.Random(0); ids_sh = ids_c[:]; rng.shuffle(ids_sh); half = len(ids_sh) // 2
    tr, te = ids_sh[:half], ids_sh[half:]
    levels = [k for k in lv]
    zd = zstd.train_dictionary(110 * 1024, [lv[l][i].encode() for l in levels for i in tr if i in lv[l]])
    cd, cn = zstd.ZstdCompressor(level=19, dict_data=zd), zstd.ZstdCompressor(level=19)
    M = {}
    for l in levels:
        tokens = {i: ntok(lv[l][i]) for i in ids_c if i in lv[l]}
        cps = {i: len(lv[l][i]) for i in ids_c if i in lv[l]}
        bd = {i: len(cd.compress(lv[l][i].encode())) * 8 for i in te if i in lv[l]}
        bn = {i: len(cn.compress(lv[l][i].encode())) * 8 for i in ids_c if i in lv[l]}
        M[l] = dict(n=len(tokens), tokens=tokens, cps=cps, bits_dict=bd, bits_nodict=bn)
    return M, len(ids_c), len(te)

def med(d): return st.median(d.values()) if d else float("nan")
def ratio_rows(M, pairs):
    out = []
    for a, b in pairs:
        r = {}
        for u in ("tokens", "cps", "bits_dict", "bits_nodict"):
            common = [i for i in M[a][u] if i in M[b][u]]
            r[u] = dict(ratio_of_medians=med({i: M[b][u][i] for i in common}) / med({i: M[a][u][i] for i in common}),
                        median_of_ratios=st.median(M[b][u][i] / M[a][u][i] for i in common), n=len(common))
        out.append(dict(frm=a, to=b, **r))
    return out

out = {}
itemsR, lvR = block("report", {"A1_conv": RUN / "diag_native_test.jsonl.zst"})
MR, nR, teR = stats(itemsR, lvR, "A1_conv")
out["report"] = dict(n_items=len(itemsR), n_A1_correct=nR, n_zstd_heldout=teR,
                     medians={l: {u: med(MR[l][u]) for u in ("tokens", "cps", "bits_dict", "bits_nodict")} | {"n": MR[l]["n"]} for l in MR},
                     ratios=ratio_rows(MR, [("A1_conv", "N1"), ("N1", "N2"), ("N2", "N3"), ("N2", "N2s"), ("A1_conv", "N2"), ("A1_conv", "N3")]))
itemsS, lvS = block("stop", {"frozen_step0": RUN / "eval_step0000.jsonl.zst", "A1_step350": RUN / "eval_step0350.jsonl.zst"})
MS, nS, teS = stats(itemsS, lvS, "A1_step350")
out["stop"] = dict(n_items=len(itemsS), n_A1_correct=nS, n_zstd_heldout=teS,
                   medians={l: {u: med(MS[l][u]) for u in ("tokens", "cps", "bits_dict", "bits_nodict")} | {"n": MS[l]["n"]} for l in MS},
                   ratios=ratio_rows(MS, [("frozen_step0", "A1_step350"), ("A1_step350", "N1"), ("N1", "N2"), ("N2", "N3"), ("N2", "N2s")]))
# 全部题（不限 A1 答对）上的 N 版本中位
out["report"]["N_medians_all_items"] = {v: dict(tokens=st.median(ntok(lvR[v][i]) for i in itemsR), cps=st.median(len(lvR[v][i]) for i in itemsR)) for v in D.VERSIONS}
out["budget"] = dict(base_L=diag["budget_base_L"], split="report", curve=diag["budget_curve"],
                     points={b: dict(token_budget=round(float(b) * diag["budget_base_L"]), acc=diag["budget_curve"][b],
                                     mean_tokens_all=p["mean_tokens_all"]) for b, p in diag["budget_points"].items()},
                     native_acc=diag["acc"], native_L_median=diag["L_median"])
json.dump(out, open(ROOT / "ops/analysis/length_ladder.out.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "points"} if isinstance(v, dict) else v for k, v in out.items()}, indent=1)[:6000])
