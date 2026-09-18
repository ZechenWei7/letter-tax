"""内容诊断 v7，对 A、A″、B、Bwarm、C_rand 跑（kk：预算用 stopping-eval 定、报告用 reporting-eval；两者不混）：
  端点：训练后策略的 native (L, acc)、direct 准确率、强制预算曲线 {0.1,0.2,…,1.0}×自身收敛长度（每点：acc、外化准确率 acc−direct、正确轨迹 token 数均值）
  kk 附加：策略类分布、模板命中率、字母占比、命中率（exact / Hamming≤2）、每题 L（按题匹配的次要长度用）
  诊断：2 轨迹移植 i→j；3 自身轨迹 unigram 重采样；4 自身轨迹填充替换（白名单 unigram）；5 前缀评分（cups：0.25/0.5/0.75 截断 + 强制，
        与各操作前缀后的真实状态比对 → 峰值位置）；6 反事实状态编辑（B 记法解析器尝试，失败 → not applicable）
  描述量：禁集占比、唯一 token 率、组内 n-gram 多样性、强制收尾模板复制率、提示词复制率、zstd 比、0.6B-Base PPL、
        长度三单位（token / code point / 自适应 5-gram 比特）
用法: python scripts/06_diagnose.py --run runs/<run> [--ckpt final] --config configs/cloud_4b.yaml --key cups_n6_k12 [--n 1000] [--base]
写 runs/<run>/diagnostics.json、diagnostics_rows.jsonl。H2 判定在 analyze.py（需要 C_rand 与多 seed）。
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import cot_compress  # noqa
import argparse, json, random, re, statistics, time
from collections import Counter
import torch
import tasks
from cot_compress.config import load_config
from cot_compress.span import HOOK, CLOSE_TEXT
from cot_compress.evaluate import (run_eval, forced_answer_eval, prefilled_answer_eval, summarize, masked_fraction_rows, ref_ppl_rows)
from cot_compress.vocab_mask import banned_ids, think_allowed_ids, lift_id
from cot_compress.traj_stats import TrajStats, think_ids, think_text
from cot_compress.lengths import code_points, adaptive_bits, zstd_ratio
from cot_compress.parsing import ANSWER_OPEN
from cot_compress.rewards import violations
from cot_compress.archive import append_jsonl_zst

def _save(run_dir, name, rows):
    """诊断中间产物全量留档：runs/<run>/diag_<name>.jsonl.zst（覆盖写：先删旧文件）"""
    p = run_dir / f"diag_{name}.jsonl.zst"
    if p.exists(): p.unlink()
    append_jsonl_zst(p, rows)

FRACS = tuple(round(0.1 * i, 1) for i in range(1, 11))     # v7：0.1 … 1.0 × 自身收敛长度

def load_for_diag(cfg, adapter_dir):
    m = cfg["model"]
    if m.get("use_unsloth", True):
        from unsloth import FastLanguageModel
        name = str(adapter_dir) if adapter_dir else m["name"]
        model, tok = FastLanguageModel.from_pretrained(model_name=name, max_seq_length=int(m["max_seq_length"]),
                                                       load_in_4bit=bool(m["load_in_4bit"]), fast_inference=False, dtype=m.get("dtype"))
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(m["name"])
        model = AutoModelForCausalLM.from_pretrained(m["name"], dtype=torch.bfloat16, device_map="cuda")
        if adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
        model.eval()
    tok.padding_side = "left"
    return model, tok

def ngrams(ids, n): return [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]

def descriptives(tok, rows, items_by_id, lf_banned, lid, vocab_size):
    th = [think_ids(tok, r["completion"], lid) for r in rows]
    uniq = [len(set(x)) / len(x) for x in th if x]
    grp = Counter(); tot = 0
    for x in th:
        g = ngrams(x, 4); grp.update(g); tot += len(g)
    prompt_copy = []
    for r, x in zip(rows, th):
        p_ids = tok(items_by_id[r["id"]]["prompt"], add_special_tokens=False)["input_ids"]; ps = set(ngrams(p_ids, 4))
        g = ngrams(x, 4)
        if g: prompt_copy.append(sum(t in ps for t in g) / len(g))
    close_ids = tok(CLOSE_TEXT, add_special_tokens=False)["input_ids"]
    tmpl = [1.0 if r.get("forced") else 0.0 for r in rows]
    texts = [think_text(r["completion"]) for r in rows]
    z = [zstd_ratio(t)[0] for t in texts]; z = [v for v in z if v is not None]
    return dict(banned_frac=masked_fraction_rows(tok, rows, set(lf_banned)),
                unique_token_rate=statistics.mean(uniq) if uniq else None,
                group_4gram_diversity=(len(grp) / tot) if tot else None,
                forced_close_template_rate=statistics.mean(tmpl),
                prompt_copy_rate_4gram=statistics.mean(prompt_copy) if prompt_copy else None,
                zstd_ratio=statistics.mean(z) if z else None, zstd_name=zstd_ratio("x" * 100)[1],
                len_tokens=statistics.mean(len(x) for x in th), len_code_points=statistics.mean(code_points(t) for t in texts),
                adaptive_bits=adaptive_bits(th, vocab_size), viol_rate=statistics.mean(1.0 if violations(r["completion"], tasks.task_for_key(r["key"]), tok) else 0.0 for r in rows))

run_dir_g = [None]
def prefix_scoring(model, tok, key, rows, items_by_id, gen, seed):
    """cups：0.25/0.5/0.75 截断 + 强制作答；答案与每个操作前缀后的真实状态（查询位置）比对，报告命中的前缀步数分布与峰值。"""
    out = {}
    for f in (0.25, 0.5, 0.75):
        fr = forced_answer_eval(model, tok, rows, None, gen, seed=seed, batch_size=gen["batch_size"], cutoff_frac=f)
        _save(run_dir_g[0], f"prefix{f}", fr)
        hits = Counter(); n = 0
        for r in fr:
            it = items_by_id[r["id"]]; meta = it["meta"]; k = meta["k"]
            n += 1
            if r["pred"] is None: continue
            for step in range(k + 1):                       # 完整终态：预测状态与第 step 步之后的真实全状态比对
                st = [(i + 1, True) for i in range(meta["n"])] if step == 0 else [(c, bool(u)) for c, u in meta["states"][step - 1]]
                if r["pred"] == tasks.cups.state_str(st):
                    hits[round(step / k, 2)] += 1; break
        out[str(f)] = dict(acc=summarize(fr)["acc"], matched_prefix_frac_hist={str(k_): v / n for k_, v in sorted(hits.items())},
                           peak=(max(hits, key=hits.get) if hits else None))
    return out

_STATE_RE = re.compile(r"(\d+)\s*[:=]\s*(\d+)\s*([↑↓+\-]|up|down)?", re.I)
def try_parse_notation(text: str, n: int):
    """尝试把 B 的记法解析成 {位置: (杯号, 朝向)}；覆盖 ≥ 80% 位置才算可解析。"""
    st = {}
    for m in _STATE_RE.finditer(text):
        pos, cup = int(m.group(1)), int(m.group(2))
        if 1 <= pos <= n and 1 <= cup <= n: st[pos] = (cup, (m.group(3) or "").lower() not in ("↓", "-", "down"))
    return st if len(st) >= 0.8 * n else None

def counterfactual_edit(model, tok, key, rows, items_by_id, gen, seed):
    """诊断 6：解析记法 → 交换两个位置的杯号 → 强制作答，看答案是否跟着变。不可解析 → not applicable。"""
    edits, thinks, its = [], [], []
    for r in rows:
        it = items_by_id[r["id"]]; n = it["meta"]["n"]
        t = think_text(r["completion"]); st = try_parse_notation(t, n)
        if st is None or it["meta"]["query_pos"] not in st: continue
        q = it["meta"]["query_pos"]; other = next((p for p in st if p != q), None)
        if other is None: continue
        cq, oq = st[q]; co, oo = st[other]
        # 把最后一次出现的 "q: cq" 改成 "q: co"
        m = list(_STATE_RE.finditer(t))
        last = [x for x in m if int(x.group(1)) == q]
        if not last: continue
        x = last[-1]; t2 = t[:x.start()] + f"{q}: {co}" + (x.group(3) or "") + t[x.end():]
        thinks.append(tok(t2, add_special_tokens=False)["input_ids"]); its.append(it); edits.append((cq, co, oq))
    if len(thinks) < max(20, 0.2 * len(rows)):
        return dict(applicable=False, n_parsed=len(thinks))
    fr = prefilled_answer_eval(model, tok, key, its, thinks, gen, seed=seed, batch_size=gen["batch_size"], tag="cf")
    _save(run_dir_g[0], "counterfactual", fr)
    def cup_at(pred, pos):
        toks = (pred or "").split(); return int(re.match(r"\d+", toks[pos - 1]).group()) if len(toks) >= pos and re.match(r"\d+", toks[pos - 1]) else None
    follow = [1.0 if cup_at(r["pred"], it["meta"]["query_pos"]) == co else 0.0 for r, it, (cq, co, oq) in zip(fr, its, edits)]
    return dict(applicable=True, n_parsed=len(thinks), follow_edit_rate=statistics.mean(follow))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="final"); ap.add_argument("--config", default="configs/local_1p7b.yaml")
    ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--base", action="store_true")
    ap.add_argument("--key"); ap.add_argument("--select-seed", type=int, default=0); ap.add_argument("--test-seed", type=int, default=7)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="vllm：把 adapter 合并进权重、存到 /workspace/tmp_merged/<run>，用 vLLM 生成（n=1000 必需）")
    ap.add_argument("--gpu-util", type=float, default=0.5)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set); run_dir = ROOT / args.run; run_dir.mkdir(parents=True, exist_ok=True)
    key = args.key or cfg["task"]["key"]; task = tasks.task_for_key(key)
    adapter = None if args.base else (run_dir / args.ckpt)
    if adapter and not adapter.exists():
        c = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])); assert c, f"no checkpoint in {run_dir}"; adapter = c[-1]
    cap = int(cfg["train"]["max_completion_length"]); reserve = int(cfg["train"].get("force_reserve", 24))
    gen = dict(cfg["generation"], batch_size=int(cfg["generation"].get("eval_batch_size", 4)), max_new_tokens_think=cap)
    model, tok = load_for_diag(cfg, adapter)
    from cot_compress.memory import tune_allocator; print(tune_allocator())
    if args.backend == "vllm":
        import cot_compress.evaluate as _ev
        from cot_compress.rollout import engine_kwargs
        from vllm import LLM
        merged_dir = pathlib.Path("/workspace/tmp_merged") / (run_dir.name + ("_base" if args.base else ""))
        if not (merged_dir / "config.json").exists():
            merged_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(merged_dir), safe_serialization=True); tok.save_pretrained(str(merged_dir))
            print(f"[vllm] merged weights saved to {merged_dir}", flush=True)
        _ev.VLLM["llm"] = LLM(model=str(merged_dir), dtype="bfloat16", gpu_memory_utilization=args.gpu_util,
                              max_model_len=int(cfg["model"]["max_seq_length"]), max_num_seqs=128, **engine_kwargs())
    lid = lift_id(tok); V = int(model.get_input_embeddings().weight.shape[0])
    lf_banned = banned_ids(tok, "letterfree", vocab_size=V); lf_allowed = think_allowed_ids(tok, "letterfree", vocab_size=V)
    arm = run_dir.name.split("_")[0]
    if arm == "evalonly": arm = run_dir.name.split("_")[1]                                   # evalonly_Bwarm_<tag>
    if arm == "dryrun": arm = run_dir.name.split("_")[1]
    mask_mode = {"A": "none", "A2": "letterfree_plus_single", "B": "letterfree", "Bwarm": "letterfree", "Crand": "none"}.get(arm, "none")
    banned = banned_ids(tok, mask_mode, vocab_size=V) if mask_mode != "none" else None
    HOOK.configure(tok, reserve=reserve, banned_ids=banned).install(model)
    rng = random.Random(args.select_seed); t0 = time.time(); run_dir_g[0] = run_dir
    D = dict(run=args.run, ckpt=(adapter.name if adapter else "base"), key=key, arm=arm, mask=mask_mode, chance=task.chance(key))
    if key.startswith(("kk_", "ord_")):
        sel = tasks.make_eval_set(key, args.n, split="stop"); test = tasks.make_eval_set(key, args.n, split="report")   # 停止集定预算，报告集报告
    else:
        sel = tasks.make_eval_set(key, args.n, seed=args.select_seed); test = tasks.make_eval_set(key, args.n, seed=args.test_seed)
    by_id = {it["id"]: it for it in sel + test}
    # ---- 端点（测试集）----
    rows = run_eval(model, tok, key, test, "think", gen, seed=args.test_seed); s = summarize(rows)
    Ls = [r["think_tokens"] for r in rows]; Lc = float(statistics.median(Ls))
    D.update(n=len(rows), acc=s["acc"], acc_clean=s["acc_clean"], viol_rate=s["viol_rate"], L_mean=round(statistics.mean(Ls), 1), L_median=Lc,
             capped_rate=s["capped_rate"], forced_rate=s["forced_rate"], fmt_err=s["format_err"])
    D["direct_acc"] = summarize(run_eval(model, tok, key, test, "direct", gen, seed=args.test_seed))["acc"]          # 诊断 1
    # ---- 强制预算曲线 {f × 自身收敛长度}：选择集定预算，测试集报告 ----
    sel_rows = run_eval(model, tok, key, sel, "think", gen, seed=args.select_seed)
    L_sel = float(statistics.median(r["think_tokens"] for r in sel_rows))
    D["budget_curve"] = {}; D["budget_points"] = {}
    for f in FRACS:
        fr = forced_answer_eval(model, tok, rows, int(f * L_sel), gen, seed=args.test_seed, batch_size=gen["batch_size"])
        fs = summarize(fr); corr_tok = [min(r["think_tokens"], int(f * L_sel)) for r in fr if r["correct"]]; all_tok = [min(r["think_tokens"], int(f * L_sel)) for r in fr]
        for r in fr: r["think_tokens"] = min(r["think_tokens"], int(f * L_sel))
        D["budget_curve"][str(f)] = fs["acc"]
        D["budget_points"][str(f)] = dict(acc=fs["acc"], externalized=fs["acc"] - D["direct_acc"], mean_tokens_correct=(statistics.mean(corr_tok) if corr_tok else None),
                                          mean_tokens_all=statistics.mean(all_tok), n_correct=len(corr_tok), lenient_acc=fs["lenient_acc"])
        _save(run_dir, f"budget{f}", fr)
    _save(run_dir, "native_test", rows); _save(run_dir, "native_select", sel_rows)
    D["budget_base_L"] = L_sel
    # ---- 诊断 2–4（测试集；填充 / 重采样的统计来自自身轨迹）----
    st = TrajStats(tok, rows, lid)
    th2 = [st.sample_transplant(it["id"], len(st.by_item[it["id"]][0]) if st.by_item.get(it["id"]) else None, rng) for it in test]
    r2 = prefilled_answer_eval(model, tok, key, test, th2, gen, seed=args.test_seed, batch_size=gen["batch_size"], tag="transplant"); D["diag2_transplant_acc"] = summarize(r2)["acc"]; _save(run_dir, "transplant", r2)
    th3 = [st.sample_unigram(len(st.by_item[it["id"]][0]) if st.by_item.get(it["id"]) else int(Lc), rng) for it in test]
    r3 = prefilled_answer_eval(model, tok, key, test, th3, gen, seed=args.test_seed, batch_size=gen["batch_size"], tag="unigram"); D["diag3_unigram_resample_acc"] = summarize(r3)["acc"]; _save(run_dir, "unigram", r3)
    support = [t for t in st.support if t in lf_allowed] or sorted(lf_allowed - {lid}); w = [st.unigram[t] for t in support] if support and support[0] in st.unigram else None
    th4 = [rng.choices(support, weights=w, k=len(st.by_item[it["id"]][0]) if st.by_item.get(it["id"]) else int(Lc)) for it in test]
    r4 = prefilled_answer_eval(model, tok, key, test, th4, gen, seed=args.test_seed, batch_size=gen["batch_size"], tag="filler"); D["diag4_filler_acc"] = summarize(r4)["acc"]; _save(run_dir, "filler", r4)
    if key.startswith("cups"):
        D["diag5_prefix"] = prefix_scoring(model, tok, key, rows, by_id, gen, args.test_seed)
        D["diag6_counterfactual"] = counterfactual_edit(model, tok, key, rows, by_id, gen, args.test_seed)
    # ---- 描述量 ----
    D["descriptives"] = descriptives(tok, rows, by_id, lf_banned, lid, V)
    if key.startswith("kk_"):
        from cot_compress.strategy import annotate, class_distribution
        from cot_compress.templates import hit_rate
        from cot_compress.lengths import letter_fraction
        n_p = task.parse_key(key)["n"]
        ann = [annotate(think_text(r["completion"]), by_id[r["id"]]["prompt"], n_p, tok) for r in rows]
        for r, a in zip(rows, ann): r["strategy"] = a
        D["descriptives"]["strategy_class"] = class_distribution(ann)
        D["descriptives"]["copy_rate_mean"] = statistics.mean(a["copy_rate"] for a in ann if a["copy_rate"] is not None)
        D["descriptives"]["template_hits"] = hit_rate([think_text(r["completion"]) for r in rows], [by_id[r["id"]]["prompt"] for r in rows], n_p)
        D["descriptives"]["letter_frac"] = statistics.mean(letter_fraction(think_text(r["completion"])) for r in rows)
        hd = [task.hamming(r["pred"] or "", r["gold"]) for r in rows]
        D["hit_exact"] = s["acc"]; D["hit_hamming2"] = sum(1 for h in hd if h is not None and h <= 2) / len(rows)
        D["externalized_acc"] = s["acc"] - D["direct_acc"]
    if key.startswith("ord_"):                                            # v8：捷径命中、策略类分布、复制率、字母占比、命中率、下限中位
        from cot_compress.shortcuts import strategy_class, detect_shortcuts, class_distribution, shortcut_rate
        from cot_compress.lengths import letter_fraction
        pk = task.parse_key(key)
        ann = []
        for r in rows:
            it = by_id[r["id"]]; th = think_text(r["completion"]); a = strategy_class(th, it["prompt"], pk["n"], it["meta"]["S"], tok); r["strategy"] = a; r["shortcut_hits"] = detect_shortcuts(th, it["prompt"], pk["n"], pk["d"]); ann.append(a)
        corr_ann = [a for a, r in zip(ann, rows) if r["correct"]]
        D["descriptives"]["strategy_class"] = class_distribution(ann); D["descriptives"]["strategy_class_correct"] = class_distribution(corr_ann)
        D["descriptives"]["copy_rate_mean"] = statistics.mean(a["copy_rate"] for a in ann if a["copy_rate"] is not None)
        D["descriptives"]["shortcut_hits"] = shortcut_rate([think_text(r["completion"]) for r in rows], [by_id[r["id"]]["prompt"] for r in rows], pk["n"], pk["d"])
        D["descriptives"]["letter_frac"] = statistics.mean(letter_fraction(think_text(r["completion"])) for r in rows)
        hd = [task.hamming(r["pred"] or "", r["gold"]) for r in rows]
        D["hit_exact"] = s["acc"]; D["hit_hamming2"] = sum(1 for h in hd if h is not None and h <= 2) / len(rows); D["lenient_acc"] = s["lenient_acc"]
        D["externalized_acc"] = s["acc"] - D["direct_acc"]
        D["decision_lb_tokens_median"] = statistics.median(by_id[r["id"]]["meta"]["decision_lb_tokens"]["2.5"] for r in rows)
        D["kahn_lb_tokens_median"] = statistics.median(by_id[r["id"]]["meta"]["kahn_lb_tokens"]["2.5"] for r in rows)
    if key.startswith("cups"):
        from cot_compress.style import style_summary
        D["descriptives"]["rewrite_style"] = style_summary([think_text(r["completion"]) for r in rows], tasks.cups.parse_key(key)["n"])
    try:
        D["ref_ppl"] = ref_ppl_rows(rows[:50])
    except Exception as e:
        D["ref_ppl"] = None; D["ref_ppl_error"] = repr(e)[:200]
    D["sec"] = round(time.time() - t0)
    json.dump(D, open(run_dir / "diagnostics.json", "w"), indent=1)
    with open(run_dir / "diagnostics_rows.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(ROOT / "results/overnight_log.md", "a") as f:
        f.write(f"\n- diagnose {args.run}/{D['ckpt']} ({key}, n={D['n']}): acc={D['acc']:.3f} L_med={Lc} direct={D['direct_acc']:.3f} "
                f"transplant={D['diag2_transplant_acc']:.3f} unigram={D['diag3_unigram_resample_acc']:.3f} filler={D['diag4_filler_acc']:.3f} "
                f"budget={D['budget_curve']} prompt_copy={D['descriptives']['prompt_copy_rate_4gram']} ({D['sec']} s)\n")
    print(json.dumps(D, indent=1))

if __name__ == "__main__":
    main()
