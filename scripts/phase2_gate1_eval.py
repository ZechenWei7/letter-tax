"""第二阶段闸门 1 评估入口（新评估集 500 题）。不在 OSF 注册的 stage-1 协议之内。定义见 cot_compress/gate1.py 文件头。
每个 run（一种写法 × 一个 seed）做：
  native_<mask>     原生：think 段由模型生成（cap = gate 配置 eval.cap，预算强制同 stage-1），N3 在 letterfree 与 none 两种屏蔽下各一次，N2 / N2s 只 none
  direct            直接作答（<think></think> 预填；evaluate.run_eval mode="direct"）
  transplant_gold   移植：题 i 的 think 段 = 题 π(i) 的金标推导（同写法），只生成答案（evaluate.prefilled_answer_eval，同 stage-1 准入第 5 条机制）
  transplant_own    副：题 π(i) 上本 run 自己生成的 think 段（主屏蔽条件下的 native）
  tamper_control / tamper   篡改检验（预填前半，续写；主屏蔽条件）
模型调用只用 stage-1 已在 pod 上跑通的函数：evaluate.run_eval、evaluate.prefilled_answer_eval、rollout.generate_texts_vllm；
载入方式同 scripts/06_diagnose.py --backend vllm（adapter 合并进权重 → /workspace/tmp_merged/<run> → vLLM + 引擎级 SpanLogitsProcessor）。
--stub oracle|mixed：不载模型，用桩生成器代替 vLLM（只用于 CPU 上测整条打分流水线，产物写到 --out-root，结果无意义）。
输出：<out-root>/gate1_<N>_s<seed>/gate1_eval.json（各条件汇总）+ gate1_rows.jsonl.zst（逐题记录，含完整生成文本）。
用法：python scripts/phase2_gate1_eval.py --notation N3 --seed 1 [--gate-config configs/phase2_gate1.yaml] [--gpu-util 0.5] [--stub oracle --out-root <dir>]"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.environ.setdefault("COT_NO_UNSLOTH", "1")
import argparse, json, random, statistics, time, hashlib
import yaml
import tasks
from tasks import ordering as O
from cot_compress.config import load_config
from cot_compress import derivation as D, gate1 as G

EVAL_SAMPLING_SEED = 0          # 与 stage-1 的 eval 相同（seed=0）；训练 seed 只影响 adapter


def summarize_native(recs):
    n = len(recs); f = lambda k: sum(bool(r[k]) for r in recs) / n
    return dict(n=n, acc=f("correct"), exact=f("exact"), parsed=f("parsed"), sound=f("sound"), loop=f("loop"), capped=f("capped"),
                letter_frac_mean=statistics.mean(r["letter_frac"] for r in recs), think_tokens_median=statistics.median(r["think_tokens"] for r in recs))


def install_stub(tok, items, notation, mode, cap, reserve):
    """桩生成器：按 prompt 找到题，按 prompt 结尾判断要生成的部分。oracle = 永远给金标；mixed = 按题哈希分成 金标 / 循环 / 答错。"""
    import torch, cot_compress.rollout as R, cot_compress.evaluate as E
    from cot_compress.generation import GenOut
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None; torch.cuda.max_memory_allocated = lambda *a, **k: 0
    by_hard = {it["prompt"].split("\n")[1]: it for it in items}
    gold = {it["id"]: D.derive(it) for it in items}
    def kind(it):
        if mode == "oracle": return "gold"
        h = int(hashlib.md5(it["id"].encode()).hexdigest(), 16) % 10
        return "gold" if h < 6 else ("loop" if h < 8 else "wrong")
    def wrong(it): p = it["answer"].split(); return " ".join(p[1:] + p[:1])
    def fake(llm, tok_, prompts, *, max_new_tokens, sampling, reserve=24, mask_mode="none", banned_ids=None, force=True, seed=0):
        outs = []
        for pr in prompts:
            it = next(v for k, v in by_hard.items() if k in pr); kd = kind(it); txt = D.render(gold[it["id"]], it, notation)
            if pr.endswith("<answer>"):                                   # 只生成答案：直接作答 / 移植（读 think 段里的 order 行）
                body = pr.rsplit("<think>", 1)[-1].split("</think>", 1)[0] if "<think>" in pr else ""
                om = [l for l in body.split("\n") if l.strip().split(" ")[:1] == [D.MAPS[notation]["order"]]]
                if om: ans = " ".join(om[-1].split()[1:])              # 移植：读 think 段里的 order 行（模拟"从轨迹读答案"）
                elif not body.strip(): ans = wrong(it)                  # 直接作答：think 为空
                else: ans = it["answer"]
                t = ans + "</answer>"; forced = False
            else:
                prefix = pr.rsplit("<think>\n", 1)[-1]
                lines = txt.split("\n"); done = [l for l in prefix.split("\n") if l.strip()]
                rest = "\n".join(lines[len(done):])
                if kd == "loop":
                    t = "\n".join([lines[0]] * 60) + "\n" + R.CLOSE_TEXT + wrong(it) + "</answer>"; forced = True
                else:
                    t = rest + "\n</think>\n\n<answer>" + (it["answer"] if kd == "gold" else wrong(it)) + "</answer>"; forced = False
            ids = tok(t, add_special_tokens=False)["input_ids"]
            outs.append(GenOut(ids=ids, text=t, finished=True, think_tokens=0, has_think_close="</think>" in t, forced=forced))
        return outs
    R.generate_texts_vllm = fake
    E.VLLM["llm"] = object()
    return fake


def load_vllm(cfg, adapter_dir, run_name, gpu_util):
    import torch, cot_compress.evaluate as E
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from vllm import LLM
    from cot_compress.rollout import engine_kwargs
    m = cfg["model"]; tok = AutoTokenizer.from_pretrained(m["name"])
    merged = pathlib.Path("/workspace/tmp_merged") / run_name
    if not (merged / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(m["name"], dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
        merged.mkdir(parents=True, exist_ok=True); model.save_pretrained(str(merged), safe_serialization=True); tok.save_pretrained(str(merged))
        del model; torch.cuda.empty_cache()
    E.VLLM["llm"] = LLM(model=str(merged), dtype="bfloat16", gpu_memory_utilization=gpu_util, max_model_len=int(m["max_seq_length"]),
                        max_num_seqs=128, **engine_kwargs())
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notation", required=True, choices=["N2", "N2s", "N3"]); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gate-config", default="configs/phase2_gate1.yaml"); ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--stub", choices=["oracle", "mixed"]); ap.add_argument("--out-root", default=str(ROOT / "runs")); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    g = yaml.safe_load(open(ROOT / a.gate_config)); cfg = load_config(g["base_config"], None); key = g["key"]; task = tasks.task_for_key(key)
    N = a.notation; run_name = f"gate1_{N}_s{a.seed}"; out_dir = pathlib.Path(a.out_root) / run_name; out_dir.mkdir(parents=True, exist_ok=True)
    items = json.load(open(ROOT / g["eval"]["set"]))["p2eval"][: a.limit]
    cap = int(g["eval"]["cap"]); reserve = int(cfg["train"].get("force_reserve", 40))
    S = g["eval"]["sampling"]
    gen = dict(cfg["generation"], max_new_tokens_think=cap, temperature=S["temperature"], top_p=S["top_p"], top_k=S["top_k"],
               batch_size=int(cfg["generation"].get("eval_batch_size", 64)))
    masks = list(g["eval"]["mask"][N]) if isinstance(g["eval"]["mask"][N], list) else [g["eval"]["mask"][N]]
    primary = masks[0]
    T0 = time.time(); timing = {}                                            # 分段墙钟（链脚本按它推算全量费用）
    if a.stub:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], local_files_only=True); install_stub(tok, items, N, a.stub, cap, reserve)
    else:
        tok = load_vllm(cfg, ROOT / "runs" / run_name / "final", run_name, a.gpu_util)
    import cot_compress.evaluate as E, cot_compress.rollout as R
    from cot_compress.span import HOOK
    from cot_compress.lengths import letter_fraction
    from cot_compress.archive import append_jsonl_zst
    timing["load_sec"] = round(time.time() - T0, 1); t_ = time.time()
    gold = {it["id"]: D.derive(it) for it in items}
    gold_txt = {it["id"]: D.render(gold[it["id"]], it, N) for it in items}
    rows_out, summ = [], dict(run=run_name, notation=N, seed=a.seed, n_items=len(items), cap=cap, masks=masks, primary_mask=primary,
                              eval_sampling_seed=EVAL_SAMPLING_SEED, stub=a.stub, started=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
    native_primary = None
    for mask in masks:                                                       # 原生
        HOOK.configure(tok, reserve=reserve, mask_mode=mask)
        rows = E.run_eval(None, tok, key, items, "think", gen, seed=EVAL_SAMPLING_SEED)
        recs = []
        for it, r in zip(items, rows):
            th = G.think_of(r["completion"]); sd = G.score_derivation(th, it, N, gold[it["id"]])
            recs.append(dict(cond=f"native_{mask}", id=it["id"], seed=a.seed, correct=bool(r["correct"]), pred=r["pred"], capped=bool(r["capped"]),
                             loop=G.loop_flag(th), letter_frac=letter_fraction(th), think_tokens=r["think_tokens"], completion=r["completion"], **sd))
        summ[f"native_{mask}"] = summarize_native(recs); rows_out += recs
        timing[f"native_{mask}_sec"] = round(time.time() - t_, 1); t_ = time.time()
        if mask == primary: native_primary = recs
    HOOK.configure(tok, reserve=reserve, mask_mode="none")                   # 直接作答
    rows = E.run_eval(None, tok, key, items, "direct", gen, seed=EVAL_SAMPLING_SEED)
    rows_out += [dict(cond="direct", id=it["id"], seed=a.seed, correct=bool(r["correct"]), pred=r["pred"], completion=r["completion"]) for it, r in zip(items, rows)]
    summ["direct"] = dict(n=len(rows), acc=sum(r["correct"] for r in rows) / len(rows))
    timing["direct_sec"] = round(time.time() - t_, 1); t_ = time.time()
    pi = G.transplant_pairing(len(items), seed=0)                            # 移植
    for cond, src in (("transplant_gold", lambda j: gold_txt[items[j]["id"]]), ("transplant_own", lambda j: G.think_of(native_primary[j]["completion"]))):
        ids = [tok(src(pi[i]), add_special_tokens=False)["input_ids"] for i in range(len(items))]
        rows = E.prefilled_answer_eval(None, tok, key, items, ids, gen, seed=EVAL_SAMPLING_SEED, max_new_tokens=24)
        rows_out += [dict(cond=cond, id=it["id"], seed=a.seed, donor=items[pi[i]]["id"], correct=bool(r["correct"]), pred=r["pred"],
                          donor_answer_copied=(r["pred"] == items[pi[i]]["answer"])) for i, (it, r) in enumerate(zip(items, rows))]
        summ[cond] = dict(n=len(rows), acc=sum(r["correct"] for r in rows) / len(rows),
                          donor_answer_copied=sum(r["pred"] == items[pi[i]]["answer"] for i, r in enumerate(rows)) / len(rows))
    timing["transplant_sec"] = round(time.time() - t_, 1); t_ = time.time()
    plans = {it["id"]: G.tamper_plan(gold[it["id"]], it) for it in items}    # 篡改
    T = [(it, plans[it["id"]]) for it in items if plans[it["id"]]]
    HOOK.configure(tok, reserve=reserve, mask_mode=primary)
    pre = {}
    for it, p in T:
        c, t = G.tamper_texts(gold[it["id"]], it, N, p); base = E._prompt_text(tok, task, it, "think", gen)[0]
        pre[it["id"]] = dict(control=(base + c + "\n", c), tamper=(base + t + "\n", t))
    plen = max(len(tok(v[1], add_special_tokens=False)["input_ids"]) for d_ in pre.values() for v in d_.values()) if pre else 0
    max_new = cap - plen - 1
    samp = dict(do_sample=True, temperature=S["temperature"], top_p=S["top_p"], top_k=S["top_k"])
    tam = {}
    for arm in ("control", "tamper"):
        prompts = [pre[it["id"]][arm][0] for it, _ in T]
        outs = R.generate_texts_vllm(E.VLLM["llm"], tok, prompts, max_new_tokens=max_new, sampling=samp, reserve=reserve,
                                     mask_mode=primary, force=True, seed=EVAL_SAMPLING_SEED)
        recs = []
        for (it, p), o in zip(T, outs):
            prefix = pre[it["id"]][arm][1]
            r = E._row(tok, task, it, "think", prefix + "\n" + o.text, o, cap=max_new)
            fol = G.continuation_follow(G.think_of(o.text), N, p["edge"], r["pred"])
            recs.append(dict(cond=("tamper_control" if arm == "control" else "tamper"), id=it["id"], seed=a.seed, correct=bool(r["correct"]), pred=r["pred"], edge=list(p["edge"]),
                             prefix_steps=p["h"], completion=o.text, **fol))
        rows_out += recs; tam[arm] = recs
    if T:
        f = lambda rs, k: sum(bool(r[k]) for r in rs) / len(rs)
        ment = [r for r in tam["tamper"] if r["uses_flip"] or r["uses_orig"]]
        ment_c = [r for r in tam["control"] if r["uses_flip"] or r["uses_orig"]]
        summ["tamper"] = dict(n_items=len(T), coverage=len(T) / len(items), prefix_token_max=plen, continuation_max_new=max_new,
                              acc_control=f(tam["control"], "correct"), acc_tamper=f(tam["tamper"], "correct"),
                              delta=f(tam["tamper"], "correct") - f(tam["control"], "correct"),
                              n_mentioning=len(ment), follows_flip_among_mentioning=(sum(r["follows_flip"] for r in ment) / len(ment) if ment else None),
                              uses_flip=f(tam["tamper"], "uses_flip"), uses_orig=f(tam["tamper"], "uses_orig"),
                              control_uses_flip=f(tam["control"], "uses_flip"), control_uses_orig=f(tam["control"], "uses_orig"),
                              control_follows_flip_among_mentioning=(sum(r["follows_flip"] for r in ment_c) / len(ment_c) if ment_c else None),
                              answer_order_flip=sum(r["answer_order"] == "flip" for r in tam["tamper"]) / len(T),
                              control_answer_order_flip=sum(r["answer_order"] == "flip" for r in tam["control"]) / len(T))
    timing["tamper_sec"] = round(time.time() - t_, 1); timing["total_sec"] = round(time.time() - T0, 1)
    summ["timing"] = timing
    summ["max_think_tokens"] = max((r.get("think_tokens") or 0) for r in rows_out if r["cond"].startswith("native_"))
    summ["finished"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    json.dump(summ, open(out_dir / "gate1_eval.json", "w"), indent=1)
    p = out_dir / "gate1_rows.jsonl.zst"
    if p.exists(): p.unlink()
    append_jsonl_zst(p, rows_out)
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
