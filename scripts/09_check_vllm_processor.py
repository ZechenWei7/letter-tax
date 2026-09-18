"""云端验收第 4 步：vLLM engine-level SpanLogitsProcessor + logp 一致性。
  1) 用 rollout.VLLMSpanProcessor（臂 B，letter-free）对 vllm.LLM 生成 n 条，cap 小（默认 96）→ 必须出现强制收尾 "</think>\\n\\n<answer>"、
     think 段 0 个禁集 token；臂 A（无 mask）同样检查强制收尾。
  2) logp 一致性：对同一批完成，用 HF 模型（同权重 bf16）在同一 id 级 mask 下重算 think 位置的 logp，与 vLLM 返回的 logprobs 比较，
     同时与"未屏蔽（raw）"的 HF logp 比较 → 判断 vLLM 返回的是 processed 还是 raw logprobs。
用法: python scripts/09_check_vllm_processor.py --model Qwen/Qwen3-4B [--n 4] [--cap 96] [--skip-hf]
退路：失败时先 `export VLLM_USE_V1=0` 再跑；仍失败 → 训练改 HF generate 路径（configs: generation.hook=generate, train.use_vllm=false）。
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os
os.environ.setdefault("COT_NO_UNSLOTH", "1")
import cot_compress  # noqa
import argparse, json, statistics, time
import torch
import tasks
from cot_compress.generation import chat_prompt
from cot_compress.evaluate import THINK_PREFILL
from cot_compress.rollout import VLLMSpanProcessor
from cot_compress.vocab_mask import banned_ids, think_allowed_ids, lift_id
from cot_compress.masked_logps import masked_selective_log_softmax, think_position_masks
from cot_compress.span import CLOSE_TEXT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B"); ap.add_argument("--n", type=int, default=4); ap.add_argument("--cap", type=int, default=96)
    ap.add_argument("--key", default="cups_n6_k12"); ap.add_argument("--gpu-util", type=float, default=0.45); ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--num-scheduler-steps", type=int, default=None, help="V0 引擎默认多步调度不支持 logits processors；设 1 关闭")
    args = ap.parse_args()
    import vllm
    from vllm import LLM, SamplingParams
    print("vllm", vllm.__version__, "VLLM_USE_V1=", os.environ.get("VLLM_USE_V1"))
    from cot_compress.rollout import engine_kwargs, EXTRA_KEY
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_util, max_model_len=4096, enforce_eager=False, **engine_kwargs())
    tok = llm.get_tokenizer()
    V = llm.llm_engine.model_config.get_vocab_size() if hasattr(llm, "llm_engine") else len(tok)
    task = tasks.task_for_key(args.key); items = tasks.make_eval_set(args.key, args.n, seed=0)
    prompts = [chat_prompt(tok, task.SYSTEM_PROMPT, it["prompt"], think=True, prefill=THINK_PREFILL) for it in items]
    lid = lift_id(tok); close_ids = tok(CLOSE_TEXT, add_special_tokens=False)["input_ids"]
    report = dict(vllm=vllm.__version__, vocab=V, results={})
    for arm, mode in (("B", "letterfree"), ("A", "none")):
        banned = banned_ids(tok, mode, vocab_size=V) if mode != "none" else None
        proc = VLLMSpanProcessor(tok, cap=args.cap, reserve=8, banned_ids=None)      # 只用来取 force_start / close_ids
        try:
            sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=args.cap, logprobs=0, seed=0,
                                extra_args={EXTRA_KEY: dict(cap=args.cap, reserve=8, mode=mode)})
        except Exception as e:
            print(f"[{arm}] SamplingParams(logits_processors=...) REJECTED: {e!r}"); report["results"][arm] = dict(ok=False, stage="SamplingParams", error=repr(e)[:300]); continue
        try:
            t0 = time.time(); outs = llm.generate(prompts, sp, use_tqdm=False); dt = time.time() - t0
        except Exception as e:
            print(f"[{arm}] generate with engine-level SpanLogitsProcessor FAILED: {e!r}"); report["results"][arm] = dict(ok=False, stage="generate", error=repr(e)[:300]); continue
        bset = set(banned or []); rows = []
        for o in outs:
            c = o.outputs[0]; ids = list(c.token_ids); text = tok.decode(ids, skip_special_tokens=False)
            think, forced = think_position_masks(ids, lid, proc.force_start, close_ids)
            think_ids = [t for t, m in zip(ids, think) if m and t != lid]
            rows.append(dict(n=len(ids), forced=any(forced), close_present=CLOSE_TEXT in text, banned_in_think=sum(t in bset for t in think_ids),
                             think_len=len(think_ids), text=text[:200], ids=ids, prompt_ids=list(o.prompt_token_ids),
                             lps=[lp[t].logprob for t, lp in zip(ids, c.logprobs)] if c.logprobs else None))
        ok = all(r["close_present"] for r in rows) and all(r["banned_in_think"] == 0 for r in rows)
        print(f"[{arm}] ok={ok} {dt:.1f}s close_present={[r['close_present'] for r in rows]} forced={[r['forced'] for r in rows]} "
              f"banned_in_think={[r['banned_in_think'] for r in rows]} think_len={[r['think_len'] for r in rows]}")
        print(f"[{arm}] sample: {rows[0]['text']!r}")
        report["results"][arm] = dict(ok=ok, sec=round(dt, 1), rows=[{k: v for k, v in r.items() if k not in ("ids", "prompt_ids", "lps")} for r in rows])
        report["results"][arm]["_rows_full"] = rows
    # ---- logp 一致性（臂 B）----
    rb = report["results"].get("B", {})
    if rb.get("ok") and not args.skip_hf and rb["_rows_full"][0]["lps"] is not None:
        del llm; import gc; gc.collect(); torch.cuda.empty_cache()
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
        Vm = model.get_input_embeddings().weight.shape[0]
        allowed = think_allowed_ids(tok, "letterfree", vocab_size=Vm); am = torch.zeros(Vm, dtype=torch.bool, device="cuda"); am[torch.tensor(sorted(allowed), device="cuda")] = True
        d_masked, d_raw = [], []
        for r in rb["_rows_full"]:
            ids = torch.tensor([r["prompt_ids"] + r["ids"]], device="cuda"); T = len(r["ids"])
            with torch.no_grad():
                logits = model(input_ids=ids).logits[:, :-1, :][:, -T:, :].float()
            cids = torch.tensor([r["ids"]], device="cuda")
            th, fo = think_position_masks(r["ids"], lid, args.cap - 8 - len(close_ids), close_ids)
            thm = torch.tensor([th], device="cuda")
            lp_m = masked_selective_log_softmax(logits, cids, am, thm)[0]; lp_r = masked_selective_log_softmax(logits, cids, None, thm)[0]
            v = torch.tensor(r["lps"], device="cuda")
            sel = torch.tensor([t and not f for t, f in zip(th, fo)], device="cuda")      # think 位置、非强制段
            if sel.any():
                d_masked += (v[sel] - lp_m[sel]).abs().tolist(); d_raw += (v[sel] - lp_r[sel]).abs().tolist()
        cons = dict(n_tokens=len(d_masked), vs_masked_mean=statistics.mean(d_masked), vs_masked_max=max(d_masked),
                    vs_raw_mean=statistics.mean(d_raw), vs_raw_max=max(d_raw))
        cons["vllm_returns"] = "processed(masked) logprobs" if cons["vs_masked_mean"] < cons["vs_raw_mean"] else "RAW (unmasked) logprobs"
        print("[logp]", json.dumps(cons)); report["logp_consistency"] = cons
    for a in report["results"].values():
        a.pop("_rows_full", None)
    out = ROOT / "results/vllm_processor_check.json"; out.parent.mkdir(exist_ok=True); json.dump(report, open(out, "w"), indent=1)
    print("wrote", out)
    sys.exit(0 if all(a.get("ok") for a in report["results"].values()) else 2)

if __name__ == "__main__":
    main()
