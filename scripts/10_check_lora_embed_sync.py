"""v7 硬依赖验证：LoRA 挂在 embed_tokens / lm_head 上时，TRL 0.25.1 colocate 能否把权重同步进 vLLM 0.10.2 引擎。
不构造 GRPOTrainer，而是**逐行镜像** TRL 0.25.1 `_move_model_to_vllm` 的 PEFT 分支：
    merge_adapter() → named_parameters() → 去 "base_model.model." / ".base_layer" → 跳过含 adapter 前缀（"lora_"）与 "original_module" 的名字
    → 去 "modules_to_save.default." → llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)]) → unmerge_adapter()
然后比较 HF（merge 后）与 vLLM 对同一批 prompt 的逐 token logprob（prompt_logprobs）。
四个变体：
  q_only        LoRA 只挂 q_proj（对照，预期同步 OK）
  +embed        再挂 embed_tokens（lora.Embedding）
  +embed+lmhead 再挂 lm_head——Qwen3-4B tie_word_embeddings=true：vLLM 的 Qwen3ForCausalLM.load_weights 会跳过 lm_head.weight（tied），
                预期 lm_head 的 delta **同步不进去**；且 PEFT 在共享张量上 merge 两个 adapter 会互相污染
判定：max |Δlogp| < 0.05（bf16 跨引擎噪声量级；未同步时 delta 会是 O(1)）。
v7 定稿（2026-09-18）：只挂 embed_tokens，不挂 lm_head，不做 untied 副本；训练侧门控读 variants["q+embed"].ok（cot_compress/lora.py）。
lm_head_tied 变体保留为阴性演示（预期 fail）。
用法（pod 空闲时）: python scripts/10_check_lora_embed_sync.py --model Qwen/Qwen3-4B --out results/lora_embed_sync_check.json
"""
import sys, pathlib, os
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault("COT_NO_UNSLOTH", "1")
import argparse, json, gc, statistics, time
import torch

PROMPTS = ["The capital of France is", "Compute 17 * 23. Answer:", "List three prime numbers:", "def fib(n):\n    return",
           "n=8\nhard: 2<4 1<5 1<4 6<0\ndisj: (6<7)|(3<1) (5<7)|(4<0) (0<3)|(6<3) (7<6)|(3<6) (5<2)|(4<3)\nreason inside <think>, then give the order\nanswer: 8 digits, first to last",
           "» 3<0 → 2<3<0<1 ⇒ × ⇒ 1<2 ⇒ 0<1<2<3 ✓ 0<2 + ⇒ 0 1 2 3"]

def trl_sync(peft_model, llm, adapter_prefix="lora_"):
    """镜像 TRL 0.25.1 _move_model_to_vllm（PEFT，非 FSDP/ZeRO-3 分支）。返回同步的参数名列表。"""
    peft_model.merge_adapter()
    loaded = []
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    try:
        for name, param in peft_model.named_parameters():
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            if adapter_prefix in name:
                continue
            if "original_module" in name:
                continue
            name = name.replace("modules_to_save.default.", "")
            llm_model.load_weights([(name, param.data)])
            loaded.append(name)
    finally:
        peft_model.unmerge_adapter()
    llm.reset_prefix_cache()
    return loaded

@torch.no_grad()
def hf_logps(peft_model, tok, prompts, merged=True):
    if merged: peft_model.merge_adapter()
    out = []
    try:
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to("cuda")
            logits = peft_model(input_ids=ids).logits[0, :-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, ids[0, 1:, None]).squeeze(-1)
            out.append(lp.cpu().tolist())
    finally:
        if merged: peft_model.unmerge_adapter()
    return out

def vllm_logps(llm, prompts):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    res = llm.generate(prompts, sp, use_tqdm=False)
    out = []
    for r in res:
        toks = list(r.prompt_token_ids)
        lps = []
        for t, d in zip(toks[1:], r.prompt_logprobs[1:]):
            lps.append(d[t].logprob if d and t in d else float("nan"))
        out.append(lps)
    return out

def max_abs_diff(a, b):
    ds = [abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb) if x == x and y == y]
    return (max(ds), statistics.mean(ds), len(ds)) if ds else (float("nan"), float("nan"), 0)

def make_untied_copy(model_name, out_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_dir = pathlib.Path(out_dir)
    if (out_dir / "config.json").exists():
        return str(out_dir)
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    m.config.tie_word_embeddings = False
    m.lm_head.weight = torch.nn.Parameter(m.get_input_embeddings().weight.detach().clone())
    m.save_pretrained(str(out_dir), safe_serialization=True); AutoTokenizer.from_pretrained(model_name).save_pretrained(str(out_dir))
    del m; gc.collect()
    return str(out_dir)

def run_variant(name, model_dir, targets, args, report):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from vllm import LLM
    print(f"\n=== variant {name}: targets={targets} model={model_dir} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    peft_model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM"))
    g = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():                       # 所有 lora_ 参数随机非零（PEFT：Linear 的 B、Embedding 的 A 初始为 0），使 delta 可检测
        for n, p in peft_model.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn(p.shape, generator=g, device="cuda", dtype=p.dtype) * 0.02)
    ref = hf_logps(peft_model, tok, PROMPTS, merged=True)
    base_only = hf_logps(peft_model, tok, PROMPTS, merged=False)
    delta_hf = max_abs_diff(ref, base_only)
    llm = LLM(model=model_dir, dtype="bfloat16", gpu_memory_utilization=args.gpu_util, max_model_len=512, enforce_eager=True,
              distributed_executor_backend="external_launcher", seed=0)
    v_before = vllm_logps(llm, PROMPTS); before = max_abs_diff(v_before, ref)
    loaded = trl_sync(peft_model, llm)
    v_after = vllm_logps(llm, PROMPTS); after = max_abs_diff(v_after, ref)
    ok = after[0] < args.tol
    # D3 追加诊断（不改原判定 ok）：跨引擎 bf16 噪声在"改动量"上一阶抵消。
    #   noise_floor = vLLM(基座) vs HF(基座)；delta_agreement = (vLLM 同步后 − 同步前) vs (HF 合并 − HF 基座) 的逐 token 差；delta_corr = 两个改动量的相关系数
    # D4：base_only 其实是"adapter 激活但未合并"的前向 = **训练端**算 logp 用的前向（TRL 同步后 unmerge）。真正的基座要 disable_adapter。
    with peft_model.disable_adapter():
        true_base = hf_logps(peft_model, tok, PROMPTS, merged=False)
    trainer_fwd = base_only
    true_noise = max_abs_diff(v_before, true_base)                       # vLLM(基座) vs HF(adapter 关)：纯跨引擎噪声
    sampler_vs_trainer = max_abs_diff(v_after, trainer_fwd)             # 采样端（同步后的 vLLM）vs 训练端前向（未合并）：GRPO 重要性比率看到的就是它
    merged_vs_trainer = max_abs_diff(ref, trainer_fwd)                  # 同一引擎内：HF 合并 vs HF 未合并（tied 时 embed 的 delta 合并后会漏进 lm_head）
    noise = max_abs_diff(v_before, base_only)
    d_v = [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(v_after, v_before)]; d_h = [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(ref, base_only)]
    agree = max_abs_diff(d_v, d_h)
    pairs = [(x, y) for rv, rh in zip(d_v, d_h) for x, y in zip(rv, rh) if x == x and y == y]
    fv = [x for x, _ in pairs]; fh = [y for _, y in pairs]
    mv, mh = statistics.mean(fv), statistics.mean(fh)
    den = (sum((x - mv) ** 2 for x in fv) * sum((y - mh) ** 2 for y in fh)) ** 0.5
    corr = (sum((x - mv) * (y - mh) for x, y in zip(fv, fh)) / den) if den else None
    rec = dict(targets=targets, model_dir=model_dir, hf_delta_vs_base=delta_hf, vllm_vs_hf_before_sync=before, vllm_vs_hf_after_sync=after,
               noise_floor_vllm_base_vs_hf_base=noise, delta_agreement=agree, delta_corr=corr, vllm_delta_max=max(abs(x) for x in fv),
               true_noise_floor=true_noise, sampler_vs_trainer=sampler_vs_trainer, hf_merged_vs_trainer_forward=merged_vs_trainer,
               synced_names_sample=[n for n in loaded if any(k in n for k in ("embed_tokens", "lm_head"))][:6], ok=ok)
    print(json.dumps(rec, indent=1), flush=True)
    report["variants"][name] = rec
    del llm, peft_model, base; gc.collect(); torch.cuda.empty_cache(); time.sleep(3)

VARIANTS = {"q_only": ["q_proj"], "q+embed": ["q_proj", "embed_tokens"], "q+embed+lmhead_tied": ["q_proj", "embed_tokens", "lm_head"]}   # 最后一个是阴性演示（tied → lm_head delta 不同步）

def _dist_env():
    """与 TRL 0.25.1 colocate 相同：external_launcher 需要 RANK / LOCAL_RANK / WORLD_SIZE 与 rendezvous 地址（单进程）。"""
    import socket
    os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    if "MASTER_PORT" not in os.environ:
        with socket.socket() as sk: sk.bind(("", 0)); os.environ["MASTER_PORT"] = str(sk.getsockname()[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B"); ap.add_argument("--out", default="results/lora_embed_sync_check.json")
    ap.add_argument("--gpu-util", type=float, default=0.35); ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--variant", choices=list(VARIANTS), help="只跑一个变体（每个变体单独进程：一个进程一个 vLLM 引擎 / 一次分布式初始化）")
    args = ap.parse_args()
    if args.variant:
        _dist_env()
        report = dict(variants={})
        run_variant(args.variant, args.model, VARIANTS[args.variant], args, report)
        json.dump(report["variants"][args.variant], open(args.out, "w"), indent=1); return
    import subprocess, vllm, trl, transformers, peft
    from transformers import AutoConfig
    report = dict(model=args.model, versions=dict(vllm=vllm.__version__, trl=trl.__version__, transformers=transformers.__version__, peft=peft.__version__), variants={},
                  tie_word_embeddings=bool(getattr(AutoConfig.from_pretrained(args.model), "tie_word_embeddings", False)))
    for name in VARIANTS:
        part = str(pathlib.Path(args.out).with_suffix(f".{name.replace('+', '_')}.json"))
        rc = subprocess.call([sys.executable, __file__, "--model", args.model, "--gpu-util", str(args.gpu_util), "--tol", str(args.tol), "--variant", name, "--out", part])
        report["variants"][name] = json.load(open(part)) if rc == 0 and pathlib.Path(part).exists() else dict(ok=False, error=f"subprocess rc={rc}")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True); json.dump(report, open(args.out, "w"), indent=1)
    print("\nSUMMARY:", {k: v.get("ok") for k, v in report["variants"].items()}, "->", args.out)

if __name__ == "__main__":
    main()
