"""环境验证：torch/cu128/sm_120、bitsandbytes 4-bit、Unsloth 加载 + 生成、TRL 版本与 GRPO 钩子。
用法: python scripts/00_check_env.py [--skip-model] [--qwen35]

NOTE: `import unsloth` 必须在任何 trl / transformers 导入之前。trl 0.24.0 的 callbacks.py 无条件
import mergekit（未安装），裸 `from trl import GRPOTrainer` 会 ModuleNotFoundError；Unsloth 先导入则会绕过。
项目内约定：所有脚本第一行 import cot_compress（其 __init__ 无条件 import unsloth）。
"""
import argparse, os, sys, time, importlib, json
try:
    import unsloth  # noqa: F401  显式先于 trl 导入（本地 trl 0.24 的硬要求）
    HAVE_UNSLOTH = True
except Exception as _e:      # 云端 TRL 原生路径可以没有 unsloth
    HAVE_UNSLOTH = False; print("unsloth not importable:", repr(_e)[:200])

def section(t):
    print(f"\n==== {t} ====", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--qwen35", action="store_true", help="额外尝试 Qwen3.5-2B bf16 加载")
    ap.add_argument("--hf-model", default=None, help="云端：用 transformers bf16 加载该模型并生成（如 Qwen/Qwen3-4B），不走 Unsloth")
    args = ap.parse_args()
    report = {}

    section("torch")
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    assert torch.cuda.is_available(), "CUDA not available"
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print("gpu", name, "cap", cap, f"total {total:.2f} GiB")
    print("arch list", torch.cuda.get_arch_list())
    need = f"sm_{cap[0]}{cap[1]}"
    assert need in torch.cuda.get_arch_list(), f"torch build lacks {need} kernels; wrong wheel"
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().sum().item()
    print("bf16 matmul ok", y != 0)
    report["torch"] = dict(version=torch.__version__, cuda=torch.version.cuda, cap=cap, gpu=name, total_gib=round(total, 2))

    section("bitsandbytes 4-bit")
    try:
        import bitsandbytes as bnb
    except ImportError:
        bnb = None; print("bitsandbytes not installed (OK on cloud: bf16 LoRA)"); report["bnb"] = None
    if bnb is not None:
        _bnb_check(bnb, torch, report)

    _rest(args, report, section, torch)

def _bnb_check(bnb, torch, report):
    print("bnb", bnb.__version__)
    lin = bnb.nn.Linear4bit(1024, 1024, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
    out = lin(torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16))
    print("Linear4bit forward ok", tuple(out.shape), "finite", bool(torch.isfinite(out).all()))
    report["bnb"] = bnb.__version__


def _rest(args, report, section, torch):
    section("versions")
    vers = {}
    for m in ["unsloth", "unsloth_zoo", "trl", "transformers", "peft", "accelerate", "datasets", "triton", "vllm", "zstandard"]:
        try:
            mod = importlib.import_module(m)
            vers[m] = getattr(mod, "__version__", "?")
        except Exception as e:
            vers[m] = f"IMPORT FAIL: {e}"
    print(json.dumps(vers, indent=2))
    report["versions"] = vers

    section("TRL GRPO hooks")
    assert (not HAVE_UNSLOTH) or "unsloth" in sys.modules, "unsloth must be imported before trl"
    import trl
    from trl import GRPOConfig, GRPOTrainer
    import inspect
    cfg_fields = set(GRPOConfig.__dataclass_fields__.keys())
    for k in ["loss_type", "scale_rewards", "beta", "generation_kwargs", "num_generations", "max_completion_length", "use_vllm"]:
        print(f"  GRPOConfig.{k}:", k in cfg_fields)
    # Unsloth 把 GRPOTrainer 换成 UnslothGRPOTrainer，外层 __init__(*args, **kwargs)，签名检查失效；
    # 改为直接读 trl 原始模块源码。rollout_func 需要 trl>=0.25（README §9）。
    import trl.trainer.grpo_trainer as _g
    grpo_src = inspect.getsource(_g)
    has_rollout = "rollout_func" in grpo_src
    has_reward = "reward_funcs" in grpo_src
    print("  trl version:", trl.__version__)
    print("  GRPOTrainer(rollout_func=...):", has_rollout, "" if has_rollout else "(requires trl>=0.25; local generate hook only)")
    print("  GRPOTrainer(reward_funcs=...):", has_reward)
    report["trl_hooks"] = dict(trl=trl.__version__, rollout_func=has_rollout, reward_funcs=has_reward,
                               loss_type="loss_type" in cfg_fields, scale_rewards="scale_rewards" in cfg_fields)

    if args.skip_model:
        os.makedirs("results", exist_ok=True); json.dump(report, open("results/env_check.json", "w"), indent=2, ensure_ascii=False)
        print(json.dumps(report, indent=2, ensure_ascii=False)); return

    if args.hf_model:
        section(f"transformers bf16 load {args.hf_model} + generate")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        tok = AutoTokenizer.from_pretrained(args.hf_model)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype=torch.bfloat16, device_map="cuda")
        print(f"loaded in {time.time()-t0:.1f}s, alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB")
        ids = tok.apply_chat_template([{"role": "user", "content": "What is 17 * 23? Think step by step."}], add_generation_prompt=True, return_tensors="pt", enable_thinking=True)
        ids = (ids["input_ids"] if hasattr(ids, "input_ids") else ids).to("cuda")
        t0 = time.time(); out = model.generate(ids, max_new_tokens=64, do_sample=False); dt = time.time() - t0
        print(f"generated 64 tokens in {dt:.1f}s ({64/dt:.1f} tok/s)"); print("sample:", repr(tok.decode(out[0, ids.shape[1]:])[:200]))
        report["hf_model"] = dict(name=args.hf_model, load_alloc_gib=round(torch.cuda.memory_allocated()/2**30, 2),
                                  peak_gib=round(torch.cuda.max_memory_allocated()/2**30, 2), tok_per_s=round(64/dt, 1))
        os.makedirs("results", exist_ok=True); json.dump(report, open("results/env_check.json", "w"), indent=2, ensure_ascii=False)
        print("\nwrote results/env_check.json"); return

    section("Unsloth load Qwen3-1.7B 4-bit + generate")
    from unsloth import FastLanguageModel
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-1.7B", max_seq_length=2048, load_in_4bit=True, fast_inference=False,
    )
    print(f"loaded in {time.time()-t0:.1f}s, alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB")
    FastLanguageModel.for_inference(model)
    msgs = [{"role": "user", "content": "What is 17 * 23? Think step by step."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", enable_thinking=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = ids.to("cuda")
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=64, do_sample=False)
    dt = time.time() - t0
    text = tok.decode(out[0, ids.shape[1]:])
    print(f"generated 64 tokens in {dt:.1f}s ({64/dt:.1f} tok/s)")
    print("sample:", repr(text[:300]))
    print(f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    report["qwen3_1p7b_4bit"] = dict(load_alloc_gib=round(torch.cuda.memory_allocated()/2**30, 2),
                                     peak_gib=round(torch.cuda.max_memory_allocated()/2**30, 2), tok_per_s=round(64/dt, 1))
    del model; torch.cuda.empty_cache()

    if args.qwen35:
        section("Unsloth load Qwen3.5-2B bf16 (optional)")
        try:
            torch.cuda.reset_peak_memory_stats()
            model, tok = FastLanguageModel.from_pretrained(
                model_name="unsloth/Qwen3.5-2B", max_seq_length=2048, load_in_4bit=False, fast_inference=False,
            )
            print(f"loaded, alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB")
            report["qwen35_2b_bf16"] = dict(load_alloc_gib=round(torch.cuda.memory_allocated()/2**30, 2))
            del model; torch.cuda.empty_cache()
        except Exception as e:
            print("Qwen3.5-2B load FAILED:", repr(e)[:500])
            report["qwen35_2b_bf16"] = f"FAIL: {repr(e)[:300]}"

    os.makedirs("results", exist_ok=True)
    with open("results/env_check.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nwrote results/env_check.json")

if __name__ == "__main__":
    main()
