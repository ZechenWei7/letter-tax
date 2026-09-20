#!/usr/bin/env python3
"""宿主机 backward 阶梯检查（排查 / 新机器验收用；不走 TRL、不走 vLLM、不 import 项目代码，不是实验的一部分）。

背景：A1 与不带 eval 的 3 步 dry-run 都在迁移后的宿主机（driver 595.91.07）第 1 个训练步 loss.backward() 处
报 CUDA error: invalid argument；上一台（580.159.04）同代码同 venv 跑通。
每一级独立 try，打印 PASS / FAIL 与异常首行；任何一级 FAIL 都继续跑后面的，最后汇总。
  R1 纯 torch：bf16 matmul backward
  R2 纯 torch：SDPA 因果注意力 backward，seq 2k / 10.7k（训练用的就是 sdpa）
  R3 纯 torch：non-reentrant checkpoint 包一层 SDPA 的 backward
  R4 HF Qwen3-4B bf16 + PEFT LoRA(r32, attention+MLP) + 梯度检查点，单条序列 seq 2k / 10.7k 的 loss.backward()（与 train.py 云端分支同样的加载方式）
用法（pod 上）：CUDA_LAUNCH_BLOCKING=1 python ops/host_check.py [--skip-model]
"""
import argparse, json, sys, time, traceback
import torch

RES = []


def rung(name, fn):
    torch.cuda.synchronize(); t0 = time.time()
    try:
        fn(); torch.cuda.synchronize()
        RES.append(dict(rung=name, ok=True, sec=round(time.time() - t0, 1), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2)))
        print(f"PASS {name} ({RES[-1]['sec']} s, peak {RES[-1]['peak_gib']} GiB)", flush=True)
    except Exception as e:
        RES.append(dict(rung=name, ok=False, err=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"))
        print(f"FAIL {name}: {RES[-1]['err']}", flush=True); traceback.print_exc(limit=6)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def r1():
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    (a @ a).float().pow(2).mean().backward(); assert torch.isfinite(a.grad).all()


def sdpa(seq, ckpt=False):
    def f():
        q, k, v = (torch.randn(1, 32, seq, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
        att = lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = torch.utils.checkpoint.checkpoint(att, q, k, v, use_reentrant=False) if ckpt else att(q, k, v)
        out.float().mean().backward(); assert torch.isfinite(q.grad).all()
    return f


def model_rungs(name, with_vllm=False):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.train()

    def bw(seq):
        def f():
            ids = torch.randint(1000, 50000, (1, seq), device="cuda")
            logits = model(input_ids=ids).logits[:, -1024:, :].float()          # 只对最后 1024 个位置算 loss，避免 [T,V] fp32 撑爆显存；backward 仍穿过整条序列
            loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, -1023:]); loss.backward(); model.zero_grad(set_to_none=True)
            assert torch.isfinite(loss)
        return f
    def bw_masked(seq, pad):
        def f():                                                              # 左填充 + attention_mask：HF 会传 4D mask，SDPA 走带 mask 的 kernel（训练实况；R4 的 is_causal 路径不覆盖）
            ids = torch.randint(1000, 50000, (1, seq), device="cuda"); am = torch.ones_like(ids); am[:, :pad] = 0
            logits = model(input_ids=ids, attention_mask=am).logits[:, -1024:, :].float()
            loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, -1023:]); loss.backward(); model.zero_grad(set_to_none=True)
            assert torch.isfinite(loss)
        return f
    rung("R4a model+LoRA+ckpt backward seq 2048", bw(2048))
    rung("R4b model+LoRA+ckpt backward seq 10752", bw(10752))
    rung("R5a masked (left-pad 300) backward seq 2048", bw_masked(2048, 300))
    rung("R5b masked (left-pad 300) backward seq 10752", bw_masked(10752, 300))
    if with_vllm:                                                             # R6：与 TRL colocate 相同的方式在同进程起 vLLM（external_launcher），生成一次，再做同样的 backward
        import os, socket
        for k, v in dict(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="localhost").items(): os.environ.setdefault(k, v)
        if "MASTER_PORT" not in os.environ:
            with socket.socket() as sk: sk.bind(("", 0)); os.environ["MASTER_PORT"] = str(sk.getsockname()[1])
        holder = {}
        def start():
            from vllm import LLM, SamplingParams
            holder["llm"] = LLM(model=name, dtype="bfloat16", gpu_memory_utilization=0.45, max_model_len=11264, distributed_executor_backend="external_launcher", seed=0)
            out = holder["llm"].generate(["n=8\nhard: 0<3 2<5"] * 8, SamplingParams(max_tokens=256, temperature=1.0))
            assert len(out) == 8
        rung("R6a start colocated vLLM (util 0.45) + generate", start)
        rung("R6b backward seq 2048 with vLLM resident", bw(2048))
        rung("R6c masked backward seq 10752 with vLLM resident", bw_masked(10752, 300))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-model", action="store_true"); ap.add_argument("--with-vllm", action="store_true"); ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="results/host_check.json"); a = ap.parse_args()
    import subprocess
    drv = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"host: {drv} | torch {torch.__version__} cuda {torch.version.cuda}", flush=True)
    rung("R1 bf16 matmul backward", r1)
    rung("R2a SDPA causal backward seq 2048", sdpa(2048))
    rung("R2b SDPA causal backward seq 10752", sdpa(10752))
    rung("R3 checkpoint(SDPA) backward seq 10752", sdpa(10752, ckpt=True))
    if not a.skip_model: model_rungs(a.model, a.with_vllm)
    rep = dict(time=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), host=drv, torch=torch.__version__, cuda=torch.version.cuda, rungs=RES, all_ok=all(r["ok"] for r in RES))
    json.dump(rep, open(a.out, "w"), indent=1); print(json.dumps(rep, ensure_ascii=False)); sys.exit(0 if rep["all_ok"] else 1)


if __name__ == "__main__":
    main()
