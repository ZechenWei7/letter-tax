"""第二阶段闸门 1：在一种写法的推导上 SFT（固定步数）。不在 OSF 注册的 stage-1 协议之内。
训练部分逐行照搬 scripts/11_sft_warm.py（B_warm-SFT）：新 LoRA adapter（与 RL 同配置，attention+MLP，D5）、completion-only loss、
lr = cloud_4b.yaml train.learning_rate、warmup 10、per-device 1 × GA 16、bf16。差别只有三处：
  (1) 数据读 data/phase2/gate1_sft_<N>.jsonl（scripts/phase2_build_sft.py 产出，sha256 核对 manifest）；
  (2) 训练长度用固定 max_steps（configs/phase2_gate1.yaml，默认 250）而不是 epochs；
  (3) seed 显式设置（transformers.set_seed 在建 LoRA 之前 → 决定 LoRA 初始化与数据顺序）。
输出：runs/gate1_<N>_s<seed>/final（adapter）+ train_log.json。
用法：python scripts/phase2_gate1_train.py --notation N3 --seed 1 [--gate-config configs/phase2_gate1.yaml] [--dry-run]"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.environ.setdefault("COT_NO_UNSLOTH", "1")
import argparse, hashlib, json, time
import yaml
import tasks
from cot_compress.config import load_config


def load_rows(gcfg, notation):
    path = ROOT / gcfg["train"]["data"][notation]
    man = json.load(open(ROOT / "data/phase2/manifest.json"))
    want = man["files"][notation]["sha256"]; got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == want, f"{path} sha256 {got} != manifest {want}"
    rows = [json.loads(l) for l in open(path)]
    assert all(r["notation"] == notation for r in rows)
    return rows, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notation", required=True, choices=["N2", "N2s", "N3"]); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gate-config", default="configs/phase2_gate1.yaml"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    g = yaml.safe_load(open(ROOT / a.gate_config))
    assert a.notation in g["notations"] and a.seed in g["seeds"], "notation / seed not in gate config"
    cfg = load_config(g["base_config"], None); key = g["key"]; task = tasks.task_for_key(key)
    rows, sha = load_rows(g, a.notation)
    T = g["train"]; run_dir = ROOT / "runs" / f"gate1_{a.notation}_s{a.seed}"
    plan = dict(notation=a.notation, seed=a.seed, n_rows=len(rows), data_sha256=sha, max_steps=int(T["max_steps"]),
                lr=float(cfg["train"]["learning_rate"]), warmup_steps=int(T["warmup_steps"]),
                per_device=int(T["per_device_train_batch_size"]), grad_accum=int(T["gradient_accumulation_steps"]), run_dir=str(run_dir))
    assert float(T["learning_rate"]) == plan["lr"], "gate config lr must equal cloud_4b train.learning_rate (B_warm-SFT)"
    print(json.dumps(plan, indent=1))
    if a.dry_run: return
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from cot_compress.generation import chat_prompt
    from cot_compress.evaluate import THINK_PREFILL
    from cot_compress.lora import lora_targets_for
    transformers.set_seed(a.seed)
    m, tcfg = cfg["model"], cfg["train"]
    tok = AutoTokenizer.from_pretrained(m["name"])
    model = AutoModelForCausalLM.from_pretrained(m["name"], dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=int(tcfg["lora_r"]), lora_alpha=int(tcfg["lora_alpha"]), lora_dropout=0.0, bias="none",
                                             target_modules=lora_targets_for(cfg), task_type="CAUSAL_LM"))
    ds = Dataset.from_list([dict(prompt=chat_prompt(tok, task.SYSTEM_PROMPT, r["prompt"], think=True, prefill=THINK_PREFILL), completion=r["completion"]) for r in rows])
    t0 = time.time()
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds,
                         args=SFTConfig(output_dir=str(run_dir), max_steps=plan["max_steps"], learning_rate=plan["lr"], warmup_steps=plan["warmup_steps"],
                                        per_device_train_batch_size=plan["per_device"], gradient_accumulation_steps=plan["grad_accum"], bf16=True,
                                        logging_steps=5, save_strategy="no", max_length=int(m["max_seq_length"]), completion_only_loss=True,
                                        seed=a.seed, data_seed=a.seed, report_to=[]))
    trainer.train(); trainer.save_model(str(run_dir / "final"))
    log = [h for h in trainer.state.log_history if "loss" in h]
    json.dump(dict(plan, sec=round(time.time() - t0, 1), log=log, final_loss=(log[-1]["loss"] if log else None),
                   final_token_acc=(log[-1].get("mean_token_accuracy") if log else None)), open(run_dir / "train_log.json", "w"), indent=1)
    print("saved", run_dir / "final")


if __name__ == "__main__":
    main()
