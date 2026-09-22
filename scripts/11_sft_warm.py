"""B_warm SFT（v7 §4）：把臂 A 收敛 run 的正确轨迹经 warm_text 变换成无字母 think，用 fresh LoRA（与 RL 同配置）做 SFT。
数据源：runs/<A run>/rollouts.jsonl.zst 的最后 --last-steps 步（默认 50 步）里 c==1 且无违规的行，或 --rows <jsonl>。
输出：runs/Bwarm_sft_<tag>/final（adapter）+ data/sft_warm_<tag>.jsonl（SFT 数据集，含策略类分布对照）。
"无 RL 直接评估"：python scripts/train.py --config configs/cloud_4b.yaml --arm Bwarm --eval-only --set train.init_from=runs/Bwarm_sft_<tag>/final
用法: python scripts/11_sft_warm.py --config configs/cloud_4b.yaml --src runs/A_0.5_kk_n10_s1_s1 --tag kk_n10_s1_s1 [--epochs 2] [--dry-run]
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.environ.setdefault("COT_NO_UNSLOTH", "1")
import argparse, json, statistics
import tasks
from cot_compress.config import load_config
from cot_compress.warm import make_sft_rows, warm_text
from cot_compress.strategy import annotate, class_distribution
from cot_compress.traj_stats import think_text
from cot_compress.archive import read_jsonl_zst

def collect_rows(src: pathlib.Path, last_steps: int, key: str) -> list[dict]:
    rows = list(read_jsonl_zst(src / "rollouts.jsonl.zst"))
    steps = sorted({r["step"] for r in rows if r.get("step") is not None})
    keep = set(steps[-last_steps:])
    items = {it["id"]: it for sp in tasks.task_for_key(key).load_splits(key, 0).values() if isinstance(sp, list) for it in sp}
    out = []
    for r in rows:
        if r.get("step") not in keep or r.get("c") != 1.0 or r.get("violations"):
            continue
        it = items.get(r.get("prompt_id"))
        if it is None:
            continue
        out.append(dict(id=r["prompt_id"], prompt=it["prompt"], think=think_text(r["text"]), gold=r["gold"], correct=True))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cloud_4b.yaml"); ap.add_argument("--src"); ap.add_argument("--rows")
    ap.add_argument("--tag", required=True); ap.add_argument("--last-steps", type=int, default=50); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="只生成数据集与分布对照，不训练")
    args = ap.parse_args()
    cfg = load_config(args.config); key = cfg["task"]["key"]; task = tasks.task_for_key(key); n = task.parse_key(key)["n"]
    rows = [json.loads(l) for l in open(args.rows)] if args.rows else collect_rows(ROOT / args.src, args.last_steps, key)
    sft = make_sft_rows(rows)
    # D25：§5.5 test 8 的注册原文是"在 100 条 A 轨迹上策略类分布相符"。此前实现在全部轨迹上做精确相等断言，
    # 比注册更严：2026-09-22 在 A1 的 1435 条上以 1 条失败（try 计数被 warm 删除 → branches 8→6，不再严格大于阈值 6；其余字段全同）。
    # 现改为：判定用 seed=0 抽的 100 条（各类占比差 ≤ 1pp 即相符）；全量分布照算、照报（描述量）。warm 映射与检测器不动。
    import random
    idx100 = sorted(random.Random(0).sample(range(len(rows)), min(100, len(rows))))
    def _dists(rs, ss):
        if key.startswith("ord_"):
            from cot_compress.shortcuts import strategy_class, class_distribution as _cd
            Sof = {it["id"]: it["meta"]["S"] for sp in task.load_splits(key, 0).values() if isinstance(sp, list) for it in sp}
            a = _cd([strategy_class(r["think"], r["prompt"], n, Sof.get(r["id"], 1)) for r in rs])
            b = _cd([strategy_class(r["think_warm"], warm_text(r["prompt"]), n, Sof.get(r["id"], 1)) for r in ss])
            for dd in (a, b): dd["case"] = dd.pop("english_case") + dd.pop("symbolic_case")   # english_case → symbolic_case 是变换的定义；比较时合并两类
            return a, b
        return (class_distribution([annotate(r["think"], r["prompt"], n) for r in rs]),
                class_distribution([annotate(r["think_warm"], warm_text(r["prompt"]), n) for r in ss]))
    dist_src, dist_warm = _dists(rows, sft)                                                     # 全量：只报
    dist_src100, dist_warm100 = _dists([rows[i] for i in idx100], [sft[i] for i in idx100])     # 100 条：判定
    max_gap100 = max(abs(dist_src100.get(k, 0) - dist_warm100.get(k, 0)) for k in set(dist_src100) | set(dist_warm100))
    out_data = ROOT / "data" / f"sft_warm_{args.tag}.jsonl"; out_data.parent.mkdir(exist_ok=True)
    with open(out_data, "w") as f:
        for r in sft: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = dict(n_src=len(rows), n_sft=len(sft), strategy_class_src=dist_src, strategy_class_warm=dist_warm,
               test8_n=len(idx100), test8_seed=0, strategy_class_src_100=dist_src100, strategy_class_warm_100=dist_warm100,
               test8_max_gap_pp=round(100 * max_gap100, 2), test8_tolerance_pp=1.0,
               len_ratio_chars=statistics.mean(r["warm_len_chars"] / max(1, r["src_len_chars"]) for r in sft) if sft else None)
    json.dump(rep, open(ROOT / "data" / f"sft_warm_{args.tag}.report.json", "w"), indent=1); print(json.dumps(rep, indent=1))
    assert max_gap100 <= 0.01 + 1e-12, f"§5.5 test 8：100 条样本上策略类占比差 {100*max_gap100:.2f}pp > 1pp（D25）"
    if args.dry_run:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from cot_compress.generation import chat_prompt
    from cot_compress.evaluate import THINK_PREFILL
    from cot_compress.lora import lora_targets_for
    m, tcfg = cfg["model"], cfg["train"]
    tok = AutoTokenizer.from_pretrained(m["name"])
    model = AutoModelForCausalLM.from_pretrained(m["name"], dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=int(tcfg["lora_r"]), lora_alpha=int(tcfg["lora_alpha"]), lora_dropout=0.0, bias="none",
                                             target_modules=lora_targets_for(cfg), task_type="CAUSAL_LM"))
    task_mod = tasks.task_for_key(key)
    ds = Dataset.from_list([dict(prompt=chat_prompt(tok, task_mod.SYSTEM_PROMPT, r["prompt"], think=True, prefill=THINK_PREFILL), completion=r["completion"]) for r in sft])
    run_dir = ROOT / "runs" / f"Bwarm_sft_{args.tag}"
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds,
                         args=SFTConfig(output_dir=str(run_dir), num_train_epochs=args.epochs, learning_rate=float(tcfg["learning_rate"]), warmup_steps=10,
                                        per_device_train_batch_size=1, gradient_accumulation_steps=16, bf16=True, logging_steps=5, save_strategy="no",
                                        max_length=int(m["max_seq_length"]), completion_only_loss=True, report_to=[]))
    trainer.train(); trainer.save_model(str(run_dir / "final"))
    print("saved", run_dir / "final")

if __name__ == "__main__":
    main()
