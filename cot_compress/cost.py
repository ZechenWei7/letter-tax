"""--dry-run 成本测量（v7 §11）：20 步的每步墙钟、token/步、显存峰值 → 按 400 步 × 矩阵 run 数投影总 GPU 小时与费用。"""
from __future__ import annotations
import pathlib, statistics, yaml

def dry_run_report(recs: list[dict], cfg: dict, arm: str, matrix_path=None, steps_per_run: int = 400, usd_per_hour: float | None = None) -> dict:
    recs = [r for r in recs if r.get("step", 0) > 1] or recs           # 第 1 步含编译 / 首次同步，去掉
    G = int(cfg["train"]["num_generations"]); per = int(cfg["train"]["per_device_train_batch_size"]) * int(cfg["train"]["gradient_accumulation_steps"])
    sec = statistics.mean(r["sec"] for r in recs); gen = statistics.mean(r.get("gen_sec", 0) or 0 for r in recs)
    L = statistics.mean(r.get("L_mean") or 0 for r in recs)
    tokens_per_step = L * per                                              # 完成段 token（think 部分）× 每步样本数
    usd = usd_per_hour if usd_per_hour is not None else float(cfg.get("cost", {}).get("usd_per_hour", 1.9))
    n_runs = None
    if matrix_path and pathlib.Path(matrix_path).exists():
        m = yaml.safe_load(open(matrix_path)); n_runs = len(m.get("runs", [])); steps_per_run = int(m.get("defaults", {}).get("steps", steps_per_run))
    hours_per_run = sec * steps_per_run / 3600
    eval_min = float(cfg.get("cost", {}).get("eval_min_per_eval", 0.0)); n_evals = steps_per_run // max(1, int(cfg["task"].get("eval_every", 50))) + 1
    hours_per_run_total = hours_per_run + eval_min * n_evals / 60
    return dict(arm=arm, n_steps_measured=len(recs), sec_per_step=round(sec, 1), gen_sec_per_step=round(gen, 1), train_sec_per_step=round(sec - gen, 1),
                samples_per_step=per, G=G, L_mean=round(L, 1), tokens_per_step=round(tokens_per_step),
                peak_alloc_gib=max(r["peak_alloc_gib"] for r in recs), peak_reserved_gib=max(r["peak_reserved_gib"] for r in recs),
                projection=dict(steps_per_run=steps_per_run, hours_per_run_train=round(hours_per_run, 2), hours_per_run_incl_eval=round(hours_per_run_total, 2),
                                n_runs_in_matrix=n_runs, total_gpu_hours=(round(hours_per_run_total * n_runs, 1) if n_runs else None),
                                usd_per_hour=usd, total_usd=(round(hours_per_run_total * n_runs * usd, 0) if n_runs else None)),
                summary=dict(sec_per_step=round(sec, 1), gib=max(r["peak_reserved_gib"] for r in recs), gpu_h_total=(round(hours_per_run_total * n_runs, 1) if n_runs else None)))
