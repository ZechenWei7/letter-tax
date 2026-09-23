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


def budget_projection(spent_h: float, rate: float, steps_left: int, sec_per_step: float, evals_left: int, sec_per_eval: float, tail_h: float = 0.0) -> dict:
    """D26 费用守卫的投影（纯函数，单测 tests/test_budget.py）：已花 + 剩余步 × 步时 + 剩余 eval × eval 时长 + 收尾预留，均按墙钟 × 单价。"""
    left_h = (max(steps_left, 0) * sec_per_step + max(evals_left, 0) * sec_per_eval) / 3600 + tail_h
    return dict(spent_usd=round(spent_h * rate, 2), remaining_usd=round(left_h * rate, 2), projected_usd=round((spent_h + left_h) * rate, 2),
                steps_left=int(steps_left), sec_per_step=round(sec_per_step, 1), evals_left=int(evals_left), sec_per_eval=round(sec_per_eval, 1))


def gate1_projection(spent_h: float, rate: float, runs_left_train: int, train_sec: float, runs_left_eval: int, eval_sec: float, tail_h: float = 0.0) -> dict:
    """闸门 1 费用投影：已花 + 未训练的 run × 实测训练时长 + 未评估的 run × 每 run 评估时长（冒烟后是线性外推上界）+ 收尾（判读 / 报告）。"""
    left_h = (runs_left_train * train_sec + runs_left_eval * eval_sec) / 3600 + tail_h
    return dict(spent_usd=round(spent_h * rate, 2), remaining_usd=round(left_h * rate, 2), projected_usd=round((spent_h + left_h) * rate, 2),
                runs_left_train=runs_left_train, train_sec=round(train_sec, 1), runs_left_eval=runs_left_eval, eval_sec=round(eval_sec, 1))
