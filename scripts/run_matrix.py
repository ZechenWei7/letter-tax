"""云端批量启动（v8）：按矩阵顺序跑 run，每个 run 结束跑内容诊断（reporting-eval）。
- --key 覆盖矩阵里所有 run 的格（准入通过的格）。speed_test: true 的 run 先 --dry-run 20 步（写 results/dry_run_*.json）再正式跑。
- warm_check: true 的 run（B1）跑完后读其 eval.jsonl 中 step ≤ 200 的最佳准确率，与 A1 的最佳准确率比：< A1 − 15pp → cold_start_failure=True（写 results/warm_decision_<key>.json），
  条件项 Bwarm（conditional: cold_start_failure）据此决定是否跑；跑之前做预算投影检查（budget_projection：已花 + 待跑 × 每 run 投影 vs cost.budget_usd，超则不跑）。
  Bwarm_sft 项 source: A_seed → 用同 seed 的 A run 作 SFT 源（s1 紧跟 A1；s2 / s3 分别在 A2 / A3 之后）。C_rand seed i 绑 B seed i（crand_source=auto）；think 长度每个 prompt 组抽一次（G 条生成共享）。
- 第一个 run 固定为 A seed 1（操纵门）：相对臂 0（step-0 eval）token 减少 ≥ 30%、acc 损失 ≤ 5pp、且收敛 L_median ≥ 2× Kahn 下限（c=2.5，stopping-eval 中位；c=2 / 3 只报）。
  kill 判定：A1 收敛 L_median ≤ 1.5 × decision_lb_tokens["2.5"] 中位数 → 矩阵直接停下报告，不重调。
  不满足 → 按 λ=0.3 → λ=1.0 → G=32 顺序重调 A s1（tag _lam0.3 / _lam1.0 / _G32）；某次通过则后续全部 run 继承该设置；全失败 → 停下报告。
- --max-runs N：按 priority（越小越先裁）裁剪；C_rand(1) → A″(2) → A/B(3)。
- 抢占恢复：已有 checkpoint 自动 --resume；已有 final/ 跳过。
- Bwarm_sft 项：11_sft_warm.py（源 = 预注册端点下最佳 A seed）+ train.py --arm Bwarm --eval-only（B_warm-SFT 无 RL 直接评估）。
- Bwarm 项（conditional: cold_start_failure）：仅当 B 各 seed 最佳外化准确率 < A 最佳 − 15pp 才跑；init_from=auto → SFT adapter。
- C_rand 的 `train.crand_source=auto` 解析为同 seed 的已完成 B run（B 全部完成后再跑）。
用法: python scripts/run_matrix.py --matrix configs/matrix_core.yaml [--max-runs 12] [--dry-run] [--post-cmd "..."]
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import argparse, json, subprocess, time, yaml
from cot_compress.guard import assert_writable, WorkspaceNotWritable

def run_name(r):
    if r["arm"] == "Bwarm_sft":
        return f"Bwarm_sft_{r['key']}_s{int(r['seed'])}{r.get('tag', '')}"
    return f"{r['arm']}_{float(r['lam'])}_{r['key']}_s{int(r['seed'])}{r.get('tag', '')}"

RETUNE = [dict(tag="_lam0.3", set=["reward.lambda=0.3"], lam=0.3), dict(tag="_lam1.0", set=["reward.lambda=1.0"], lam=1.0),
          dict(tag="_G32", set=["train.num_generations=32", "train.gradient_accumulation_steps=64"], lam=None)]

def oracle_lb_median(key, c="2.5", which="kahn"):
    """stopping-eval 500 题下限（token）的中位数。which="kahn"：Kahn 下限（操纵门的 2× 条件，与 §4.2 第 8 条同一下限）；which="decision"：决策下限 d+n（kill 判定）。
    默认 c=2.5；门里另报 c=2 / 3。无划分文件则读 results/admission_<key>.json。"""
    import statistics
    try:
        import tasks
        items = tasks.task_for_key(key).load_splits(key, 0)["stop"]
        lbk = ("kahn_lb_tokens" if which == "kahn" else "decision_lb_tokens") if key.startswith("ord_") else "oracle_lb"
        return float(statistics.median(it["meta"][lbk][c] for it in items))
    except Exception:
        f = ROOT / f"results/admission_{key}.json"
        return float(json.load(open(f))[("kahn_lb" if which == "kahn" else "decision_lb")][c]) if f.exists() else None

def externalized(diag):
    return diag["acc"] - diag.get("direct_acc", 0.0)

def best_acc_upto(run_dir: pathlib.Path, max_step=None):
    ev = [json.loads(l) for l in open(run_dir / "eval.jsonl")] if (run_dir / "eval.jsonl").exists() else []
    ev = [e for e in ev if max_step is None or e["step"] <= max_step]
    return max((e["acc"] for e in ev), default=None)

def warm_decision(key, lam, thr=0.15):
    """v8：B1 在 ≤200 步的最佳准确率 < A1 最佳准确率 − 15pp → cold_start_failure。写 results/warm_decision_<key>.json。"""
    a = best_acc_upto(ROOT / "runs" / f"A_{float(lam)}_{key}_s1"); b = best_acc_upto(ROOT / "runs" / f"B_{float(lam)}_{key}_s1", 200)
    dec = dict(A1_best=a, B1_best_upto200=b, cold_start_failure=(None if a is None or b is None else b < a - thr))
    json.dump(dec, open(ROOT / f"results/warm_decision_{key}.json", "w"), indent=1); return dec

def budget_projection(cfg_path, n_extra_runs: int) -> dict:
    """B_warm-RL 前的预算检查：已花 GPU 小时（各 run steps.jsonl 的 sec 之和）+ n_extra_runs × 每 run 投影小时（results/dry_run_*.json），
    × cost.usd_per_hour，对比 cost.budget_usd（未设 → ok=None，只记录投影）。超预算 → 不跑 B_warm-RL。"""
    from cot_compress.config import load_config
    cfg = load_config(cfg_path); cost = cfg.get("cost", {}) or {}
    usd_h = float(cost.get("usd_per_hour", 1.9)); budget = cost.get("budget_usd")
    spent_h = 0.0
    for f in (ROOT / "runs").glob("*/steps.jsonl"):
        spent_h += sum(json.loads(l).get("sec", 0) for l in open(f)) / 3600
    per_run = [json.load(open(f))["projection"]["hours_per_run_incl_eval"] for f in (ROOT / "results").glob("dry_run_*.json")]
    h_run = max(per_run) if per_run else None
    proj_h = (spent_h + n_extra_runs * h_run) if h_run is not None else None
    ok = None if (budget is None or proj_h is None) else (proj_h * usd_h <= float(budget))
    return dict(ok=ok, spent_gpu_hours=round(spent_h, 1), hours_per_run=h_run, n_extra_runs=n_extra_runs, projected_gpu_hours=(round(proj_h, 1) if proj_h is not None else None),
                projected_usd=(round(proj_h * usd_h) if proj_h is not None else None), budget_usd=budget)

def cold_start_failure(key, lam, thr=0.15):
    f = ROOT / f"results/warm_decision_{key}.json"
    if f.exists():
        d = json.load(open(f)); return d["cold_start_failure"], d
    d = warm_decision(key, lam, thr); return d["cold_start_failure"], d

def best_A_run(key, lam):
    """B_seq 的 init_from：预注册端点（收敛 (L, acc)）下的最佳 A seed = acc 最高，平手取 L 更短。需要 diagnostics.json。"""
    cands = []
    for d in (ROOT / "runs").glob(f"A_{float(lam)}_{key}_s*"):
        f = d / "diagnostics.json"
        if f.exists() and (d / "final").exists():
            j = json.load(open(f)); cands.append((-j["acc"], j["L_median"], d.name))
    if not cands:
        raise SystemExit(f"[FATAL] no finished A runs with diagnostics for {key} λ={lam}; cannot resolve init_from=auto")
    cands.sort(); return cands[0][2]

def resolve_auto(sets, r, d):
    out = []
    for s_ in sets:
        if s_ == "train.init_from=auto":
            name = f"Bwarm_sft_{r['key']}_s{int(r['seed'])}"
            if not (ROOT / "runs" / name / "final").exists():
                raise SystemExit(f"[FATAL] Bwarm needs SFT adapter runs/{name}/final (run the Bwarm_sft matrix item first)")
            out.append(f"train.init_from=runs/{name}/final"); print(f"[auto] init_from -> {name}")
        elif s_ == "train.crand_source=auto":
            name = f"B_{float(r['lam'])}_{r['key']}_s{int(r['seed'])}"
            if not (ROOT / "runs" / name / "diagnostics_rows.jsonl").exists():
                raise SystemExit(f"[FATAL] C_rand needs finished B run {name} (diagnostics_rows.jsonl); B runs must all complete first")
            out.append(f"train.crand_source=runs/{name}"); print(f"[auto] crand_source -> {name}")
        else:
            out.append(s_)
    return out

def prune(runs, max_runs):
    """按 priority 升序裁掉（同 priority 内后面的先裁），直到 ≤ max_runs；priority 缺省 = 99（最后裁）。"""
    if max_runs is None or len(runs) <= max_runs:
        return runs
    order = sorted(range(len(runs)), key=lambda i: (runs[i].get("priority", 99), -i))
    drop = set(order[:len(runs) - max_runs])
    return [r for i, r in enumerate(runs) if i not in drop]

def first_run_gate(run_dir: pathlib.Path, min_reduction=0.30, max_acc_loss=0.05, oracle_lb=None, oracle_mult=2.0, kill_lb=None, kill_mult=1.5) -> dict:
    """操纵门（A seed 1）：token 减少 ≥30%、acc 损失 ≤5pp、收敛 L_median ≥ 2 × Kahn 下限中位（c=2.5）。
    kill 判定：收敛 L_median ≤ 1.5 × decision_lb_tokens["2.5"] 中位数 → kill=True（"压到底了没有空间"；矩阵直接停下报告，不进入 λ / G 重调——重调是"压不动"，两者互斥）。"""
    ev = [json.loads(l) for l in open(run_dir / "eval.jsonl")]
    e0 = next((e for e in ev if e["step"] == 0), None); e1 = ev[-1]
    if e0 is None:
        return dict(ok=False, reason="no step-0 eval (arm 0 anchor missing)")
    red = 1 - e1["L_mean"] / max(e0["L_mean"], 1); loss = e0["acc"] - e1["acc"]
    above_oracle = (e1["L_median"] >= oracle_mult * oracle_lb) if oracle_lb is not None else None
    kill = (e1["L_median"] <= kill_mult * kill_lb) if kill_lb is not None else False
    return dict(ok=(red >= min_reduction and loss <= max_acc_loss and above_oracle is not False and not kill), kill=kill, token_reduction=round(red, 3), acc_loss=round(loss, 3),
                kahn_lb_c2_5=oracle_lb, decision_lb_c2_5=kill_lb, L_median_final=e1["L_median"], above_2x_kahn=above_oracle,
                arm0=dict(L=e0["L_mean"], acc=e0["acc"]), final=dict(step=e1["step"], L=e1["L_mean"], acc=e1["acc"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True); ap.add_argument("--config"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int); ap.add_argument("--post-cmd", default=""); ap.add_argument("--diag-n", type=int, default=500)
    ap.add_argument("--skip-gate", action="store_true"); ap.add_argument("--key", help="覆盖矩阵所有 run 的格（准入通过的格）")
    args = ap.parse_args()
    m = yaml.safe_load(open(args.matrix)); d = m.get("defaults", {}); runs = prune(m["runs"], args.max_runs)
    if args.key:
        for r in runs: r["key"] = args.key
    assert runs[0]["arm"] == "A" and int(runs[0]["seed"]) == 1, "matrix 第一条必须是 A seed 1（操纵检查）"
    log = ROOT / "results/matrix_log.md"
    inherited = []                                                   # 门重调后继承的设置（λ / G）
    idx = -1
    queue = list(runs)
    while queue:
        r = queue.pop(0); idx += 1
        cfg = args.config or r.get("config") or d.get("config", "configs/cloud_4b.yaml")
        if r["arm"] == "Bwarm_sft":
            name = run_name(r); rd = ROOT / "runs" / name
            if (rd / "final").exists():
                print(f"[skip] {name}: final exists"); continue
            src = (f"A_{float(d.get('lam', 0.5))}_{r['key']}_s{int(r['seed'])}" if r.get("source") == "A_seed" else best_A_run(r["key"], float(d.get("lam", 0.5))))
            if not (ROOT / "runs" / src / "final").exists():
                print(f"[skip] {name}: source A run {src} not finished"); continue
            sft = [sys.executable, "scripts/11_sft_warm.py", "--config", cfg, "--src", f"runs/{src}", "--tag", f"{r['key']}_s{int(r['seed'])}{r.get('tag', '')}"]
            ev = [sys.executable, "scripts/train.py", "--config", cfg, "--arm", "Bwarm", "--eval-only", "--tag", f"_{r['key']}_s{int(r['seed'])}_sft",
                  "--set", f"task.key={r['key']}", f"train.init_from=runs/{name}/final", *inherited]
            print(f"[run {idx+1}] {name}\n  {' '.join(sft)}\n  then {' '.join(ev)}", flush=True)
            if not args.dry_run:
                rc = subprocess.call(sft, cwd=ROOT); rc2 = subprocess.call(ev, cwd=ROOT) if rc == 0 else None
                ev_run = f"evalonly_Bwarm_{r['key']}_s{int(r['seed'])}_sft"
                diag = [sys.executable, "scripts/06_diagnose.py", "--run", f"runs/{ev_run}", "--config", cfg, "--n", str(args.diag_n), "--key", r["key"], "--backend", "vllm"]
                rc3 = subprocess.call(diag, cwd=ROOT) if rc2 == 0 else None
                with open(log, "a") as f: f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} {name}: sft_rc={rc} evalonly_rc={rc2} diag_rc={rc3}\n")
                if rc != 0: print(f"[fail] {name}"); return
            continue
        if r.get("conditional") == "cold_start_failure" and not args.dry_run:
            fail, best = cold_start_failure(r["key"], float(r["lam"]))
            with open(log, "a") as f: f.write(f"- cold-start check for {run_name(r)}: failure={fail} best_externalized={json.dumps(best)}\n")
            if not fail:
                print(f"[skip] {run_name(r)}: conditional cold_start_failure not met ({best})"); continue
            n_left = sum(1 for q in [r] + queue if q.get("conditional") == "cold_start_failure" and not (ROOT / "runs" / run_name(q) / "final").exists())
            bp = budget_projection(cfg, n_left)                          # r3：B_warm-RL 前检查预算投影
            with open(log, "a") as f: f.write(f"- budget projection before {run_name(r)}: {json.dumps(bp)}\n")
            print(f"[budget] {json.dumps(bp)}", flush=True)
            if bp["ok"] is False:
                print(f"[skip] {run_name(r)}: projected spend exceeds cost.budget_usd — B_warm-RL not run", flush=True); continue
        name = run_name(r); rd = ROOT / "runs" / name
        tag = f"_{r['key']}_s{int(r['seed'])}{r.get('tag', '')}"
        sets = [f"reward.lambda={float(r['lam'])}", f"task.key={r['key']}", f"train.seed={int(r['seed'])}",
                f"train.max_steps={int(r.get('steps', d.get('steps', 400)))}"] + list(d.get("set", [])) + list(r.get("set", [])) + list(inherited)
        if not args.dry_run:
            sets = resolve_auto(sets, r, d)
        cmd = [sys.executable, "scripts/train.py", "--config", cfg, "--arm", r["arm"], "--tag", tag, "--set", *sets]
        if (rd / "final").exists():
            print(f"[skip] {name}: final exists"); 
        else:
            if r.get("speed_test") and not args.dry_run and not (ROOT / f"results/dry_run_{r['arm']}{tag}.json").exists():
                dr = [sys.executable, "scripts/train.py", "--config", cfg, "--arm", r["arm"], "--dry-run", "--tag", tag, "--set", *sets]
                print(f"[speed-test 20 steps] {' '.join(dr)}", flush=True); rc0 = subprocess.call(dr, cwd=ROOT)
                with open(log, "a") as f: f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} speed test {name}: rc={rc0}\n")
            if any(rd.glob("checkpoint-*")):
                cmd += ["--resume", "--no-eval-at-start"]
            diag = [sys.executable, "scripts/06_diagnose.py", "--run", f"runs/{name}", "--config", cfg, "--n", str(args.diag_n), "--key", r["key"], "--backend", "vllm"]
            print(f"[run {idx+1}/{len(runs)}] {name}\n  {' '.join(cmd)}\n  then {' '.join(diag)}", flush=True)
            if not args.dry_run:
                try:                                   # 持久盘断过一次：每个 run 启动前探测，不可写就停
                    for d in (ROOT / "runs", ROOT / "samples", ROOT / "results"):
                        assert_writable(d, f"before run {name}")
                except WorkspaceNotWritable as e:
                    print(f"[FATAL] {e} — matrix stopped", flush=True); sys.exit(3)
                t0 = time.time(); rc = subprocess.call(cmd, cwd=ROOT)
                rc2 = subprocess.call(diag, cwd=ROOT) if rc == 0 else None
                with open(log, "a") as f:
                    f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} {name}: train_rc={rc} diag_rc={rc2} {(time.time()-t0)/3600:.1f} h\n")
                if args.post_cmd:
                    subprocess.call(args.post_cmd.replace("{run}", name), shell=True, cwd=ROOT)
                if rc != 0:
                    print(f"[fail] {name} rc={rc}; stopping matrix (re-run to resume)", flush=True); return
                if r.get("warm_check"):
                    wd = warm_decision(r["key"], float(r["lam"])); print(f"[warm decision] {json.dumps(wd)}", flush=True)
                    with open(log, "a") as f: f.write(f"- warm decision after {name}: {json.dumps(wd)}\n")
        if r.get("_gate", idx == 0) and not args.skip_gate and not args.dry_run:
            g = first_run_gate(rd, oracle_lb=oracle_lb_median(r["key"], "2.5", "kahn"), kill_lb=oracle_lb_median(r["key"], "2.5", "decision"))
            g["kahn_lb_c2"] = oracle_lb_median(r["key"], "2", "kahn"); g["kahn_lb_c3"] = oracle_lb_median(r["key"], "3", "kahn")
            g["decision_lb_c2"] = oracle_lb_median(r["key"], "2", "decision"); g["decision_lb_c3"] = oracle_lb_median(r["key"], "3", "decision")
            with open(log, "a") as f:
                f.write(f"- first-run gate {name}: {json.dumps(g)}\n")
            print(f"[gate] {json.dumps(g)}", flush=True)
            if g.get("kill"):
                print("[gate] KILL: converged L_median <= 1.5 × median decision_lb_tokens[2.5] — matrix stopped for review (no retune)", flush=True); return
            if not g["ok"]:
                k = r.get("_retune", 0)
                if k >= len(RETUNE):
                    print("[gate] FAILED after λ=0.3 → λ=1.0 → G=32 — matrix stopped for review", flush=True); return
                rt = RETUNE[k]; nxt = dict(runs[0], tag=rt["tag"], set=list(runs[0].get("set", [])) + rt["set"], _gate=True, _retune=k + 1)
                if rt["lam"] is not None: nxt["lam"] = rt["lam"]
                print(f"[gate] FAILED → retune {rt['tag']}", flush=True); queue.insert(0, nxt); continue
            if r.get("_retune"):
                rt = RETUNE[r["_retune"] - 1]; inherited = list(rt["set"])
                if rt["lam"] is not None:
                    for q in queue:
                        if "lam" in q: q["lam"] = rt["lam"]
                with open(log, "a") as f: f.write(f"- gate passed with retune {rt['tag']}; all later runs inherit {inherited}\n")

if __name__ == "__main__":
    main()
