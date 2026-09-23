"""第二阶段自动链（pod 上 setsid nohup 启动；用户说"开始"之后）。不在 OSF 注册的 stage-1 协议之内。
执行顺序写在 ops/state.md（用户 2026-09-23 定）。第 1 步"把 A1 的 final/ 与 checkpoint-350 拉回本地并校验 sha256"由本地会话做，
完成后才启动本脚本。本脚本做第 2–5 步：
  smoke    闸门 1 冒烟：训练 gate1_N2_s1（全量 250 步，之后全量直接复用）→ 评估 --limit 20（写到 runs/_gate1_smoke/）
           → 检查输出格式 / 解析 / 预算强制 → 按实测速度推算闸门 1 全量费用；有问题或推算 > $15 → 停下汇报
  full     其余 5 个 run 训练 + 6 个 run 全量评估；每个 run 结束复算"已花 + 剩余"，> $15 → 停下汇报
  analyze  判读（scripts/phase2_gate1_analyze.py）+ 报告 results/phase2_gate1_report.md。判读结果（含 P 不成立 / R4"说不清，停"）
           是闸门 1 的结论，不是链的故障：链照常进入 A1 延续（两者无依赖；用户定的顺序是"接着自动跑"）
  a1cont   探索性 A1 延续（不属于闸门 1 判读分区）：λ=1.0，固定 150 步，每 50 步 eval，费用守卫 $35（train.py BudgetGuard，D26）
           → 报告 ops/analysis/a1_cont_report.py（含 CoT 抽样）
  stop     全部完成 → pod_stop --delay 300 --force
停下汇报 = 写 results/phase2_halt.json + /workspace/logs/DECISION_PENDING（pod_watchdog 90 min 后停机）+ cloud_log 一行，退出码 4；
不自动重试、不自动降配。
费用（墙钟 × $1.6/h）：闸门 1 从 pod 开机算起（/proc/uptime）到报告写完；A1 延续从本阶段开始算起，两者分开计。
用法（pod 上）：setsid nohup .venv/bin/python scripts/phase2_chain.py > /workspace/logs/phase2_chain.log 2>&1 &
本地演练：HF_HOME=~/hf COT_NO_UNSLOTH=1 .venv/bin/python scripts/phase2_chain.py --dry-run --root <临时目录>
  （训练走 phase2_gate1_train --dry-run、评估走 --stub mixed、A1 延续只打印命令并用 A1 自己的归档演练报告脚本；不写 DECISION_PENDING、不停机）"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import argparse, json, os, shutil, subprocess, time
import yaml
from cot_compress.cost import gate1_projection

RATE = 1.6
GATE1_CAP = 15.0
A1CONT_CAP = 35.0
SMOKE_N = 20
A1_SRC = "A_0.5_ord_n8_h4_d5_s1"
A1CONT_TAG = "_cont_from_A1s350"                  # run 目录 runs/A_1.0_cont_from_A1s350（新目录；A1 原目录只读）
A1CONT_RUN = f"A_1.0{A1CONT_TAG}"
A1_BASE_SET = ["task.key=ord_n8_h4_d5", "train.seed=1"]          # 与 run_matrix 给 A1 的一致（λ、max_steps 另给）
A1CONT_SET = ["reward.lambda=1.0", "train.convergence_stop=false", f"train.budget_usd={A1CONT_CAP}", f"train.budget_rate={RATE}"]

PY = sys.executable
DRY = False
LOG = None


def now(): return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())


def say(msg):
    line = f"[{now()}] {msg}"; print(line, flush=True)
    if LOG: open(LOG, "a").write(line + "\n")


def cloud_log(msg):
    if DRY: return
    with open(ROOT / "results/cloud_log.md", "a") as f: f.write(f"- {now()} phase2_chain: {msg}\n")


def halt(stage, reason, **info):
    rec = dict(time=now(), stage=stage, reason=reason, **info)
    out = (ROOT / "results/phase2_halt.json") if not DRY else pathlib.Path(LOG).with_name("phase2_halt.json")
    json.dump(rec, open(out, "w"), indent=1, ensure_ascii=False)
    say(f"HALT [{stage}] {reason}")
    if not DRY:
        cloud_log(f"HALT [{stage}] {reason} → results/phase2_halt.json；DECISION_PENDING 已写（watchdog 90 min 后停机）")
        pathlib.Path("/workspace/logs").mkdir(parents=True, exist_ok=True)
        open("/workspace/logs/DECISION_PENDING", "w").write(f"{now()} phase2_chain HALT [{stage}]: {reason}\n{json.dumps(info, ensure_ascii=False)[:2000]}\n")
    sys.exit(4)


def run(cmd, env=None, log_to=None):
    say("$ " + " ".join(map(str, cmd)))
    t = time.time()
    with open(log_to, "a") if log_to else open(os.devnull, "w") as lf:
        rc = subprocess.call(list(map(str, cmd)), cwd=ROOT, env=dict(os.environ, **(env or {})),
                             stdout=(lf if log_to else None), stderr=(subprocess.STDOUT if log_to else None))
    return rc, time.time() - t


def tree_sha256(d):
    """A1 原目录只读保护（用户 2026-09-23）：目录下每个文件的 sha256（相对路径 → 值），含子目录（final/、checkpoint-*、诊断文件）。"""
    import hashlib
    out = {}
    for f in sorted(pathlib.Path(d).rglob("*")):
        if f.is_file():
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
            out[str(f.relative_to(d))] = h.hexdigest()
    return out


def check_readonly(src, before, where, logdir):
    after = tree_sha256(src)
    changed = sorted(k for k in before if k in after and after[k] != before[k])
    removed = sorted(k for k in before if k not in after); added = sorted(k for k in after if k not in before)
    json.dump(dict(time=now(), where=where, n_files=len(after), changed=changed, removed=removed, added=added),
              open(logdir / f"a1_readonly_check_{where}.json", "w"), indent=1)
    if changed or removed or added:
        halt("a1-readonly", f"A1 原目录 {src.name} 在 {where} 与启动前的 sha256 不一致（改 {len(changed)} / 删 {len(removed)} / 增 {len(added)}）",
             changed=changed[:20], removed=removed[:20], added=added[:20])
    say(f"A1 原目录只读检查 @ {where}: {len(after)} 个文件 sha256 与启动前一致")


def boot_time():
    try: return time.time() - float(open("/proc/uptime").read().split()[0])
    except Exception: return time.time()


class Ledger:
    def __init__(self, t0, cap, name): self.t0, self.cap, self.name = t0, cap, name
    def spent_h(self): return (time.time() - self.t0) / 3600


def eval_est_sec(timing, n_native_masks, scale):
    """冒烟（SMOKE_N 题）→ 每 run 全量评估墙钟的线性外推（上界：小批次时 vLLM 并发低）。载入时间不随题数变。"""
    per_native = [v for k, v in timing.items() if k.startswith("native_")]
    native = (sum(per_native) / len(per_native)) if per_native else 0.0
    rest = timing.get("direct_sec", 0) + timing.get("transplant_sec", 0) + timing.get("tamper_sec", 0)
    return timing.get("load_sec", 0) + scale * (native * n_native_masks + rest)


def gate1_runs(g): return [(N, s) for N in g["notations"] for s in g["seeds"]]


def n_masks(g, N):
    m = g["eval"]["mask"][N]; return len(m) if isinstance(m, list) else 1


def train_run(N, s, root, logdir):
    fin = root / f"gate1_{N}_s{s}" / "final"
    if not DRY and fin.exists():
        say(f"train gate1_{N}_s{s}: final/ exists, skip"); return 0.0
    cmd = [PY, "scripts/phase2_gate1_train.py", "--notation", N, "--seed", s] + (["--dry-run"] if DRY else [])
    rc, sec = run(cmd, log_to=logdir / f"train_{N}_s{s}.log")
    if rc != 0: halt("gate1-train", f"gate1_{N}_s{s} 训练退出码 {rc}", log=str(logdir / f"train_{N}_s{s}.log"))
    return sec


def eval_run(N, s, root, logdir, limit=None, out_root=None):
    cmd = [PY, "scripts/phase2_gate1_eval.py", "--notation", N, "--seed", s, "--out-root", out_root or root]
    if limit: cmd += ["--limit", limit]
    if DRY: cmd += ["--stub", "mixed"]
    tag = f"eval_{N}_s{s}" + (f"_limit{limit}" if limit else "")
    rc, sec = run(cmd, log_to=logdir / f"{tag}.log")
    if rc != 0: halt("gate1-eval", f"gate1_{N}_s{s} 评估退出码 {rc}" + (f"（--limit {limit}）" if limit else ""), log=str(logdir / f"{tag}.log"))
    return json.load(open(pathlib.Path(out_root or root) / f"gate1_{N}_s{s}" / "gate1_eval.json")), sec


def smoke_checks(summ, g, N, cap):
    """冒烟检查（运维层面，不是判读）：每项失败都停下汇报。"""
    from cot_compress.archive import read_jsonl_zst
    problems = []
    need = [f"native_{m}" for m in (g["eval"]["mask"][N] if isinstance(g["eval"]["mask"][N], list) else [g["eval"]["mask"][N]])] + ["direct", "transplant_gold", "transplant_own"]
    for k in need:
        if k not in summ or summ[k].get("n") != summ["n_items"]: problems.append(f"条件 {k} 缺失或题数不对")
    if "tamper" not in summ: problems.append("篡改组缺失（前 20 题里没有可篡改的题？）")
    if summ.get("max_think_tokens", 0) > cap: problems.append(f"think 段 {summ['max_think_tokens']} token > cap {cap}（预算强制失效）")
    rows = list(read_jsonl_zst(pathlib.Path(summ["_dir"]) / "gate1_rows.jsonl.zst"))
    nat = [r for r in rows if r["cond"].startswith("native_")]
    no_think = sum("</think>" not in r["completion"] for r in nat)
    if no_think: problems.append(f"原生输出中 {no_think}/{len(nat)} 条没有 </think>（输出格式）")
    no_pred = sum(r["pred"] is None for r in nat)
    if no_pred > 0.2 * len(nat): problems.append(f"原生输出中 {no_pred}/{len(nat)} 条抽不出答案（> 20%）")
    parsed = sum(r["parsed"] for r in nat)
    if parsed < 0.5 * len(nat): problems.append(f"推导可解析 {parsed}/{len(nat)}（< 50%：解析器与生成格式不符？）")
    forced_ok = all(r["think_tokens"] <= cap for r in nat)
    if not forced_ok: problems.append("存在 think_tokens > cap 的行")
    samples = [dict(cond=r["cond"], id=r["id"], correct=r["correct"], parsed=r["parsed"], think_tokens=r["think_tokens"], completion=r["completion"][:600]) for r in nat[:3]]
    return problems, dict(n_native_rows=len(nat), no_think=no_think, no_pred=no_pred, parsed=parsed, max_think_tokens=summ.get("max_think_tokens")), samples


def stage_gate1(root, logdir, g):
    led = Ledger(time.time() if DRY else boot_time(), GATE1_CAP, "gate1")
    runs = gate1_runs(g); cap = int(g["eval"]["cap"])
    budget_log = logdir / "gate1_budget.jsonl"
    def check(where, p):
        open(budget_log, "a").write(json.dumps(dict(time=now(), where=where, **p)) + "\n")
        say(f"[budget gate1 @ {where}] spent ${p['spent_usd']} projected ${p['projected_usd']} (cap ${GATE1_CAP})")
        if p["projected_usd"] > GATE1_CAP: halt("gate1-budget", f"闸门 1 费用推算 ${p['projected_usd']} > ${GATE1_CAP}（{where}）", projection=p)
    # ---- 冒烟 ----
    N0, s0 = runs[0]
    say(f"== gate1 smoke: train gate1_{N0}_s{s0} (full) + eval --limit {SMOKE_N}")
    train_sec = train_run(N0, s0, root, logdir)
    smoke_root = root / "_gate1_smoke"
    summ, _ = eval_run(N0, s0, root, logdir, limit=SMOKE_N, out_root=smoke_root)
    summ["_dir"] = str(smoke_root / f"gate1_{N0}_s{s0}")
    problems, facts, samples = smoke_checks(summ, g, N0, cap)
    scale = 500 / SMOKE_N
    est = {N: eval_est_sec(summ["timing"], n_masks(g, N), scale) for N in g["notations"]}
    rep = dict(time=now(), run=f"gate1_{N0}_s{s0}", train_sec=round(train_sec, 1), smoke_timing=summ["timing"], facts=facts,
               eval_full_est_sec_linear_upper={N: round(v, 1) for N, v in est.items()}, problems=problems, samples=samples,
               native=summ.get(f"native_{g['eval']['mask'][N0][0] if isinstance(g['eval']['mask'][N0], list) else g['eval']['mask'][N0]}"))
    json.dump(rep, open(logdir / "gate1_smoke.json", "w"), indent=1, ensure_ascii=False)
    say(f"smoke: {json.dumps({k: rep[k] for k in ('train_sec', 'facts', 'eval_full_est_sec_linear_upper', 'problems')}, ensure_ascii=False)}")
    cloud_log(f"闸门 1 冒烟：训练 {train_sec/60:.1f} min；评估 --limit {SMOKE_N} {summ['timing']['total_sec']:.0f} s；检查 {'通过' if not problems else '失败：' + '；'.join(problems)}")
    if problems: halt("gate1-smoke", "冒烟检查失败：" + "；".join(problems), smoke=str(logdir / "gate1_smoke.json"))
    tr_sec = train_sec if train_sec > 0 else 1200.0
    check("after smoke", gate1_projection(led.spent_h(), RATE, len(runs) - 1, tr_sec, len(runs), sum(est[N] for N, _ in runs) / len(runs), tail_h=0.1))
    # ---- 全量 ----
    train_secs, eval_secs = ([train_sec] if train_sec > 0 else []), []
    for i, (N, s) in enumerate(runs):
        say(f"== gate1 full: gate1_{N}_s{s}")
        if i > 0:
            t = train_run(N, s, root, logdir)
            if t > 0: train_secs.append(t)
        _, esec = eval_run(N, s, root, logdir); eval_secs.append(esec)
        left_train = len(runs) - 1 - i; left_eval = len(runs) - 1 - i
        mt = sum(train_secs) / len(train_secs) if train_secs else tr_sec
        me = max(eval_secs) if eval_secs else 0
        check(f"after gate1_{N}_s{s}", gate1_projection(led.spent_h(), RATE, left_train, mt, left_eval, me, tail_h=0.05))
    # ---- 判读 + 报告 ----
    say("== gate1 analyze")
    verdict = (root / "phase2_gate1_verdict.json") if DRY else (ROOT / "results/phase2_gate1_verdict.json")
    rc, _ = run([PY, "scripts/phase2_gate1_analyze.py", "--root", root, "--out", verdict], log_to=logdir / "gate1_analyze.log")
    if rc != 0: halt("gate1-analyze", f"判读脚本退出码 {rc}", log=str(logdir / "gate1_analyze.log"))
    write_gate1_report(verdict, logdir, led, train_secs, eval_secs)
    return led


def write_gate1_report(verdict_path, logdir, led, train_secs, eval_secs):
    v = json.load(open(verdict_path)); out = pathlib.Path(verdict_path).with_name("phase2_gate1_report.md")
    L = [f"# 第二阶段 · 闸门 1 报告（自动生成 {now()}）", "",
         f"**读法：{v['reading']}** —— {v['text']}", ""]
    for n_ in v.get("notes", []): L.append(f"- 注：{n_}")
    L += ["", "## 前提 P", "", "| 量 | 值 |", "|---|---|"] + [f"| {k} | {x:.3f} |" for k, x in v["premise"].items()]
    L += ["", "检查：" + json.dumps(v["premise_checks"], ensure_ascii=False), "",
          "## 比较（每题两 seed 取平均，按题 bootstrap 10000 次）", "",
          "| 比较 | 题数 | A | B | B−A | 90% 区间 | 95% 区间 | seed 1 | seed 2 | 等效 | 显著更差 | 显著更好 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in v["comparisons"].items():
        ps = c["per_seed_diff"]; sk = sorted(ps, key=lambda z: int(z))
        L.append(f"| {k} | {c['n_items']} | {c['acc_a']:.3f} | {c['acc_b']:.3f} | {c['diff']:+.3f} | [{c['ci90'][0]:+.3f}, {c['ci90'][1]:+.3f}] | "
                 f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {ps[sk[0]]:+.3f} | {ps[sk[-1]]:+.3f} | {c['equivalent']} | {c['sig_worse']} | {c['sig_better']} |")
    if v.get("subreadings"): L += ["", "R2 子读法：" + json.dumps(v["subreadings"], ensure_ascii=False)]
    L += ["", "## 各条件（两 seed 合并；描述量）", "", "| 条件 | 数 |", "|---|---|"]
    L += [f"| {k} | {json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in t.items()}, ensure_ascii=False)} |" for k, t in v["per_condition_pooled"].items()]
    L += ["", f"参照（不进判读）：A1 收敛准确率 {v['reference_only']['A1_converged_acc_stop_split']}", "",
          f"费用：闸门 1 已花 ${led.spent_h() * RATE:.2f}（上限 ${GATE1_CAP}；墙钟自开机）；训练 {len(train_secs)} 次均 {(sum(train_secs) / max(len(train_secs), 1)) / 60:.1f} min；"
          f"评估 {len(eval_secs)} 次均 {(sum(eval_secs) / max(len(eval_secs), 1)) / 60:.1f} min", ""]
    out.write_text("\n".join(L)); say(f"gate1 report → {out}")
    cloud_log(f"闸门 1 完成：读法 {v['reading']}；已花 ${led.spent_h() * RATE:.2f}；报告 results/phase2_gate1_report.md")


def stage_a1cont(root, logdir):
    t0 = time.time(); led = Ledger(t0, A1CONT_CAP, "a1cont")
    src = root / A1_SRC; dst = root / A1CONT_RUN; ck = src / "checkpoint-350"
    have_opt = all((ck / f).exists() for f in ("optimizer.pt", "scheduler.pt", "trainer_state.json", "adapter_model.safetensors"))
    if have_opt:
        origin = dict(mode="resume_checkpoint_350", optimizer_state="restored (optimizer.pt / scheduler.pt / rng from checkpoint-350)",
                      global_step_start=350, max_steps=500, eval_steps=[400, 450, 500], note="数据：build_dataset 前缀与 A1 相同（同 seed 顺序取题），恢复后跳过已用批次 → 延续段用 A1 未见过的训练题")
        extra = ["train.max_steps=500"]; flags = ["--resume", "--no-eval-at-start"]
    else:
        origin = dict(mode="init_from_final_fresh_optimizer", optimizer_state="re-initialized (checkpoint-350 缺优化器状态；从 final/ adapter 起训，warmup 10 步重新走)",
                      global_step_start=0, max_steps=150, eval_steps=[50, 100, 150], init_from=f"runs/{A1_SRC}/final",
                      note="步号从 0 记；报告里换算为 A1 步 350 + k。build_dataset 取前 300 题 = A1 训练过的题")
        extra = ["train.max_steps=150", f"train.init_from=runs/{A1_SRC}/final", "train.init_from_any_arm=true"]; flags = ["--no-eval-at-start"]
        if not DRY and not (src / "final" / "adapter_model.safetensors").exists():
            halt("a1cont-setup", f"runs/{A1_SRC} 既没有完整的 checkpoint-350 也没有 final/ adapter")
    cmd = [PY, "scripts/train.py", "--config", "configs/cloud_4b.yaml", "--arm", "A", "--tag", A1CONT_TAG, *flags, "--set", *A1_BASE_SET, *A1CONT_SET, *extra]
    origin.update(time=now(), src=f"runs/{A1_SRC}", cmd=" ".join(map(str, cmd)), budget_cap_usd=A1CONT_CAP,
                  git=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    say(f"== A1 continuation: {origin['mode']}")
    assert dst.resolve() != src.resolve() and not str(dst.resolve()).startswith(str(src.resolve()) + "/"), "A1 延续必须在新目录里跑"
    if src.exists():
        before = tree_sha256(src)
        json.dump(before, open(logdir / "a1_src_sha256_before.json", "w"), indent=1)
        say(f"A1 原目录 sha256 快照：{len(before)} 个文件 → {logdir / 'a1_src_sha256_before.json'}")
    if DRY:
        say("dry-run: 不训练；命令 = " + origin["cmd"])
        rc, _ = run([PY, "ops/analysis/a1_cont_report.py", "--run", src, "--base", src, "--out", root / "a1_cont_report_DRYRUN.md", "--samples-out", root / "a1_cont_samples_DRYRUN", "--dry-run"],
                    log_to=logdir / "a1cont_report.log")
        if rc != 0: halt("a1cont-report", f"报告脚本（演练）退出码 {rc}", log=str(logdir / "a1cont_report.log"))
        check_readonly(src, before, "after_report", logdir)
        return led
    dst.mkdir(parents=True, exist_ok=True)
    if have_opt and not (dst / "checkpoint-350").exists():
        shutil.copytree(ck, dst / "checkpoint-350")
    json.dump(origin, open(dst / "continuation_origin.json", "w"), indent=1, ensure_ascii=False)
    cloud_log(f"A1 延续开始：{origin['mode']}；{origin['optimizer_state']}；上限 ${A1CONT_CAP}")
    rc, sec = run(cmd, env=dict(BUDGET_T0=str(t0)), log_to=logdir / "a1cont_train.log")
    check_readonly(src, before, "after_train", logdir)
    if rc == 5: halt("a1cont-budget", f"A1 延续费用投影 > ${A1CONT_CAP}（train.py BudgetGuard）", budget=str(dst / "budget_halt.json"))
    if rc != 0: halt("a1cont-train", f"A1 延续训练退出码 {rc}", log=str(logdir / "a1cont_train.log"))
    if (dst / "uninterpretable.json").exists(): halt("a1cont-residual", "A1 延续触发残留检查（uninterpretable.json）")
    rc, _ = run([PY, "ops/analysis/a1_cont_report.py", "--run", dst, "--base", src, "--out", ROOT / "results/a1_cont_report.md",
                 "--samples-out", ROOT / "ops/cot_samples/a1_cont"], log_to=logdir / "a1cont_report.log")
    if rc != 0: halt("a1cont-report", f"报告脚本退出码 {rc}", log=str(logdir / "a1cont_report.log"))
    check_readonly(src, before, "after_report", logdir)
    spent = led.spent_h() * RATE
    cloud_log(f"A1 延续完成：{sec / 3600:.2f} h，已花 ${spent:.2f}（上限 ${A1CONT_CAP}）；报告 results/a1_cont_report.md")
    if spent > A1CONT_CAP: halt("a1cont-budget", f"A1 延续实花 ${spent:.2f} > ${A1CONT_CAP}（事后）")
    return led


def main():
    global DRY, LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--root", default=str(ROOT / "runs"))
    ap.add_argument("--from", dest="start", choices=["gate1", "a1cont"], default="gate1")
    a = ap.parse_args(); DRY = a.dry_run
    root = pathlib.Path(a.root); logdir = (root / "_chain_logs") if DRY else pathlib.Path("/workspace/logs/phase2")
    logdir.mkdir(parents=True, exist_ok=True); LOG = str(logdir / "phase2_chain.log")
    if not DRY:
        for p in (ROOT / "results/phase2_halt.json", ROOT / "results/PHASE2_DONE"):
            if p.exists(): p.unlink()
    g = yaml.safe_load(open(ROOT / "configs/phase2_gate1.yaml"))
    assert float(g["budget"]["cap_usd"]) == GATE1_CAP, "configs/phase2_gate1.yaml budget.cap_usd 与脚本不一致"
    say(f"phase2 chain start (dry={DRY}, from={a.start}, root={root})"); cloud_log(f"启动（from={a.start}）")
    if a.start == "gate1": stage_gate1(root, logdir, g)
    stage_a1cont(root, logdir)
    say("== all done")
    if DRY: return
    open(ROOT / "results/PHASE2_DONE", "w").write(now() + "\n"); cloud_log("全部完成 → pod_stop --delay 300 --force")
    subprocess.call(["bash", "scripts/pod_stop.sh", "--delay", "300", "--force"], cwd=ROOT)


if __name__ == "__main__":
    main()
