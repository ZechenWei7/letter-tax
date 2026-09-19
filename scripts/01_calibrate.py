"""阶段 2：难度校准。每个难度键 × {think, guided, direct} 各 n 题，写 results/calib/*.jsonl，汇总到 results/calibration.csv。
判据（README §2）：某思考模式（think=原生 / guided=关原生思考+提示）准确率 60–85%、中位思考长度 ≤ 1500、
触顶比例 < 10%，且 direct 准确率 < 30%。每个 batch 的耗时 / 峰值显存 / 分段 steps-per-s 写到 results/calib/*.batches.json。

用法:
  python scripts/01_calibrate.py --grid coarse --n 40
  python scripts/01_calibrate.py --keys cups_n6_k12 mul_4x4 --n 100
  python scripts/01_calibrate.py --summarize-only

NOTE: 第一个 import 必须是 cot_compress（其 __init__ 无条件 import unsloth，须先于 trl/transformers）。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import cot_compress  # noqa: F401  unsloth first
import argparse, csv, json, time
import tasks
from cot_compress.config import load_config
from cot_compress.evaluate import MODES, run_eval, summarize, forced_answer_eval, prefilled_answer_eval

FORCE_CUTOFFS = [1536, 1024, 768]        # (a′) 预算 B 候选：从同一条原生轨迹派生，几乎免费
FORCE_MODES = [f"force{c}" for c in FORCE_CUTOFFS]
ALL_MODES = list(MODES) + FORCE_MODES

GRIDS = {
    "coarse": ["cups_n5_k8", "cups_n6_k12", "cups_n8_k16", "cups_n8_k24",
               "mul_3x3", "mul_4x3", "mul_4x4", "add_6x4"],
}
CRIT = dict(acc_lo=0.60, acc_hi=0.85, direct_max=0.30, think_median_max=1500, capped_max=0.10)
CALIB_DIR = pathlib.Path("results/calib")
CSV_PATH = pathlib.Path("results/calibration.csv")
COLS = ["key", "task", "mode", "n", "acc", "lenient_acc", "format_err", "truncated",
        "think_mean", "think_median", "think_p90", "completion_mean", "sec", "tok_per_s", "peak_gib", "ok"]
# ok 列：think/guided 行 = 该模式满足 acc∈[60,85] & 中位≤1500 & 触顶<10%，且同 key 的 direct<30；direct 行 = direct<30

def jsonl_path(key, mode):
    return CALIB_DIR / f"{key}__{mode}.jsonl"

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def mode_ok(s, mode, direct):
    if mode == "direct":
        return bool(s["acc"] < CRIT["direct_max"])
    if mode.startswith("force"):   # (a′)：思考长度 = B（按构造），只看准确率窗口 + direct
        return bool(direct and direct["acc"] < CRIT["direct_max"] and CRIT["acc_lo"] <= s["acc"] <= CRIT["acc_hi"])
    return bool(direct and direct["acc"] < CRIT["direct_max"]
                and CRIT["acc_lo"] <= s["acc"] <= CRIT["acc_hi"]
                and s["think_median"] <= CRIT["think_median_max"] and s["truncated"] < CRIT["capped_max"])

def summarize_all():
    """从所有 jsonl 重建 calibration.csv。"""
    stats = {}
    for p in sorted(CALIB_DIR.glob("*.jsonl")):
        key, mode = p.stem.rsplit("__", 1)
        rows = load_rows(p)
        if rows:
            stats[(key, mode)] = summarize(rows)
    out = []
    for key in sorted({k for k, _ in stats}):
        di = stats.get((key, "direct"))
        for mode in ALL_MODES:
            s = stats.get((key, mode))
            if not s:
                continue
            row = dict(key=key, task=tasks.task_for_key(key).__name__.split(".")[-1], mode=mode,
                       ok=mode_ok(s, mode, di))
            row.update({c: s[c] for c in COLS if c in s})
            for c in ("acc", "lenient_acc", "format_err", "truncated"):
                row[c] = round(row[c], 3)
            out.append(row)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    return out

def print_table(rows):
    hdr = f"{'key':14} {'mode':7} {'n':>4} {'acc':>6} {'len':>6} {'fmt':>6} {'trunc':>6} {'thk_med':>8} {'thk_p90':>8} {'sec':>6} {'tok/s':>6} {'peak':>5} ok"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['key']:14} {r['mode']:7} {r['n']:>4} {r['acc']:>6.2f} {r['lenient_acc']:>6.2f} {r['format_err']:>6.2f} "
              f"{r['truncated']:>6.2f} {r['think_median']:>8.0f} {r['think_p90']:>8.0f} {r['sec']:>6.0f} {r['tok_per_s']:>6.0f} "
              f"{r['peak_gib']:>5.2f} {'*' if r['ok'] else ''}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_1p7b.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--grid", choices=list(GRIDS))
    ap.add_argument("--keys", nargs="*", default=[])
    ap.add_argument("--modes", nargs="*", default=["think", "direct"], help="think 会自动派生 force{B} 格")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="已有同 key/mode 且 n>= 本次时也重跑")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--admission", action="store_true", help="v7 准入：对 --keys 各格在冻结模型上跑 8 项检查（stopping-eval 500 题），写 results/admission_<key>.json")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="准入生成后端；vllm 比 HF generate 快约 10×（n≥500 × 多格时必需）")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--direct-only", action="store_true", help="D6：只重测直接作答（direct-check 2000 + stop 500 + 逐对诊断），写回已有的 results/admission_<key>.json 并重算判定；其余字段不动")
    ap.add_argument("--native-only", action="store_true", help="只重测 native 准确率 / M / 强制收尾率（如换 cap 后复核），写 results/retest_<key>_cap<cap>.json")
    args = ap.parse_args()

    if args.summarize_only:
        print_table(summarize_all()); return
    if args.admission:
        run_admission(args); return

    keys = list(args.keys) + (GRIDS[args.grid] if args.grid else [])
    assert keys, "give --grid or --keys"
    cfg = load_config(args.config, args.set)
    gen_cfg = cfg["generation"]

    from cot_compress.model import load_model
    model, tok = load_model(cfg)
    from cot_compress.memory import tune_allocator; print(tune_allocator(), flush=True)

    for key in keys:
        items = tasks.make_eval_set(key, args.n, seed=args.seed)
        for mode in args.modes:
            path = jsonl_path(key, mode)
            if path.exists() and not args.force and len(load_rows(path)) >= args.n:
                print(f"[skip] {key} {mode}: existing n={len(load_rows(path))} >= {args.n}", flush=True)
                continue
            print(f"\n[run] {key} {mode} n={args.n}", flush=True)
            t0 = time.time()
            bstats = []
            rows = run_eval(model, tok, key, items, mode, gen_cfg, seed=args.seed, batch_stats=bstats,
                            progress=lambda d, t: print(f"  {d}/{t} ({time.time()-t0:.0f}s) last batch: "
                                                        f"{bstats[-1]['sec']}s max_new={bstats[-1]['max_new']} "
                                                        f"peak={bstats[-1]['peak_gib']}GiB reserved={bstats[-1]['reserved_gib']}GiB "
                                                        f"steps/s={[w['steps_per_s'] for w in bstats[-1]['steps_per_s_by_256']]}", flush=True))
            write_rows(path, rows)
            with open(path.with_suffix(".batches.json"), "w") as f:
                json.dump(bstats, f, indent=1)
            s = summarize(rows)
            print(f"  acc={s['acc']:.2f} lenient={s['lenient_acc']:.2f} fmt_err={s['format_err']:.2f} "
                  f"trunc={s['truncated']:.2f} think_med={s['think_median']} p90={s['think_p90']} "
                  f"sec={s['sec']} tok/s={s['tok_per_s']} peak={s['peak_gib']}GiB", flush=True)
            bad = [r for r in rows if not r["format_ok"]][:2]
            for r in bad:
                print(f"  fmt-fail example ({r['reason']}): ...{r['completion'][-200:]!r}", flush=True)
        # (a′) 派生格：think 轨迹（本次或已有）在各 B 处强制作答；只补缺失的 B
        tp = jsonl_path(key, "think")
        if "think" in args.modes and tp.exists():
            trows = load_rows(tp)
            for c in FORCE_CUTOFFS:
                fp = jsonl_path(key, f"force{c}")
                if fp.exists() and not args.force and len(load_rows(fp)) >= len(trows):
                    continue
                fr = forced_answer_eval(model, tok, trows, c, gen_cfg, seed=args.seed)
                write_rows(fp, fr)
                fs = summarize(fr)
                print(f"  [force{c}] {key} acc={fs['acc']:.2f} fmt_err={fs['format_err']:.2f} "
                      f"forced={sum(r.get('forced', False) for r in fr)}/{len(fr)}", flush=True)
    print("\n==== calibration.csv ====")
    print_table(summarize_all())



# =============================================================================================
# v4 准入（README §2.3）：direct / native / 强制预算曲线 / 白名单填充 / 轨迹移植 / letter-free 冻结 / D 臂（pending）
# =============================================================================================
def run_admission(args):
    import random, statistics
    from cot_compress.admission import admission_decision, FRACS
    from cot_compress.vocab_mask import banned_ids, think_allowed_ids, lift_id
    from cot_compress.traj_stats import TrajStats
    from cot_compress.span import HOOK
    cfg = load_config(args.config, args.set); gen_cfg = dict(cfg["generation"])
    cap = int(cfg["train"]["max_completion_length"]); reserve = int(cfg["train"].get("force_reserve", 24))
    gen_cfg["max_new_tokens_think"] = cap
    if args.backend == "vllm":
        import cot_compress.evaluate as _ev
        from vllm import LLM
        from cot_compress.rollout import engine_kwargs
        llm = LLM(model=cfg["model"]["name"], dtype="bfloat16", gpu_memory_utilization=args.gpu_util,
                  max_model_len=int(cfg["model"]["max_seq_length"]), **engine_kwargs())
        _ev.VLLM["llm"] = llm; model = None; tok = llm.get_tokenizer()
        V = int(llm.llm_engine.model_config.get_vocab_size())
        HOOK.configure(tok, reserve=reserve, banned_ids=None)
    else:
        from cot_compress.model import load_model
        model, tok = load_model(cfg)
        from cot_compress.memory import tune_allocator; print(tune_allocator(), flush=True)
        HOOK.configure(tok, reserve=reserve, banned_ids=None).install(model)
        V = int(model.get_input_embeddings().weight.shape[0])
    lid = lift_id(tok)
    lf_banned = banned_ids(tok, "letterfree", vocab_size=V); lf_allowed = think_allowed_ids(tok, "letterfree", vocab_size=V)
    n = max(args.n, 500) if not args.force else args.n
    for key in args.keys:
        task = tasks.task_for_key(key); items = tasks.make_eval_set(key, n, seed=args.seed, **({"split": "stop"} if key.startswith(("kk_", "ord_")) else {}))
        chance = task.chance(key) if hasattr(task, "chance") else 0.0
        print(f"\n[admission] {key} n={n} chance={chance:.3f}", flush=True)
        if args.direct_only:
            out = pathlib.Path(f"results/admission_{key}.json"); metrics = json.load(open(out))
            d_items = tasks.make_eval_set(key, 2000, split="direct")
            d_rows = run_eval(model, tok, key, d_items, "direct", gen_cfg, seed=args.seed); ds = summarize(d_rows)
            d500 = summarize(run_eval(model, tok, key, items, "direct", gen_cfg, seed=args.seed))["acc"]
            pw = [task.pairwise_accuracy([int(x) for x in r["pred"].split()], it["meta"]["sigma"]) for r, it in zip(d_rows, d_items) if r.get("pred")]
            pw = [x for x in pw if x is not None]
            metrics["direct_before_D6_fix"] = dict(direct=metrics.get("direct"), direct_lenient=metrics.get("direct_lenient"), pairwise_direct=metrics.get("pairwise_direct"), note="max_new_tokens_direct=16 truncated every answer")
            metrics.update(direct=ds["acc"], direct_lenient=ds["lenient_acc"], direct_n=len(d_rows), direct_stop500=d500, direct_format_err=ds["format_err"],
                           pairwise_direct=(statistics.mean(pw) if pw else None), pairwise_direct_n=len(pw), max_new_tokens_direct=int(gen_cfg["max_new_tokens_direct"]))
            metrics["decision"] = admission_decision(metrics)
            write_rows(CALIB_DIR / f"admission_{key}__direct2000.jsonl", d_rows); json.dump(metrics, open(out, "w"), indent=1)
            print(f"  [direct-only] {key} direct={ds['acc']:.4f} lenient={ds['lenient_acc']:.4f} fmt_err={ds['format_err']:.3f} stop500={d500:.4f} pairwise={metrics['pairwise_direct']} (n={len(pw)}) admitted={metrics['decision']['admitted']}", flush=True)
            continue
        if args.native_only:
            direct_acc = summarize(run_eval(model, tok, key, items, "direct", gen_cfg, seed=args.seed))["acc"]      # 几秒钟，顺带记
            HOOK.enabled = True
            nat = run_eval(model, tok, key, items, "think", gen_cfg, seed=args.seed); ns = summarize(nat)
            Ls = sorted(r["think_tokens"] for r in nat); q = lambda f: Ls[min(len(Ls) - 1, int(f * len(Ls)))]
            rec = dict(key=key, n=n, cap=cap, direct=direct_acc, native=ns["acc"], native_clean=ns["acc_clean"], viol_rate=ns["viol_rate"], forced_rate=ns["forced_rate"],
                       M=float(statistics.median(Ls)), L_mean=round(statistics.mean(Ls), 1), L_p10=q(0.1), L_p25=q(0.25), L_p75=q(0.75), L_p90=q(0.9),
                       acc_forced=(statistics.mean(r["correct"] for r in nat if r["forced"]) if any(r["forced"] for r in nat) else None),
                       acc_natural=(statistics.mean(r["correct"] for r in nat if not r["forced"]) if any(not r["forced"] for r in nat) else None),
                       chance=chance, in_window=bool(0.60 <= ns["acc"] <= 0.80), forced_lt_20=bool(ns["forced_rate"] < 0.20))
            write_rows(CALIB_DIR / f"retest_{key}_cap{cap}__native.jsonl", nat)
            out = pathlib.Path(f"results/retest_{key}_cap{cap}.json"); json.dump(rec, open(out, "w"), indent=1)
            print(f"  [retest cap={cap}] {key} direct={direct_acc:.3f} native={rec['native']:.3f} M={rec['M']} forced={rec['forced_rate']:.3f} in_window={rec['in_window']} "
                  f"forced<20%={rec['forced_lt_20']} acc_forced={rec['acc_forced']} acc_natural={rec['acc_natural']}", flush=True)
            print(f"  wrote {out}", flush=True)
            continue
        # 1) direct、2) native（含 M）
        if key.startswith("ord_"):                  # r3：第 1 条在 direct-check 2000 题上测（训练中残留检查用同一划分）；stop 500 上的 direct 并报
            d_items = tasks.make_eval_set(key, 2000, split="direct")
            d_rows = run_eval(model, tok, key, d_items, "direct", gen_cfg, seed=args.seed); direct = summarize(d_rows)["acc"]
            direct_stop500 = summarize(run_eval(model, tok, key, items, "direct", gen_cfg, seed=args.seed))["acc"]
        else:
            d_items = items; direct_stop500 = None
            d_rows = run_eval(model, tok, key, items, "direct", gen_cfg, seed=args.seed); direct = summarize(d_rows)["acc"]
        HOOK.enabled = True
        nat = run_eval(model, tok, key, items, "think", gen_cfg, seed=args.seed); native = summarize(nat)["acc"]
        Ls = [r["think_tokens"] for r in nat]; M = float(statistics.median(Ls))
        print(f"  direct={direct:.3f} native={native:.3f} M={M}", flush=True)
        # v7 第 3 条：题面格式检查——同一批题的英文 gloss 题面原生准确率；第 7 条：模板命中率；oracle 下限 / S 分布
        native_gloss = template_rate = oracle_lb = S_dist = decision_lb = pairwise_direct = pairwise_baseline = None
        if key.startswith("ord_"):                    # v8：捷径命中率（原生轨迹）、决策下限中位、S 分布、chance = 1/中位 #LE(hard)、宽松抽取准确率
            from cot_compress.shortcuts import shortcut_rate, strategy_class, class_distribution
            from cot_compress.traj_stats import think_text
            from collections import Counter
            pk = task.parse_key(key)
            _sr = shortcut_rate([think_text(r["completion"]) for r in nat], [it["prompt"] for it in items], pk["n"], pk["d"]); template_rate = _sr["any"]
            _sr_legacy = shortcut_rate([think_text(r["completion"]) for r in nat], [it["prompt"] for it in items], pk["n"], pk["d"], legacy_gv=True)     # D9：原定义并报
            print(f"  shortcut by type (D9 def C): {_sr['by_type']} | legacy r4 definition: any={_sr_legacy['any']:.3f} {_sr_legacy['by_type']}", flush=True)
            oracle_lb = {c: float(statistics.median(it["meta"]["kahn_lb_tokens"][c] for it in items)) for c in ("2", "2.5", "3")}     # 第 8 条用 Kahn 下限
            decision_lb = {c: float(statistics.median(it["meta"]["decision_lb_tokens"][c] for it in items)) for c in ("2", "2.5", "3")}
            pw = [task.pairwise_accuracy([int(x) for x in r["pred"].split()], it["meta"]["sigma"]) for r, it in zip(d_rows, d_items) if r.get("pred")]
            pairwise_direct = statistics.mean(x for x in pw if x is not None) if any(x is not None for x in pw) else None
            pairwise_baseline = statistics.mean(it["meta"]["stats"].get("pairwise_baseline") or task.pairwise_baseline(pk["n"], [tuple(h) for h in it["meta"]["hard"]], it["meta"]["sigma"]) for it in d_items)
            S_dist = dict(Counter(str(it["meta"]["S"]) for it in items))
            chance = 1.0 / statistics.median(it["meta"]["stats"]["n_le_hard"] for it in items)
            strat = class_distribution([strategy_class(think_text(r["completion"]), it["prompt"], pk["n"], it["meta"]["S"]) for r, it in zip(nat, items)])
            print(f"  shortcut_hits={template_rate:.3f} strategy={strat} kahn_lb_median={oracle_lb} decision_lb_median={decision_lb} S_dist={S_dist} chance={chance:.5f} lenient={summarize(nat)['lenient_acc']:.3f} pairwise_direct={pairwise_direct} vs baseline={pairwise_baseline:.3f}", flush=True)
        if key.startswith("kk_"):
            from cot_compress.templates import hit_rate
            from cot_compress.traj_stats import think_text
            from collections import Counter
            g_items = [dict(it, prompt=it["prompt_gloss"]) for it in items]
            g_rows = run_eval(model, tok, key, g_items, "think", gen_cfg, seed=args.seed); native_gloss = summarize(g_rows)["acc"]
            write_rows(CALIB_DIR / f"admission_{key}__native_gloss.jsonl", g_rows)
            n_p = task.parse_key(key)["n"]
            th = hit_rate([think_text(r["completion"]) for r in nat], [it["prompt"] for it in items], n_p); template_rate = th["any"]
            oracle_lb = {c: float(statistics.median(it["meta"]["oracle_lb"][c] for it in items)) for c in ("2", "2.5", "3")}
            S_dist = dict(Counter(str(it["meta"]["S"]) for it in items))
            print(f"  native_gloss={native_gloss:.3f} template_hits={th} oracle_lb_median={oracle_lb} S_dist={S_dist}", flush=True)
        # 3) 强制预算曲线 {f·M}
        budget = {}
        for f in FRACS:
            fr = forced_answer_eval(model, tok, nat, int(f * M), gen_cfg, seed=args.seed, batch_size=gen_cfg["batch_size"])
            budget[f] = summarize(fr)["acc"]
        print(f"  budget_curve={budget}", flush=True)
        # 4) 白名单 unigram 填充：原生轨迹里落在白名单的 token 的 unigram（无则均匀）
        st = TrajStats(tok, nat, lid); rng = random.Random(args.seed)
        support = [t for t in st.support if t in lf_allowed] or sorted(lf_allowed - {lid})
        w = [st.unigram[t] for t in support] if support and support[0] in st.unigram else None
        filler = {}
        for f in FRACS:
            L = int(f * M)
            th = [rng.choices(support, weights=w, k=L) if w else rng.choices(support, k=L) for _ in items]
            filler[f] = summarize(prefilled_answer_eval(model, tok, key, items, th, gen_cfg, seed=args.seed, batch_size=gen_cfg["batch_size"], tag=f"filler{f}"))["acc"]
        print(f"  filler_curve={filler}", flush=True)
        # 5) 原生轨迹移植 i→j（同长度）
        from cot_compress.admission import derangement                       # r3：移植配对 = 随机错位（题 i 用题 π(i) 的原生轨迹，π 无不动点）
        pi = derangement(len(items), rng)
        th = [st.by_item[items[pi[i]]["id"]][0] for i in range(len(items))]
        transplant = summarize(prefilled_answer_eval(model, tok, key, items, th, gen_cfg, seed=args.seed, batch_size=gen_cfg["batch_size"], tag="transplant"))["acc"]
        # 6) 冻结模型 + letter-free mask
        HOOK.configure(tok, reserve=reserve, banned_ids=lf_banned, mask_mode="letterfree")
        masked = summarize(run_eval(model, tok, key, items, "think", gen_cfg, seed=args.seed))["acc_clean"]   # c ∧ ¬v：第 0 步 </think> 逃出屏蔽再推理的算失败
        HOOK.configure(tok, reserve=reserve, banned_ids=None)
        nat_s = summarize(nat)
        metrics = dict(key=key, n=n, backend=args.backend, chance=chance, direct=direct, native=native, native_clean=nat_s["acc_clean"],
                       native_viol_rate=nat_s["viol_rate"], native_forced_rate=nat_s["forced_rate"], M=M, L_median=M,
                       budget_curve={str(k): v for k, v in budget.items()}, filler_curve={str(k): v for k, v in filler.items()},
                       transplant_acc=transplant, masked_frozen_acc=masked, native_gloss=native_gloss, template_hit_rate=template_rate,
                       oracle_lb=oracle_lb, S_dist=S_dist, target_S=(task.parse_key(key)["s"] if key.startswith("kk_") else None),
                       native_lenient=nat_s["lenient_acc"], direct_lenient=summarize(d_rows)["lenient_acc"], direct_n=len(d_rows), direct_stop500=direct_stop500,
                       direct_format_err=summarize(d_rows)["format_err"], max_new_tokens_direct=int(gen_cfg["max_new_tokens_direct"]),
                       d=(task.parse_key(key)["d"] if key.startswith("ord_") else None), transplant_pairing="derangement",
                       strategy_class_native=(strat if key.startswith("ord_") else None),
                       shortcut_by_type=(_sr["by_type"] if key.startswith("ord_") else None), shortcut_legacy_r4=(dict(any=_sr_legacy["any"], by_type=_sr_legacy["by_type"]) if key.startswith("ord_") else None),
                       kahn_lb=(oracle_lb if key.startswith("ord_") else None), decision_lb=(decision_lb if key.startswith("ord_") else None),
                       pairwise_direct=(pairwise_direct if key.startswith("ord_") else None), pairwise_baseline=(pairwise_baseline if key.startswith("ord_") else None))
        if key.startswith("cups"):
            from cot_compress.style import style_summary
            from cot_compress.traj_stats import think_text
            metrics["rewrite_style"] = style_summary([think_text(r["completion"]) for r in nat], task.parse_key(key)["n"])
        metrics["decision"] = admission_decision(metrics)
        write_rows(CALIB_DIR / f"admission_{key}__native.jsonl", nat)
        out = pathlib.Path(f"results/admission_{key}.json"); out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(metrics, open(out, "w"), indent=1)
        print(f"  transplant={transplant:.3f} masked_frozen={masked:.3f} -> admitted={metrics['decision']['admitted']} pending={metrics['decision']['pending']}", flush=True)
        print(f"  wrote {out}", flush=True)

if __name__ == "__main__":
    main()
