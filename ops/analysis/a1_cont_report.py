"""探索性实验 · A1 延续（λ=1.0，从 A1 checkpoint-350 起固定 150 步）的报告。不属于闸门 1 判读分区；全部是描述量。
读法事先写定（用户 2026-09-23，写在任何延续数据之前），脚本按下面的规则机械套用，不另作解释：
  字母占比：延续段每次 eval（3 次）的 letter_frac 都在 [0.77, 0.80] → "更强压力下删的是内容，不换写法"；
            最后一次 eval < 0.70 → "更强压力下开始换写法"；其余情形 → 两条都不适用（如实报数，不套读法）。
  长度 × 准确率（最后一次延续 eval 对 A1 step-350 eval）：L_mean 下降且准确率掉 ≥ 5pp → "在拿准确率换长度"；
            L_mean 下降且准确率掉 < 5pp → "找到了更省的写法"；L_mean 未下降 → 两条都不适用。
  并列：A1 最后 100 步（step 250 → 300 → 350，λ=0.5）每 50 步的变化速度。
CoT 抽样（同之前的方法，ops/analysis/cot_scan.py 与 pair_dump.py）：
  (a) 训练 rollouts 里正确轨迹的中位长度 / 字符 / 字母占比 / 叙述标记（每千字符），A1 最后一步与延续段若干步；
  (b) 同题对：A1 step-350 eval 与延续最后一次 eval 都答对的题，按 token 缩减取 p25 / p50 / p75 导出并排文件。
用法：python ops/analysis/a1_cont_report.py --run runs/A_1.0_cont_from_A1s350 --base runs/A_0.5_ord_n8_h4_d5_s1 \
        --out results/a1_cont_report.md --samples-out ops/cot_samples/a1_cont
  --dry-run：用 A1 自己的 step 200 当"起点"、250/300/350 当"延续"演练代码路径（数字无意义）。"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import os; os.environ.setdefault("COT_NO_UNSLOTH", "1")
import argparse, json, re, statistics as st
from cot_compress.archive import read_jsonl_zst
from cot_compress.lengths import letter_fraction
from cot_compress.traj_stats import think_text

LETTER_KEEP = (0.77, 0.80)
LETTER_SWITCH = 0.70
ACC_DROP = 0.05
PATS = [("wait", r"\bwait\b"), ("but/however", r"\b(?:but|however)\b"), ("check/verify", r"\b(?:check|verif|confirm|recheck)"),
        ("so/thus", r"\b(?:so|thus|therefore|hence)\b"), ("if/suppose/case", r"\b(?:if|suppose|assume|case)\b"),
        ("sentences", r"[.!?](?:\s|$)"), ("arrows+<", r"→|->|<")]
KEYS = ("L_mean", "L_median", "acc", "capped_rate", "letter_frac")


def evals(run):
    return {int(r["step"]): r for r in map(json.loads, open(pathlib.Path(run) / "eval.jsonl")) if r.get("split", "stop") == "stop"}


def letter_reading(fr):
    if all(LETTER_KEEP[0] <= x <= LETTER_KEEP[1] for x in fr): return "KEEP", "更强压力下删的是内容，不换写法"
    if fr[-1] < LETTER_SWITCH: return "SWITCH", "更强压力下开始换写法"
    return "NONE", f"两条都不适用（字母占比 {', '.join(f'{x:.3f}' for x in fr)}：既非每次都在 77–80%，最后一次也不低于 70%）"


def length_reading(L0, L1, a0, a1):
    if not L1 < L0: return "NONE", f"L_mean 未下降（{L0:.1f} → {L1:.1f}），两条都不适用"
    drop = a0 - a1
    if drop >= ACC_DROP - 1e-12: return "TRADE", f"在拿准确率换长度（准确率 {a0:.3f} → {a1:.3f}，掉 {100 * drop:.1f}pp ≥ 5pp）"
    return "CHEAPER", f"找到了更省的写法（准确率 {a0:.3f} → {a1:.3f}，变化 {-100 * drop:+.1f}pp，掉不到 5pp）"


def cot_scan(run, steps):
    """正确轨迹的中位数（同 ops/analysis/cot_scan.py 与 cot_scan2.py 的量）。"""
    buck = {}
    p = pathlib.Path(run) / "rollouts.jsonl.zst"
    if not p.exists(): return {}
    for r in read_jsonl_zst(p):
        if r["step"] in steps: buck.setdefault(r["step"], []).append(r)
    out = {}
    for s in sorted(buck):
        rows = [r for r in buck[s] if r.get("c")]
        rec = dict(n=len(buck[s]), correct=len(rows))
        if rows:
            th = [think_text(r["text"]) for r in rows]
            rec |= dict(L_median=st.median(r["L"] for r in rows), chars_median=st.median(len(t) for t in th),
                        letter_frac_median=round(st.median(letter_fraction(t) for t in th), 4),
                        per_kchar={n: round(st.median(1000 * len(re.findall(pt, t, re.I)) / max(len(t), 1) for t in th), 2) for n, pt in PATS})
        out[s] = rec
    return out


def pair_dump(A_path, B_path, outdir, tagA, tagB):
    A = {r["id"]: r for r in read_jsonl_zst(A_path)}; B = {r["id"]: r for r in read_jsonl_zst(B_path)}
    both = [i for i in A if i in B and A[i]["correct"] and B[i]["correct"]]
    if len(both) < 4: return dict(n_both=len(both))
    red = sorted(((1 - B[i]["think_tokens"] / A[i]["think_tokens"]), i) for i in both)
    picks = [("q1_p25", red[len(red) // 4]), ("q2_p50", red[len(red) // 2]), ("q3_p75", red[3 * len(red) // 4])]
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, (r, i) in picks:
        for lab, src in ((tagA, A), (tagB, B)):
            rec = src[i]; th = think_text(rec["completion"])
            n = lambda p: len(re.findall(p, th, re.I))
            s = dict(chars=len(th), letter=round(100 * letter_fraction(th), 1), sent=n(r"[.!?](?:\s|$)"), wait=n(r"\bwait\b"),
                     check=n(r"\b(?:check|verif|confirm|recheck)"), arrow=n(r"→|->|<"))
            (outdir / f"{tag}_{lab}.txt").write_text(
                f"# {rec['id']}   {lab}\n# think_tokens={rec['think_tokens']}  chars={s['chars']}  字母占比={s['letter']}%  句子={s['sent']}  wait={s['wait']}  "
                f"check/verify={s['check']}  箭头+<={s['arrow']}\n# capped={rec['capped']}  correct={rec['correct']}\n# gold={rec['gold']}   pred={rec['pred']}\n"
                f"{'=' * 100}\n[题面]\n{rec['prompt']}\n{'=' * 100}\n[think 段]\n{th}\n")
            rows.append(dict(file=f"{tag}_{lab}.txt", id=i, think_tokens=rec["think_tokens"], **s))
    return dict(n_both=len(both), reduction_p25=red[len(red) // 4][0], reduction_p50=red[len(red) // 2][0], reduction_p75=red[3 * len(red) // 4][0], rows=rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--samples-out", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run, base = pathlib.Path(a.run), pathlib.Path(a.base)
    for o in (a.out, a.samples_out):                                             # A1 原目录只读（用户 2026-09-23）：输出不得落在其中
        assert not str(pathlib.Path(o).resolve()).startswith(str(base.resolve()) + "/"), f"输出 {o} 落在 A1 原目录里"
    E0 = evals(base)
    if a.dry_run:
        start, offset, origin = 200, 0, dict(mode="DRY-RUN（A1 自身的 200 → 350 当作延续，数字无意义）")
        EC = {s: E0[s] for s in (250, 300, 350)}; speed_steps = (100, 150, 200)
    else:
        origin = json.load(open(run / "continuation_origin.json"))
        start = 350; offset = 350 if origin["global_step_start"] == 0 else 0            # 从 final/ 起训时步号从 0 记 → 换算为 A1 步
        EC = {s + offset: r for s, r in evals(run).items() if s + offset > start}; speed_steps = (250, 300, 350)
    cont = sorted(EC)
    assert cont, "延续段没有 eval"
    b0 = E0[start]
    fr = [EC[s]["letter_frac"] for s in cont]
    lr = letter_reading(fr); Lr = length_reading(b0["L_mean"], EC[cont[-1]]["L_mean"], b0["acc"], EC[cont[-1]]["acc"])
    # A1 最后 100 步的速度（λ=0.5）与延续段（λ=1.0）每 50 步的变化
    sp = [dict(seg=f"A1 {x}→{y}（λ=0.5）", **{k: E0[y][k] - E0[x][k] for k in KEYS}) for x, y in zip(speed_steps, speed_steps[1:])]
    seq = [start] + cont; allE = {**{start: b0}, **EC}
    sc = [dict(seg=f"延续 {x}→{y}（λ=1.0）", **{k: allE[y][k] - allE[x][k] for k in KEYS}) for x, y in zip(seq, seq[1:])]
    # CoT 抽样
    last_base_step = max(r["step"] for r in map(json.loads, open(base / "steps.jsonl")) if r["step"] <= start) if (base / "steps.jsonl").exists() else start
    scan_base = cot_scan(base, {last_base_step - 1, last_base_step})
    rsteps = sorted({r["step"] for r in read_jsonl_zst(run / "rollouts.jsonl.zst")}) if (run / "rollouts.jsonl.zst").exists() and not a.dry_run else []
    pick = sorted({rsteps[0], rsteps[len(rsteps) // 3], rsteps[2 * len(rsteps) // 3], rsteps[-1]}) if rsteps else []
    scan_cont = cot_scan(run, set(pick)) if pick else {}
    tagA, tagB = f"A1_step{start:03d}", f"cont_step{cont[-1]:03d}"
    pdA = base / f"eval_step{start:04d}.jsonl.zst"
    pdB = (run / f"eval_step{cont[-1] - offset:04d}.jsonl.zst") if not a.dry_run else base / f"eval_step{cont[-1]:04d}.jsonl.zst"
    pdump = pair_dump(pdA, pdB, a.samples_out, tagA, tagB) if pdA.exists() and pdB.exists() else dict(missing=[str(pdA), str(pdB)])
    budget = [json.loads(l) for l in open(run / "budget.jsonl")] if (run / "budget.jsonl").exists() and not a.dry_run else []
    # 报告
    f3 = lambda x: f"{x:.3f}"; pp = lambda x: f"{100 * x:+.1f}pp"
    L = [f"# 探索性实验 · A1 延续（λ=1.0，固定 150 步）报告", "", "不属于闸门 1 判读分区；全部是描述量。读法事先写定（见脚本文件头）。", "",
         f"起点：{origin.get('mode')}；{origin.get('optimizer_state', '')}", "",
         "## 预先写定的读法", "",
         f"- **字母占比**：{lr[1]}",
         f"- **长度 × 准确率**：{Lr[1]}", "",
         "## 每 50 步 eval（stopping 500 题）", "", "| step | λ | L_mean | L_median | 准确率 | 到顶率 | 字母占比 |", "|---|---|---|---|---|---|---|"]
    for s in list(speed_steps) + ([start] if start not in speed_steps else []):
        r = E0[s]; L.append(f"| {s} | 0.5 | {r['L_mean']:.1f} | {r['L_median']:.1f} | {f3(r['acc'])} | {f3(r['capped_rate'])} | {f3(r['letter_frac'])} |")
    for s in cont:
        r = EC[s]; L.append(f"| {s} | 1.0 | {r['L_mean']:.1f} | {r['L_median']:.1f} | {f3(r['acc'])} | {f3(r['capped_rate'])} | {f3(r['letter_frac'])} |")
    L += ["", "## 每 50 步的变化：A1 最后 100 步（λ=0.5）与延续段（λ=1.0）并列", "", "| 区间 | ΔL_mean | ΔL_median | Δ准确率 | Δ到顶率 | Δ字母占比 |", "|---|---|---|---|---|---|"]
    for r in sp + sc:
        L.append(f"| {r['seg']} | {r['L_mean']:+.1f} | {r['L_median']:+.1f} | {pp(r['acc'])} | {pp(r['capped_rate'])} | {pp(r['letter_frac'])} |")
    L += ["", "## CoT 抽样（(a) 训练 rollouts，正确轨迹中位数；T=1.0；n 小）", "",
          "| 来源 | step | 正确/总 | L 中位 | 字符中位 | 字母占比 | " + " | ".join(n for n, _ in PATS) + "（每千字符） |", "|" + "---|" * (6 + len(PATS))]
    for lab, sc_ in (("A1 λ=0.5", scan_base), ("延续 λ=1.0", scan_cont)):
        for s, r in sc_.items():
            if r.get("correct"):
                L.append(f"| {lab} | {s} | {r['correct']}/{r['n']} | {r['L_median']:.0f} | {r['chars_median']:.0f} | {r['letter_frac_median']:.3f} | " + " | ".join(str(r["per_kchar"][n]) for n, _ in PATS) + " |")
            else:
                L.append(f"| {lab} | {s} | 0/{r['n']} | – | – | – | " + " | ".join("–" for _ in PATS) + " |")
    L += ["", f"(b) 同题对（{tagA} 对 {tagB} 都答对的题；按 token 缩减取 p25 / p50 / p75；文件在 `{a.samples_out}`）", ""]
    if "rows" in pdump:
        L += [f"两者都对 {pdump['n_both']} 道；缩减 p25 {100 * pdump['reduction_p25']:.1f}%、p50 {100 * pdump['reduction_p50']:.1f}%、p75 {100 * pdump['reduction_p75']:.1f}%", "",
              "| 文件 | think token | 字符 | 字母占比 | 句子 | wait | check/verify | 箭头+< |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| `{r['file']}` | {r['think_tokens']} | {r['chars']} | {r['letter']}% | {r['sent']} | {r['wait']} | {r['check']} | {r['arrow']} |" for r in pdump["rows"]]
    else:
        L.append(f"未导出：{json.dumps(pdump, ensure_ascii=False)}")
    if budget: L += ["", f"费用守卫最后一次投影：{json.dumps(budget[-1], ensure_ascii=False)}"]
    pathlib.Path(a.out).write_text("\n".join(L) + "\n")
    json.dump(dict(origin=origin, letter_reading=lr, length_reading=Lr, evals_cont={s: {k: EC[s][k] for k in KEYS} for s in cont},
                   eval_start={k: b0[k] for k in KEYS}, speed_A1=sp, speed_cont=sc, cot_scan_base=scan_base, cot_scan_cont=scan_cont, pair_dump=pdump),
              open(pathlib.Path(a.out).with_suffix(".json"), "w"), indent=1, ensure_ascii=False, default=str)
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
