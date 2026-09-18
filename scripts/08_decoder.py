"""v8 解码器（§4）：仅前缀的 n-gram 词袋 → 真值标签（按发布解析器对齐的步数 t：下一事件 / 真值就绪集 / 已定析取集）。
仪器门：任一臂 next 准确率 ≥ 90% 且在该臂的移植轨迹**和** C_rand 轨迹上都掉 ≥ 10pp（配对 bootstrap；C_rand 轨迹 = 传入的 Crand run 的行按题配对该臂标签）；B 的条款只在 B 解析率 ≥ 50% 时生效。
另报：各臂解析率、每条前缀的 soundness / completeness（描述量）、B_warm 映射配对的密码 Spearman（描述量）。
用法: python scripts/08_decoder.py --key ord_n8_h4_d5 --runs A=runs/A_0.5_ord_n8_h4_d5_s1 B=runs/B_0.5_ord_n8_h4_d5_s1 Crand=runs/Crand_0.5_ord_n8_h4_d5_s1 [--tokenizer unsloth/Qwen3-1.7B]
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.environ.setdefault("COT_NO_UNSLOTH", "1")
import argparse, json, statistics
import tasks
from cot_compress.decoder import prefix_rows, run_ordering_decoder, paired_bootstrap, transplant_rows, soundness_completeness, cipher_spearman_warm
from cot_compress.traj_stats import think_text, load_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True); ap.add_argument("--runs", nargs="+", required=True, help="arm=runs/<dir> ...")
    ap.add_argument("--tokenizer"); ap.add_argument("--out")
    args = ap.parse_args()
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained(args.tokenizer)
    n = tasks.ordering.parse_key(args.key)["n"]
    items = {it["id"]: it for k, sp in tasks.ordering.load_splits(args.key, 0).items() if isinstance(sp, list) for it in sp}
    R = dict(key=args.key, arms={}); thinks = {}; specs = dict(s.split("=", 1) for s in args.runs)
    crand_rows = {r["id"]: think_text(r.get("completion") or r.get("text")) for r in load_rows(ROOT / specs["Crand"]) if r["id"] in items} if "Crand" in specs else {}
    for arm, run in specs.items():
        rows = [dict(id=r["id"], think=think_text(r.get("completion") or r.get("text")), prompt=items[r["id"]]["prompt"]) for r in load_rows(ROOT / run) if r["id"] in items]
        thinks[arm] = [r["think"] for r in rows]
        prow, prate = prefix_rows(rows, items, tok=tok); tprow = transplant_rows(prow)
        res = {t: run_ordering_decoder(prow, n, t) for t in ("next", "ready", "decided")}
        tres = {t: run_ordering_decoder(tprow, n, t) for t in ("next", "ready", "decided")}
        cres = {}
        if crand_rows and arm != "Crand":                                     # 该臂标签 + C_rand 同题轨迹（前缀同样切分）
            crow, _ = prefix_rows([dict(id=r["id"], think=crand_rows[r["id"]], prompt=r["prompt"]) for r in rows if r["id"] in crand_rows], items, tok=tok)
            cres = {t: run_ordering_decoder(crow, n, t) for t in ("next", "ready", "decided")}
        sc = [soundness_completeness(r["think"], items[r["id"]]) for r in rows]
        R["arms"][arm] = dict(parse=prate, decoder={t: {k: v for k, v in res[t].items() if k != "per_item_acc"} for t in res},
                              transplant={t: {k: v for k, v in tres[t].items() if k != "per_item_acc"} for t in tres},
                              crand={t: {k: v for k, v in cres[t].items() if k != "per_item_acc"} for t in cres},
                              drop_vs_transplant={t: paired_bootstrap(res[t].get("per_item_acc", {}), tres[t].get("per_item_acc", {})) for t in res},
                              drop_vs_crand={t: paired_bootstrap(res[t].get("per_item_acc", {}), cres[t].get("per_item_acc", {})) for t in cres},
                              soundness_mean=statistics.mean(x["soundness"] for x in sc if x["soundness"] is not None) if any(x["soundness"] is not None for x in sc) else None,
                              completeness_mean=statistics.mean(x["completeness"] for x in sc if x["completeness"] is not None) if any(x["completeness"] is not None for x in sc) else None)
    gate = {}
    for arm, a in R["arms"].items():
        if arm == "Crand": continue
        acc = a["decoder"]["next"].get("acc"); drop = a["drop_vs_transplant"]["next"]; dropc = (a.get("drop_vs_crand") or {}).get("next", {})
        ok = (acc is not None and acc >= 0.90 and drop.get("lo") is not None and drop["lo"] >= 0.10 and dropc.get("lo") is not None and dropc["lo"] >= 0.10)
        if arm == "B" and (a["parse"]["parse_rate"] or 0) < 0.5: ok = None
        gate[arm] = dict(ok=ok, acc=acc, drop_transplant=drop, drop_crand=dropc, parse_rate=a["parse"]["parse_rate"], note=("no Crand run given" if not dropc else None))
    R["instrument_gate"] = dict(per_arm=gate, ok=any(v["ok"] for v in gate.values()))
    if "A" in thinks and "B" in thinks: R["cipher_spearman_warm"] = cipher_spearman_warm(thinks["A"], thinks["B"])
    out = ROOT / (args.out or f"results/decoder_{args.key}.json"); json.dump(R, open(out, "w"), indent=1); print(json.dumps(R, indent=1)); print("wrote", out)

if __name__ == "__main__":
    main()
