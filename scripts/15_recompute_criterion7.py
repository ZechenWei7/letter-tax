"""D12 / D13：用已存的原生轨迹（results/calib/admission_<key>__native.jsonl）重算捷径检测器的四种命中率（r4 / D9 / D12 / D13，全是描述量），不重跑生成；
D13 起准入第 7 条 = 人工审计：--audit-rate（"模板 / 抄答案"占原生轨迹的比例）与 --audit-file（审计表）写进 admission json 后重新调用 admission_decision。其余指标原样保留。
用法: python scripts/15_recompute_criterion7.py --key ord_n8_h4_d5 [--results results] [--audit-rate 0.0 --audit-file results/audit_criterion7_ord_n8_h4_d5.md] [--write]
不加 --write 只打印。"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from cot_compress.admission import admission_decision
from cot_compress.shortcuts import shortcut_rates_all
from cot_compress.traj_stats import think_text
import tasks

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key", required=True); ap.add_argument("--results", default="results"); ap.add_argument("--write", action="store_true")
    ap.add_argument("--audit-rate", type=float, default=None); ap.add_argument("--audit-file", default=None)
    a = ap.parse_args(); R = pathlib.Path(a.results)
    jp = R / f"admission_{a.key}.json"; m = json.load(open(jp))
    rows = [json.loads(l) for l in open(R / "calib" / f"admission_{a.key}__native.jsonl") if l.strip()]
    assert len(rows) == m["n"] and abs(sum(r["correct"] for r in rows) / len(rows) - m["native"]) < 1e-9, "stored rows do not match the admission json"
    pk = tasks.task_for_key(a.key).parse_key(a.key); th = [think_text(r["completion"]) for r in rows]; pr = [r["prompt"] for r in rows]
    rates = shortcut_rates_all(th, pr, pk["n"], pk["d"])
    m.update(template_hit_rate=rates["D13"]["any"], shortcut_by_type=rates["D13"]["by_type"], shortcut_rates=rates,
             shortcut_legacy_r4=dict(any=rates["r4"]["any"], by_type=rates["r4"]["by_type"]), criterion7_recomputed="D13, detector rates from stored native trajectories (no regeneration)")
    if a.audit_rate is not None:
        assert a.audit_file and pathlib.Path(a.audit_file).exists(), "--audit-rate needs an existing --audit-file"
        m.update(audit_template_rate=a.audit_rate, audit_file=a.audit_file)
    m["decision"] = admission_decision(m)
    print(json.dumps(dict(rates=rates, audit_template_rate=m.get("audit_template_rate"), admitted=m["decision"]["admitted"], pending=m["decision"]["pending"],
                          failed=[k for k, v in m["decision"]["checks"].items() if v["ok"] is False], M=m["M"]), indent=1))
    if a.write: json.dump(m, open(jp, "w"), indent=1); print("wrote", jp)

if __name__ == "__main__":
    main()
