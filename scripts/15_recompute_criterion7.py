"""D12：用已存的原生轨迹（results/calib/admission_<key>__native.jsonl）重算准入第 7 条，不重跑生成。
其余各项指标原样保留，只替换 template_hit_rate / shortcut_by_type / shortcut_legacy_r4 并重新调用 admission_decision。
用法: python scripts/15_recompute_criterion7.py --key ord_n8_h4_d5 [--results results] [--write]
不加 --write 只打印。"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from cot_compress.admission import admission_decision
from cot_compress.shortcuts import shortcut_rate
from cot_compress.traj_stats import think_text
import tasks

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key", required=True); ap.add_argument("--results", default="results"); ap.add_argument("--write", action="store_true")
    a = ap.parse_args(); R = pathlib.Path(a.results)
    jp = R / f"admission_{a.key}.json"; m = json.load(open(jp))
    rows = [json.loads(l) for l in open(R / "calib" / f"admission_{a.key}__native.jsonl") if l.strip()]
    assert len(rows) == m["n"] and abs(sum(r["correct"] for r in rows) / len(rows) - m["native"]) < 1e-9, "stored rows do not match the admission json"
    pk = tasks.task_for_key(a.key).parse_key(a.key); th = [think_text(r["completion"]) for r in rows]; pr = [r["prompt"] for r in rows]
    cur = shortcut_rate(th, pr, pk["n"], pk["d"]); d9 = shortcut_rate(th, pr, pk["n"], pk["d"], legacy_pe=True); r4 = shortcut_rate(th, pr, pk["n"], pk["d"], legacy_gv=True, legacy_pe=True)
    assert abs(d9["any"] - m["template_hit_rate"]) < 1e-9 or m.get("shortcut_d9_only"), f"D9-only rate {d9['any']} != stored {m['template_hit_rate']}"
    m.update(template_hit_rate=cur["any"], shortcut_by_type=cur["by_type"], shortcut_d9_only=dict(any=d9["any"], by_type=d9["by_type"]),
             shortcut_legacy_r4=dict(any=r4["any"], by_type=r4["by_type"]), criterion7_recomputed="D12, from stored native trajectories (no regeneration)")
    m["decision"] = admission_decision(m)
    print(json.dumps(dict(D12=cur, D9_only=d9, legacy_r4=r4, admitted=m["decision"]["admitted"], pending=m["decision"]["pending"],
                          failed=[k for k, v in m["decision"]["checks"].items() if not v["ok"]], M=m["M"]), indent=1))
    if a.write: json.dump(m, open(jp, "w"), indent=1); print("wrote", jp)

if __name__ == "__main__":
    main()
