"""第二阶段：用新 seed 生成 500 题的全新评估集，与 stage-1 的所有划分（train / stop / report / direct）不相交。
同一生成器与同一分布（tasks.ordering.generate：引导式、唯一线性序、S≥1），只换 seed 与划分名。
不相交按两层断言：同构类规范形（meta.canonical）与题面字符串。
用法：python scripts/phase2_build_eval.py [--seed 1] [--n 500] [--workers 11]
输出：data/ordering/ord_n8_h4_d5_p2eval_seed{seed}.json（+ .report.json）"""
import os; os.environ.setdefault("COT_NO_UNSLOTH", "1")
import argparse, json, pathlib, sys, statistics as st, multiprocessing as mp, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tasks import ordering as O

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key", default="ord_n8_h4_d5"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--workers", type=int, default=11); a = ap.parse_args()
    split = "p2eval"
    old = O.load_splits(a.key, 0)
    old_forms = {it["meta"]["canonical"] for sp in old.values() if isinstance(sp, list) for it in sp}
    old_prompts = {it["prompt"] for sp in old.values() if isinstance(sp, list) for it in sp}
    items, forms, j, rej_old = [], set(), 0, 0; gen = collections.Counter()
    while len(items) < a.n:
        batch = [(a.key, split, a.seed, j + t) for t in range(int((a.n - len(items)) * 1.1) + 8)]; j += len(batch)
        with mp.Pool(a.workers) as pool: cands = pool.map(O.generate_indexed, batch, chunksize=2)
        for it in sorted(cands, key=lambda x: x["_j"]):
            gen.update(it.pop("_stats")); cf = it["meta"]["canonical"]
            if cf in old_forms or it["prompt"] in old_prompts: rej_old += 1; continue
            if cf in forms or len(items) >= a.n: continue
            forms.add(cf); del it["_j"]; it["id"] = f"{a.key}/{split}/{a.seed}/{len(items)}"; items.append(it)
    assert not (forms & old_forms) and not ({it["prompt"] for it in items} & old_prompts)
    out = {split: items}
    p = O.DATA_DIR / f"{a.key}_p2eval_seed{a.seed}.json"; json.dump(out, open(p, "w"))
    S = collections.Counter(it["meta"]["S"] for it in items)
    rep = dict(key=a.key, seed=a.seed, split=split, n=len(items), candidates_drawn=j, rejected_as_stage1_duplicate=rej_old,
               distinct_isomorphism_classes=len(forms), overlap_with_stage1_canonical=0, overlap_with_stage1_prompts=0,
               S_dist=dict(sorted(S.items())), gen_stats=dict(gen),
               stats_median={k: st.median(it["meta"]["stats"][k] for it in items) for k in ("n_le_hard", "max_antichain", "ham_paths", "surviving_disj")},
               kahn_lb_elements_median=st.median(it["meta"]["kahn_lb_elements"] for it in items) if "kahn_lb_elements" in items[0]["meta"] else None,
               stage1_splits_checked={k: len(v) for k, v in old.items() if isinstance(v, list)})
    json.dump(rep, open(str(p).replace(".json", ".report.json"), "w"), indent=1); print(json.dumps(rep, indent=1))

if __name__ == "__main__": main()
