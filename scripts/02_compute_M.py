"""阶段 3：M = 该难度下原生轨迹 L 的中位数（触顶 / 被强制收尾按 cap 计）。
32 题 × 1 次采样，T=0.6，batch 4，cap = train.max_completion_length，到顶用 "</think>\\n\\n<answer>" 强制收尾。
写 results/M.json：{key: {M, n, cap, L, forced_rate, acc}}。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import cot_compress  # noqa  unsloth first
import argparse, json, statistics, time
import tasks
from cot_compress.config import load_config
from cot_compress.evaluate import run_eval, summarize
from cot_compress.span import HOOK

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_1p7b.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--key"); ap.add_argument("--n", type=int); ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    key = args.key or cfg["task"]["key"]; n = args.n or int(cfg["task"].get("m_n", 32))
    cap = int(cfg["train"]["max_completion_length"])
    gen = dict(cfg["generation"], batch_size=args.batch, max_new_tokens_think=cap)
    from cot_compress.model import load_model
    model, tok = load_model(cfg)
    from cot_compress.memory import tune_allocator; print(tune_allocator(), flush=True)
    HOOK.configure(tok, reserve=int(cfg["train"].get("force_reserve", 24)), banned_ids=None); HOOK.install(model)
    items = tasks.make_eval_set(key, n, seed=12345)      # 与 eval 集（seed 0）分开
    t0 = time.time()
    rows = run_eval(model, tok, key, items, "think", gen, seed=0,
                    progress=lambda d, t: print(f"  {d}/{t} ({time.time()-t0:.0f}s)", flush=True))
    Ls = [r["think_tokens"] for r in rows]
    s = summarize(rows)
    M = float(statistics.median(Ls))
    rec = dict(M=M, n=len(rows), cap=cap, L=Ls, L_mean=round(statistics.mean(Ls), 1), forced_rate=s["forced_rate"],
               acc=s["acc"], natural_end=1 - s["forced_rate"], sec=round(time.time() - t0), temperature=gen.get("temperature"))
    path = pathlib.Path(cfg["task"].get("m_file", "results/M.json"))
    data = json.load(open(path)) if path.exists() else {}
    data[key] = rec
    json.dump(data, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "L"}, indent=1))
    print("sorted L:", sorted(Ls))
    print("wrote", path)

if __name__ == "__main__":
    main()
