"""离线补算参考模型困惑度：对 samples/<run>/step*.jsonl 每个文件算 L 段前 1024 token 的 PPL（Qwen3-0.6B-Base，GPU bf16），
写 runs/<run>/ppl.jsonl 并打印。训练进程内算不了（Unsloth 给驻留模型打的注意力补丁会波及第二个 Qwen3 实例），
所以单独进程跑；显存 ~1.6 GB，可与训练并行（会稍微拖慢训练）。
用法: python scripts/05_ppl_samples.py --run A_0.3
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import cot_compress  # noqa
import argparse, json
from cot_compress.evaluate import ref_ppl_rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = ROOT / "runs" / args.run / "ppl.jsonl"
    done = {json.loads(l)["step"] for l in open(out)} if out.exists() and not args.force else set()
    for f in sorted((ROOT / "samples" / args.run).glob("step*.jsonl")):
        step = int(f.stem[4:])
        if step in done:
            continue
        rows = [json.loads(l) for l in open(f)]
        ppl = ref_ppl_rows(rows)
        rec = dict(step=step, n=len(rows), ref_ppl=ppl)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec, flush=True)

if __name__ == "__main__":
    main()
