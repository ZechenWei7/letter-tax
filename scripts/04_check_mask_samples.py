"""阶段 5 自动核对：臂 B smoke test 的完成样本里，L 段（生成开始→<answer>）落在屏蔽集的 token 出现次数必须为 0。
用法: python scripts/04_check_mask_samples.py --run runs/smoke_B [--n 10]（禁集 = vocab_mask.banned_ids，按字符类型）
数据源：runs/<run>/reward_log.jsonl（每步的 rows[*].text）。
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import argparse, json
from transformers import AutoTokenizer
from cot_compress.parsing import ANSWER_OPEN
from cot_compress.vocab_mask import banned_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/smoke_B")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--tokenizer", default="unsloth/Qwen3-1.7B")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    banned = set(banned_ids(tok))
    texts = []
    for line in open(ROOT / args.run / "reward_log.jsonl"):
        for r in json.loads(line)["rows"]:
            texts.append(r["text"])
    texts = texts[:args.n]
    total_bad, report = 0, []
    for i, t in enumerate(texts):
        pre = t.split(ANSWER_OPEN, 1)[0] if ANSWER_OPEN in t else t
        ids = tok(pre, add_special_tokens=False)["input_ids"]
        bad = [tok.decode([x]) for x in ids if x in banned]
        total_bad += len(bad)
        report.append(dict(i=i, L=len(ids), banned_hits=len(bad), examples=bad[:8], has_answer=ANSWER_OPEN in t))
    for r in report:
        print(r)
    print(f"samples={len(texts)} total_banned_hits={total_bad} -> {'PASS' if total_bad == 0 and texts else 'FAIL'}")
    sys.exit(0 if total_bad == 0 and texts else 1)

if __name__ == "__main__":
    main()
