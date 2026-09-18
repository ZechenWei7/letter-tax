"""仪器阳性对照（v8 §10）：同一条手写 DPLL 推导——排序题（tests/test_ordering.py 的手工实例：n=4，硬 0<1 2<3，析取 (1<2)|(3<0) (0<2)|(1<3)）
的英文版与符号版（B_warm 映射风格），分词后报告 token / code point 比值，以及捷径检测器、策略类检测器、字母占比在两版上的输出（进论文附录）。
K&K 版本保留在 results/instrument_control_kk.json（v7）。
用法: python scripts/13_instrument_control.py [--tokenizer unsloth/Qwen3-1.7B] → results/instrument_control.json
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.environ.setdefault("COT_NO_UNSLOTH", "1")
import argparse, json
from cot_compress.shortcuts import detect_shortcuts, strategy_class
from cot_compress.warm import warm_text, assert_letter_free
from cot_compress.lengths import letter_fraction

IR = "n=4\nhard: 0<1 2<3\ndisj: (1<2)|(3<0) (0<2)|(1<3)\nreason inside <think>, then give the order\nanswer: 4 digits, first to last"
ENGLISH = """The hard constraints give two chains: 0 before 1, and 2 before 3. Nothing else is forced yet, so I branch on the first disjunction.
Case 1: assume 3<0. Then 2<3<0<1, so 0<2 is false and 1<3 is false, so the second disjunction has no true side: contradiction.
Case 2: so 1<2 must hold. Then 0<1<2<3 is a total order. Check the second disjunction: 0<2 holds, consistent. Check the hard constraints: 0<1 holds and 2<3 holds.
Therefore the order is 0 1 2 3."""
SYMBOLIC = """0<1 2<3 ⇒ » 1
§1 » 3<0 → 2<3<0<1 ⇒ 0<2 - 1<3 - ⇒ ×
§2 ⇒ 1<2 → 0<1<2<3 ✓ 0<2 + ✓ 0<1 + 2<3 +
⇒ 0 1 2 3"""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tokenizer", default="unsloth/Qwen3-1.7B"); ap.add_argument("--out", default="results/instrument_control.json")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert_letter_free(SYMBOLIC)
    warmed = warm_text(ENGLISH)
    R = {}
    for name, text in (("english", ENGLISH), ("symbolic", SYMBOLIC), ("warm(english)", warmed)):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        R[name] = dict(tokens=len(ids), code_points=len(text), letter_frac=round(letter_fraction(text), 3), shortcut_hits=detect_shortcuts(text, IR, 4, 2),
                       strategy=strategy_class(text, IR, 4, 1, tok), text=text)
    R["ratio_symbolic_over_english"] = dict(tokens=R["symbolic"]["tokens"] / R["english"]["tokens"], code_points=R["symbolic"]["code_points"] / R["english"]["code_points"])
    R["ratio_warm_over_english"] = dict(tokens=R["warm(english)"]["tokens"] / R["english"]["tokens"], code_points=R["warm(english)"]["code_points"] / R["english"]["code_points"])
    out = ROOT / args.out; json.dump(R, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: (v if not isinstance(v, dict) or "text" not in v else {kk: vv for kk, vv in v.items() if kk != "text"}) for k, v in R.items()}, indent=1, ensure_ascii=False))
    print("wrote", out)

if __name__ == "__main__":
    main()
