"""解码器：按题划分无重叠；可学习的合成任务上 held-out 准确率高、移植基线低；配对 bootstrap。"""
import random
import numpy as np
from cot_compress import decoder

def _synth(n_items=120, n=6, seed=0):
    """合成：think 里明文写 "p<i> <val>"（非复制片段），标签 = 这些值；另加复制自 prompt 的片段（应被掩掉）。"""
    rng = random.Random(seed); rows = []
    for j in range(n_items):
        truth = [rng.choice("01") for _ in range(n)]
        lab2 = "".join(truth); lab1 = "".join(t if i < 2 else "?" for i, t in enumerate(truth)); lab0 = "?" * n
        prompt = f"s1: 1 says 2 & 3 s2: 2 says !3 item {j}".split()
        body = [w for i, t in enumerate(truth) for w in (f"p{i+1}", "is", "one" if t == "1" else "zero")]
        # 假设：p<i> is one/zero 的 3-gram 是可学习特征；再混入复制片段与噪声
        think = prompt[:6] + body + [rng.choice(["hmm", "so", "then", "ok"]) for _ in range(5)]
        for rep in range(2):
            rows.append(dict(id=f"it{j}", think_tokens=list(think), prompt_tokens=prompt, labels=[lab0, lab1, lab2]))
    return rows, n

def test_split_no_overlap_and_grouped_by_item():
    ids = [f"it{i}" for i in range(50)] * 2
    tr, te = decoder.split_items(ids, 0.3, 0)
    assert not (tr & te) and len(tr) + len(te) == 50 and len(te) == 15

def test_decoder_learns_and_transplant_baseline_is_low():
    rows, n = _synth()
    r2 = decoder.run_decoder(rows, 2, n, D=2048, iters=400)
    assert r2["acc_assigned"] >= 0.9 and r2["n_train"] + r2["n_test"] == len(rows)
    r0 = decoder.run_decoder(rows, 0, n, D=2048, iters=50)
    assert r0["acc_all"] == 1.0 and r0["acc_assigned"] is None            # k=0 全 '?'：平凡
    t2 = decoder.run_decoder(decoder.transplant_rows(rows), 2, n, D=2048, iters=400)
    assert t2["acc_assigned"] < 0.75
    bs = decoder.paired_bootstrap(r2["per_item_acc_assigned"], t2["per_item_acc_assigned"])
    assert bs["n"] > 0 and bs["lo"] > 0 and bs["diff"] > 0.15

def test_featurize_masks_copied_spans():
    p = "a b c d e f".split(); t = "a b c d e x y".split()
    x_masked = decoder.featurize(t, p, D=512); x_raw = decoder.featurize(t, p, D=512, mask_copy=False)
    assert x_masked.sum() < x_raw.sum()
    assert abs(np.linalg.norm(x_masked) - 1) < 1e-5
