"""v4 奖励：排序性质穷举、违规检测、侧通道 L。"""
import itertools, pytest
import cot_compress  # noqa
from transformers import AutoTokenizer
from cot_compress.rewards import score, violations, make_reward, SIDE, HARD, SOFT, is_hard
from tasks import arith, cups

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("unsloth/Qwen3-1.7B")

def _effective(c_true, vio, L, M, lam):
    """与 make_reward 相同的 c/v 折算：hard → c=0；v = 任一违规。"""
    c = 0.0 if is_hard(vio) else float(c_true); v = 1.0 if vio else 0.0
    return score(c, v, L, M, lam)

def test_score_ordering_exhaustive():
    M = 1000.0
    Ls = [0, 1, 100, 500, 900, 1000, 1800, 3072, 10**6]
    vios = [[], ["multi_answer"], ["answer_too_long"], ["multi_answer", "answer_too_long"],
            ["text_after_think"], ["second_think_tag"], ["text_after_think", "multi_answer"]]
    for lam in (0.1, 0.5, 1.0, 3.0):
        correct = [score(1, v, L, M, lam) for v in (0, 1) for L in Ls]
        wrong = [score(0, v, L, M, lam) for v in (0, 1) for L in Ls]
        assert min(correct) >= 0.5 - 1e-9, (lam, min(correct))
        assert max(wrong) <= 0.0 + 1e-9
        assert min(correct) > max(wrong)
        # hard 分支：答对与否都 = −5 = 全局最小；严格低于无违规错误（0）与任何正确样本
        hard = [_effective(ct, vio, L, M, lam) for ct in (0, 1) for vio in vios if is_hard(vio) for L in Ls]
        nonhard_wrong_clean = [_effective(0, [], L, M, lam) for L in Ls]
        nonhard_correct = [_effective(1, vio, L, M, lam) for vio in vios if not is_hard(vio) for L in Ls]
        assert all(h == pytest.approx(-5.0) for h in hard)
        assert max(hard) < min(nonhard_wrong_clean) == 0.0 and max(hard) < min(nonhard_correct)
        # 错 + soft 违规也是 −5（"错且格式差"同级），不低于 hard
        assert all(_effective(0, vio, L, M, lam) == pytest.approx(-5.0) for vio in vios if vio and not is_hard(vio) for L in Ls)
        # 对 + soft：只扣 0.5
        assert _effective(1, ["multi_answer"], 0, M, lam) == pytest.approx(9.5)
    assert score(1, 0, 0, M, 0.5) == pytest.approx(10.0) and score(0, 1, 0, M, 0.5) == pytest.approx(-5.0)
    assert score(1, 0, 1000, M, 0.5) == pytest.approx(5.0)          # 1 − 0.5
    assert score(1, 0, 10**6, M, 0.5) == pytest.approx(1.0)         # 饱和 0.9
    assert score(1, 1, 10**6, M, 0.5) == pytest.approx(0.5)
    assert score(1, 0, 5000, M, 0.0) == pytest.approx(10.0)         # 臂 D：无长度项
    # 长度单调：同 c=1、v 下 L 越长 r 越小（饱和后持平）
    for v in (0, 1):
        rs = [score(1, v, L, M, 0.5) for L in Ls]
        assert all(a >= b - 1e-12 for a, b in zip(rs, rs[1:]))

def test_violations(tok):
    ok = "work\n</think>\n\n<answer>42</answer>"                       # <think> 已预填在 prompt，完成段从 think 体开始
    assert violations(ok, arith, tok) == []
    assert "no_answer" in violations("x</think>\n\nso 42", arith, tok)
    assert "multi_answer" in violations("x</think>\n\n<answer>42</answer> <answer>43</answer>", arith, tok)          # </think> 后不同值
    assert violations("<answer>42</answer> ok</think>\n\n<answer>42</answer>", arith, tok) == []                     # 思考内草稿忽略
    assert violations("<answer>41</answer> hmm <answer>7</answer>\n</think>\n\n<answer>42</answer>", arith, tok) == []   # 思考内不同值草稿也忽略
    assert violations("x</think>\n\n<answer>42</answer> <answer>42</answer>", arith, tok) == []                      # </think> 后同值重复：不算
    assert "text_after_think" in violations("x</think>\nso the answer is\n<answer>42</answer>", arith, tok)
    assert HARD == {"text_after_think", "second_think_tag"} and SOFT == {"multi_answer", "answer_too_long"}
    assert is_hard(["text_after_think"]) and not is_hard(["multi_answer", "answer_too_long"])
    assert "second_think_tag" in violations("x</think>\n<think>y</think>\n\n<answer>42</answer>", arith, tok)
    assert "second_think_tag" in violations("<think>x</think>\n\n<answer>42</answer>", arith, tok)    # 完成段里的 <think> 也是第二个
    assert "answer_too_long" in violations("x</think>\n\n<answer>42</answer> " + "blah " * 30, arith, tok)
    assert "bad_answer" in violations("x</think>\n\n<answer>forty two</answer>", arith, tok)
    assert violations("x</think>\n\n<answer>6D 4U 8D 5D 3D 1D 2D 7D</answer>", cups, tok) == []
    assert len(tok("<answer>6D 4U 8D 5D 3D 1D 2D 7D</answer>", add_special_tokens=False)["input_ids"]) <= 32                 # n=8 终态在 32 token 内（实测 29）

def test_make_reward_and_side_channel(tok):
    r = make_reward(arith, tok, lam=0.5, M=1000, cap=3072, force_start=3040)
    good = "1 " * 50 + "</think>\n\n<answer>42</answer>"
    long_ = "1 " * 1500 + "</think>\n\n<answer>42</answer>"
    wrong = "x</think>\n\n<answer>41</answer>"
    bad = "x</think>\nhmm\n<answer>42</answer>"
    out = r(prompts=["p"] * 4, completions=[good, long_, wrong, bad], answer=["42"] * 4)
    assert out[0] > out[1] >= 0.5 and out[2] == 0.0 and out[3] == pytest.approx(-5.0)      # bad = text_after_think（hard）：答对也 −5
    esc = "</think>\n\nThe product is 42\n\n<answer>42</answer>"                            # 第 0 步 </think> 逃逸
    soft = "x <answer>42</answer> ok\n</think>\n\n<answer>42</answer> " + "w " * 30       # 思考内同值预演 + 答案区 >16 token（仅 soft）
    assert r(prompts=["p"], completions=["x</think>\n\n<answer>42</answer> <answer>42</answer>"], answer=["42"])[0] > 9.0   # </think> 后同值重复：无违规
    assert r(prompts=["p"], completions=["x</think>\n\n<answer>41</answer> <answer>42</answer>"], answer=["42"])[0] == pytest.approx(-5.0)   # 不同值：multi_answer（soft）且答案不明 → c=0 → −5
    o2 = r(prompts=["p", "p"], completions=[esc, soft], answer=["42", "42"])
    assert o2[0] == pytest.approx(-5.0) and 9.0 < o2[1] < 10.0
    # 侧通道：C_rand / D 给 L
    SIDE["L"] = [0, 0]
    rD = make_reward(arith, tok, lam=0.5, M=1000, cap=3072, force_start=3040, length_term=False)
    outD = rD(prompts=["p", "p"], completions=["42</answer>", "41</answer>"], answer=["42", "42"])
    assert outD[0] == pytest.approx(10.0) and outD[1] == pytest.approx(0.0) and SIDE["L"] is None


def test_eval_row_uses_v4_rules(tok):
    from cot_compress.evaluate import _row, summarize
    it = dict(id="x", answer="42", prompt="p", meta=dict(key="mul_3x3"))
    esc = _row(tok, arith, it, "think", "</think>\n\nThe product is 42\n\n<answer>42</answer>", None, cap=256)
    assert not esc["correct"] and not esc["correct_clean"] and "text_after_think" in esc["violations"] and not esc["format_ok"]   # hard → c=0
    soft = _row(tok, arith, it, "think", "6 * 7 = 42\n</think>\n\n<answer>42</answer> " + "w " * 30, None, cap=256)
    assert soft["correct"] and not soft["correct_clean"] and soft["violations"] == ["answer_too_long"]
    ok = _row(tok, arith, it, "think", "6 * 7 = 42\n</think>\n\n<answer>42</answer>", None, cap=256)
    assert ok["correct"] and ok["correct_clean"] and ok["format_ok"]
    s = summarize([dict(esc, _sec_total=1, _peak_gib=0), dict(ok, _sec_total=1, _peak_gib=0), dict(soft, _sec_total=1, _peak_gib=0)])
    assert s["acc"] == pytest.approx(2 / 3) and s["acc_clean"] == pytest.approx(1 / 3) and s["viol_rate"] == pytest.approx(2 / 3)
