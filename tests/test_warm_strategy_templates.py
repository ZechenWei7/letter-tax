"""v7 §2–4：模板检测器、策略类检测器、B_warm 变换（策略类分布不变性、无字母、符号单 token）。"""
import random, re
import pytest
from cot_compress import warm, strategy, templates

IR = "persons 1..4 (1 = knight, 0 = knave)\ns1: 1 says 3 = 2\ns2: 3 says 2 = 4\ns3: 4 says 1 = 3\ns4: 2 says 1 = 4\nanswer: 4 digits for persons 1..4"

def _make_traj(rng, n=4):
    """合成 A 风格轨迹：随机策略类，用控制词与非控制词混排。"""
    kind = rng.choice(["case_split", "propagate_only", "enumerate", "dump_verify", "other"])
    filler = lambda: " ".join(rng.choice(["we", "see", "that", "person", "the", "statement", "means", "look", "at", "value"]) for _ in range(rng.randint(2, 6)))
    lines = []
    if kind == "case_split":
        for _ in range(rng.randint(1, 3)):
            lines.append(f"{filler()} Assume person {rng.randint(1, n)} is a knight, then {filler()} so {filler()}.")
        lines.append(f"{filler()} contradiction. Therefore {filler()}.")
    elif kind == "propagate_only":
        for _ in range(rng.randint(1, 4)):
            lines.append(f"{filler()} so person {rng.randint(1, n)} is a knave, hence {filler()}.")
    elif kind == "enumerate":
        lines.append("Try all 2^4 = 16 assignments: " + " ".join(format(i, "04b") for i in range(16)))
    elif kind == "dump_verify":
        lines.append(f"candidate 1 0 1 1: check s1 ... consistent. candidate 1 1 1 1: consistent. {filler()}")
    else:
        lines.append(f"{filler()}. {filler()} = 3 & !2; {filler()}")
    return "\n".join(lines), kind

def test_warm_map_symbols_letter_free_and_distinct_roles():
    for w, s in warm.WARM_MAP.items():
        warm.assert_letter_free(s)
    for role, syms in strategy.ROLE_SYMBOLS.items():
        for other, syms2 in strategy.ROLE_SYMBOLS.items():
            if other != role: assert not (syms & syms2), (role, other)
    assert set(w for words in strategy.ROLES.values() for w in words) <= set(warm.CONTROL)

def test_warm_text_rules():
    t = "Assume person 3 is a knight, then 3 & !7 holds; otherwise contradiction.\nSo the answer: 1 0 1"
    w = warm.warm_text(t)
    assert w == "» 3 1, → 3 & !7 ; « ×.\n⇒ : 1 0 1"
    warm.assert_letter_free(w)
    assert warm.warm_text("IF If if") == "¿ ¿ ¿" and warm.warm_text("don't know 42 → x") == "42 →"

def test_warm_preserves_strategy_class_distribution_on_100_trajs():
    rng = random.Random(0)
    src, kinds = [], []
    for _ in range(100):
        t, k = _make_traj(rng); src.append(t); kinds.append(k)
    a_src = [strategy.annotate(t, IR, 4) for t in src]
    a_warm = [strategy.annotate(warm.warm_text(t), warm.warm_text(IR), 4) for t in src]
    assert strategy.class_distribution(a_src) == strategy.class_distribution(a_warm)
    for x, y in zip(a_src, a_warm):
        assert (x["case_splits"], x["propagation"], x["contradiction"], x["consistent"], x["enumeration"], x["dump_verify"]) == \
               (y["case_splits"], y["propagation"], y["contradiction"], y["consistent"], y["enumeration"], y["dump_verify"])
    dist = strategy.class_distribution(a_src)
    assert dist["case_split"] > 0 and dist["enumerate"] > 0 and dist["dump_verify"] > 0     # 合成集覆盖各类
    for k, t in zip(kinds, src):                                                             # 合成类 = 检测类（检测器对合成集自洽）
        assert strategy.annotate(t, IR, 4)["strategy_class"] == k

def test_sft_rows():
    rows = [dict(id="a", prompt=IR, think="Assume 1 is a knight then 2 = 3 so done", answer="1 1 1 1", correct=True),
            dict(id="b", prompt=IR, think="x", answer="0 0 0 0", correct=False)]
    out = warm.make_sft_rows(rows)
    assert len(out) == 1 and out[0]["completion"].endswith("</think>\n\n<answer>1 1 1 1</answer>") and out[0]["think_warm"] == "» 1 1 → 2 = 3 ⇒"

def test_copy_rate_and_binary_strings():
    p = "s1: 1 says 3 = 2 s2: 3 says 2 = 4".split()
    t = "well s1: 1 says 3 = 2 and so on".split()
    assert strategy.copy_rate(t, p) == pytest.approx(6 / len(t))
    assert strategy.copy_rate([], p) is None
    assert strategy.binary_strings("try 1011 and 1 0 1 1 and 10111 and 101", 4) == ["1011", "1 0 1 1"]

def test_template_detectors():
    n = 4
    assert templates.detect("we run DPLL with unit propagation", IR, n) == ["dpll_cdcl"]
    assert templates.detect("branch on 1, if it fails we backtrack", IR, n) == ["dpll_cdcl"]
    assert "bitmask_enum" in templates.detect("for mask in range(16): check", IR, n)
    assert "bitmask_enum" in templates.detect("there are 2^4 assignments", IR, n)
    assert "bitmask_enum" in templates.detect(" ".join(format(i, "04b") for i in range(8)), IR, n)
    assert "bitmask_enum" not in templates.detect(" ".join(format(i, "04b") for i in range(7)), IR, n)
    eng = " ".join(f"Assume person {i} is a knight, then person {i+1} is a knave." for i in range(1, 4))
    assert templates.detect(eng, IR, n) == ["english_assume"]
    assert templates.detect(eng.replace("Assume", "Since"), IR, n) == []
    copy = "1 says 3 = 2\n3 says 2 = 4\n4 says 1 = 3\nfrom this: 1 1 1 1"
    assert templates.detect(copy, IR, n) == ["ir_copy_assign"]
    assert templates.detect("1 says 3 = 2\n3 says 2 = 4\n" + "x" * 300 + " no assignment", IR, n) == []
    natural = "Suppose 1 is a knight. Then 3 = 2. Since 4 says 1 = 3 ... so 1 1 1 1"
    assert templates.detect(natural, IR, n) == []
    hr = templates.hit_rate([eng, natural, copy], [IR] * 3, n)
    assert hr["any"] == pytest.approx(2 / 3) and hr["by_type"]["english_assume"] == pytest.approx(1 / 3)
