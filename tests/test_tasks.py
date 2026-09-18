"""任务生成器测试：不依赖 unsloth / GPU。"""
import random
import pytest
import tasks
from tasks import cups, arith

def test_task_registry():
    assert tasks.task_for_key("cups_n6_k12") is cups
    assert tasks.task_for_key("mul_4x4") is arith
    assert tasks.task_for_key("add_6x4") is arith
    with pytest.raises(ValueError):
        tasks.task_for_key("foo_1")

def test_eval_set_deterministic_and_split_disjoint():
    a = tasks.make_eval_set("cups_n6_k12", 20, seed=0)
    b = tasks.make_eval_set("cups_n6_k12", 20, seed=0)
    assert [x["prompt"] for x in a] == [x["prompt"] for x in b]
    c = tasks.make_eval_set("cups_n6_k12", 20, seed=1)
    assert [x["prompt"] for x in a] != [x["prompt"] for x in c]
    tr = [x["prompt"] for _, x in zip(range(200), tasks.iter_train("cups_n6_k12", seed=0))]
    assert not set(tr) & {x["prompt"] for x in a}

def test_cups_op_semantics():
    s = [(1, True), (2, False), (3, True), (4, True)]
    assert cups.apply_op(s, ("swap", 1, 2)) == [(2, False), (1, True), (3, True), (4, True)]  # 朝向跟杯子走
    with pytest.raises(ValueError):
        cups.apply_op(s, ("move", 1, 2))
    s = [(1, True), (2, True), (3, True), (4, True)]
    assert cups.apply_op(s, ("swap", 1, 3)) == [(3, True), (2, True), (1, True), (4, True)]
    assert cups.apply_op(s, ("flip", 2)) == [(1, True), (2, False), (3, True), (4, True)]
    assert s == [(1, True), (2, True), (3, True), (4, True)]  # 不改原状态

def test_cups_answer_consistent_with_resimulation():
    for it in tasks.make_eval_set("cups_n8_k20", 50, seed=3):
        m = it["meta"]
        states = cups.simulate(m["n"], m["ops"])
        assert it["answer"] == cups.state_str(states[-1]) and len(it["answer"].split()) == m["n"]
        assert cups.is_valid(it["answer"])
        assert len(m["states"]) == m["k"] and len(m["states"][-1]) == m["n"]
        assert sorted(c for c, _ in m["states"][-1]) == list(range(1, m["n"] + 1))
        assert f"{m['k']} operations" in it["prompt"] and "final state of all 8 positions" in it["prompt"]
        assert all(op[0] in ("swap", "flip") for op in m["ops"]) and "move" not in it["prompt"]

def test_cups_check_and_lenient():
    gold = "6D 4U 8D 5D 3D 1D 2D 7D"
    assert cups.check("6D 4U 8D 5D 3D 1D 2D 7D", gold) and cups.check(" 6d   4u 8D 5D 3D 1D 2D 7D ", gold)
    assert not cups.check("6D 4U 8D 5D 3D 1D 2D", gold)            # 少一个位置
    assert not cups.check("6D 4U 8D 5D 3D 1D 2D 7U", gold)         # 一个朝向错
    assert not cups.check("6D 4U 8D 5D 3D 1D 7D 2D", gold)         # 顺序错
    assert not cups.is_valid("6-down 4-up") and not cups.is_valid("cup 6 down")
    assert cups.lenient_extract("state: 6D 4U 8D 5D 3D 1D 2D 7D. done") == gold
    assert cups.lenient_extract("first 1U 2U then later 6D 4U 8D") == "6D 4U 8D"
    assert cups.chance("cups_n8_k20") < 1e-6 and cups.chance("cups_n6_k12") < 1e-4

def test_arith_answers():
    for it in tasks.make_eval_set("mul_4x3", 30, seed=0):
        m = it["meta"]
        assert it["answer"] == str(m["x"] * m["y"])
        assert 1000 <= m["x"] <= 9999 and 100 <= m["y"] <= 999
    for it in tasks.make_eval_set("add_6x4", 30, seed=0):
        m = it["meta"]
        assert it["answer"] == str(sum(t * s for t, s in zip(m["terms"], m["signs"])))
        assert m["signs"][0] == 1 and len(m["terms"]) == 6

def test_arith_check():
    assert arith.check("1887257", "1887257")
    assert arith.check("-12", "-12")
    assert arith.check("007", "7")
    assert not arith.check("1,887,257", "1887257")   # 严格：不接受逗号
    assert not arith.is_valid("1.5e6")
    assert arith.lenient_extract("The result is 1,887,257.") == "1887257"
