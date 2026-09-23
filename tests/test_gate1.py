"""第二阶段闸门 1 的纯逻辑单测：打分、篡改构造与判定、McNemar / TOST、判读分区的每个分支。"""
import json, random
import pytest
from tasks import ordering as O
from cot_compress import derivation as D, gate1 as G

ITEMS = json.load(open(O.DATA_DIR / "ord_n8_h4_d5_p2eval_seed1.json"))["p2eval"]


def test_score_gold_exact_all_versions():
    it = ITEMS[13]; st = D.derive(it)
    for v in ("N2", "N2s", "N3"):
        r = G.score_derivation(D.render(st, it, v), it, v, st)
        assert r == dict(parsed=True, exact=True, sound=True)


def test_score_broken_and_wrong_version():
    it = ITEMS[13]; st = D.derive(it); t = D.render(st, it, "N2")
    assert G.score_derivation(t.replace(" since", " sinse", 1), it, "N2", st)["parsed"] is False
    assert G.score_derivation(t, it, "N3", st)["parsed"] is False            # 用错写法解析 → 不可解析


def test_think_of_and_loop():
    assert G.think_of(" a\n b\n</think>\n\n<answer>1</answer>") == " a\n b"
    assert G.loop_flag("\n".join([" x"] * 5)) and not G.loop_flag("\n".join([" x"] * 4))


def test_gold_derivations_never_flag_loop():
    assert not any(G.loop_flag(D.render(D.derive(it), it, "N2")) for it in ITEMS)


def test_tamper_plan_and_texts():
    it = ITEMS[13]; st = D.derive(it); p = G.tamper_plan(st, it)
    assert p == dict(line=5, edge=(3, 2), h=6, m=12)
    c, t = G.tamper_texts(st, it, "N3", p)
    assert c.split("\n")[5].endswith(" 3<2") and t.split("\n")[5].endswith(" 2<3") and c.split("\n")[:5] == t.split("\n")[:5]


def test_continuation_follow():
    r = G.continuation_follow(" since 1<2<3 holds (2<3)|(4<5)\n order 1 2 3 0 4 5 6 7", "N2", (3, 2), "1 2 3 0 4 5 6 7")
    assert r["uses_flip"] and not r["uses_orig"] and r["follows_flip"] and r["answer_order"] == "flip"
    r = G.continuation_follow(" : 3<2 ; (3<2)|(4<5)", "N3", (3, 2), "3 2 0 1 4 5 6 7")
    assert r["uses_orig"] and not r["follows_flip"] and r["answer_order"] == "orig"


def test_mcnemar():
    assert G.mcnemar_p(0, 0) == 1.0
    assert G.mcnemar_p(10, 0) == pytest.approx(2 / 1024)
    assert G.mcnemar_p(5, 5) == 1.0


# ---- 判读分区：1000 个配对键（500 题 × 2 seed），N2 准确率 0.8
KEYS = [(i, s) for s in (1, 2) for i in range(500)]
N2 = {k: int(j < 800) for j, k in enumerate(KEYS)}
def flip(base, n, to):           # 把 base 里前 n 个值为 1-to 的键改成 to
    out = dict(base); c = 0
    for k in KEYS:
        if c >= n: break
        if out[k] == 1 - to: out[k] = to; c += 1
    return out
OKP = dict(N2_acc=0.8, N2_exact=0.9, N2_transplant_acc=0.02)
WORSE = flip(N2, 150, 0)


@pytest.mark.parametrize("name,acc,premise,expect", [
    ("P_fail_acc", dict(N2=N2, N2s=N2, N3m=N2, N3u=N2), dict(OKP, N2_acc=0.4), "P_fail"),
    ("P_fail_exact", dict(N2=N2, N2s=N2, N3m=N2, N3u=N2), dict(OKP, N2_exact=0.7), "P_fail"),
    ("P_fail_transplant", dict(N2=N2, N2s=N2, N3m=N2, N3u=N2), dict(OKP, N2_transplant_acc=0.2), "P_fail"),
    ("R1", dict(N2=N2, N2s=N2, N3m=N2, N3u=N2), OKP, "R1"),
    ("R2a", dict(N2=N2, N2s=WORSE, N3m=WORSE, N3u=WORSE), OKP, "R2a"),
    ("R2b", dict(N2=N2, N2s=N2, N3m=WORSE, N3u=WORSE), OKP, "R2b"),
    ("R2c", dict(N2=N2, N2s=N2, N3m=WORSE, N3u=N2), OKP, "R2c"),
    ("R2_unclear", dict(N2=N2, N2s=flip(N2, 75, 0), N3m=WORSE, N3u=WORSE), OKP, "R2_unclear"),
    ("R3", dict(N2=N2, N2s=N2, N3m=flip(N2, 150, 1), N3u=N2), OKP, "R3"),
    ("R4", dict(N2=N2, N2s=N2, N3m=flip(N2, 45, 0), N3u=N2), OKP, "R4"),
])
def test_classify_branches(name, acc, premise, expect):
    assert G.classify(acc, premise)["reading"] == expect, name


def test_tamper_targets_hold_in_solution_and_coverage():
    n = 0
    for it in ITEMS:
        st = D.derive(it); p = G.tamper_plan(st, it)
        if not p: continue
        n += 1; c, d = p["edge"]; sig = it["meta"]["sigma"]
        assert sig.index(c) < sig.index(d)                                   # 被改的关系在最终解中成立 → 金标答案永远是 "orig"
        assert G.continuation_follow("", "N2", p["edge"], it["answer"])["answer_order"] == "orig"
    assert n == 262
