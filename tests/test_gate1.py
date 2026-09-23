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


def test_sound_requires_final_order():
    it = ITEMS[13]; st = D.derive(it); t = D.render(st, it, "N2")
    no_order = "\n".join(t.split("\n")[:-1])                          # 去掉 order 行：每步仍可推出，但没有给出最终顺序
    r = G.score_derivation(no_order, it, "N2", st)
    assert r["parsed"] and not r["exact"] and not r["sound"]
    wrong_order = t.rsplit(" order ", 1)[0] + " order " + " ".join(reversed(it["answer"].split()))
    assert G.score_derivation(wrong_order, it, "N2", st)["sound"] is False


# ---- 判读分区：500 题 × 2 seed。N2 在两个 seed 下都是前 400 题对（准确率 0.8）
ITEMS_N, SEEDS = 500, (1, 2)
N2 = {(i, s): int(i < 400) for i in range(ITEMS_N) for s in SEEDS}
def mod(base, per_seed):
    """per_seed = {seed: (把前 n 个"对"改成错的数量 n_down, 把前 m 个"错"改成对的数量 m_up)}"""
    out = dict(base)
    for s, (down, up) in per_seed.items():
        ones = [i for i in range(ITEMS_N) if base[(i, s)] == 1][:down]; zeros = [i for i in range(ITEMS_N) if base[(i, s)] == 0][:up]
        for i in ones: out[(i, s)] = 0
        for i in zeros: out[(i, s)] = 1
    return out
WORSE = mod(N2, {1: (75, 0), 2: (75, 0)})           # −15pp，两个 seed 同号
BETTER = mod(N2, {1: (0, 75), 2: (0, 75)})          # +15pp，两个 seed 同号
MIXED_W = mod(N2, {1: (150, 0), 2: (0, 30)})        # 合并 −12pp、显著；seed 1 为负、seed 2 为正
MIXED_B = mod(N2, {1: (0, 100), 2: (25, 0)})        # 合并 +7.5pp、显著；seed 1 为正、seed 2 为负
OKP = dict(N2_acc=0.8, N2_sound=0.9, N2_transplant_acc=0.02, N3m_transplant_acc=0.02)


def test_compare_bootstrap_basics():
    c = G.compare(N2, N2)
    assert c["diff"] == 0 and c["equivalent"] and not c["sig_worse"] and c["ci90"] == (0.0, 0.0)
    c = G.compare(N2, WORSE)
    assert abs(c["diff"] + 0.15) < 1e-12 and c["sig_worse"] and not c["equivalent"] and c["ci95"][1] < 0
    assert c["per_seed_diff"] == {1: -0.15, 2: -0.15} and c["seeds_same_sign"]
    c = G.compare(N2, MIXED_W)
    assert c["sig_worse"] and not c["seeds_same_sign"] and c["per_seed_diff"][1] < 0 < c["per_seed_diff"][2]
    small = mod(N2, {1: (10, 0), 2: (10, 0)})                        # −2pp：区间不含 0，但 |点估计| < 5pp → 不算差异
    c = G.compare(N2, small)
    assert c["ci95"][1] < 0 and not c["sig_worse"]
    assert G.compare(N2, WORSE) == G.compare(N2, WORSE)             # bootstrap seed 固定 → 可复现


@pytest.mark.parametrize("name,acc,premise,expect", [
    ("P_fail_acc", dict(N2=N2, N2s=N2, N3m=N2, N3u=N2), dict(OKP, N2_acc=0.4), "P_fail"),
    ("P_fail_sound", dict(N2=N2, N2s=N2, N3m=N2, N3u=N2), dict(OKP, N2_sound=0.7), "P_fail"),
    ("P_fail_transplant", dict(N2=N2, N2s=N2, N3m=N2, N3u=N2), dict(OKP, N2_transplant_acc=0.2), "P_fail"),
    ("R1", dict(N2=N2, N2s=N2, N3m=N2, N3u=N2), OKP, "R1"),
    ("R1_blocked_by_N3_transplant", dict(N2=N2, N2s=N2, N3m=N2, N3u=N2), dict(OKP, N3m_transplant_acc=0.3), "R4"),
    ("R2a", dict(N2=N2, N2s=WORSE, N3m=WORSE, N3u=WORSE), OKP, "R2a"),
    ("R2b", dict(N2=N2, N2s=N2, N3m=WORSE, N3u=WORSE), OKP, "R2b"),
    ("R2c", dict(N2=N2, N2s=N2, N3m=WORSE, N3u=N2), OKP, "R2c"),
    ("R2_unclear", dict(N2=N2, N2s=mod(N2, {1: (40, 0), 2: (40, 0)}), N3m=WORSE, N3u=WORSE), OKP, "R2_unclear"),
    ("R2_blocked_by_seed_sign", dict(N2=N2, N2s=N2, N3m=MIXED_W, N3u=N2), OKP, "R4"),
    ("R3", dict(N2=N2, N2s=N2, N3m=BETTER, N3u=N2), OKP, "R3"),
    ("R3_blocked_by_seed_sign", dict(N2=N2, N2s=N2, N3m=MIXED_B, N3u=N2), OKP, "R4"),
    ("R4_neither_equiv_nor_diff", dict(N2=N2, N2s=N2, N3m=mod(N2, {1: (20, 0), 2: (20, 0)}), N3u=N2), OKP, "R4"),   # −4pp：90% 区间越过 −5pp、|点估计| < 5pp
])
def test_classify_branches(name, acc, premise, expect):
    r = G.classify(acc, premise)
    assert r["reading"] == expect, (name, r["reading"], r.get("notes"), {k: (round(v["diff"], 3), v["ci90"], v["ci95"]) for k, v in r["comparisons"].items()})


def test_tamper_targets_hold_in_solution_and_coverage():
    n = 0
    for it in ITEMS:
        st = D.derive(it); p = G.tamper_plan(st, it)
        if not p: continue
        n += 1; c, d = p["edge"]; sig = it["meta"]["sigma"]
        assert sig.index(c) < sig.index(d)                                   # 被改的关系在最终解中成立 → 金标答案永远是 "orig"
        assert G.continuation_follow("", "N2", p["edge"], it["answer"])["answer_order"] == "orig"
    assert n == 262
