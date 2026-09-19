"""v8 §2–4：捷径 / 策略检测器、B_warm（ordering 轨迹的策略类分布不变）、解码器（标签不含轨迹信息、按题划分无重叠、可学习）。"""
import random
import pytest
from cot_compress import shortcuts as SC, warm, decoder as DC
from tasks import ordering as O

IR = "n=4\nhard: 0<1 2<3\ndisj: (1<2)|(3<0) (0<2)|(1<3)\nreason inside <think>, then give the order\nanswer: 4 digits, first to last"
ITEM = dict(id="x", prompt=IR, answer="0 1 2 3", meta=dict(n=4, d=2, sigma=[0, 1, 2, 3], hard=[[0, 1], [2, 3]], disj=[[[1, 2], [3, 0]], [[0, 2], [1, 3]]], S=1,
            steps=O.kahn_trajectory(4, [(0, 1), (2, 3)], [((1, 2), (3, 0)), ((0, 2), (1, 3))], [0, 1, 2, 3])))

def test_shortcut_detectors():
    n, d = 4, 2
    assert SC.detect_shortcuts("0<1 2<3 (1<2)|(3<0) (0<2)|(1<3) hence 0 1 2 3", IR, n, d) == ["ir_copy_order"]
    assert SC.detect_shortcuts("there are 4! = 24 orders, check all permutations", IR, n, d) == ["perm_enum"]
    assert "perm_enum" in SC.detect_shortcuts("\n".join(f"{a} {b} {c} {e}" for a, b, c, e in __import__("itertools").permutations(range(4))), IR, n, d)
    assert SC.detect_shortcuts("assignments 00 01 10 11: try each", IR, n, d) == ["assign_enum"]
    assert SC.detect_shortcuts("case 1 ... case 2 ... use a Gray code over sides", IR, n, d) == ["assign_enum"]
    assert SC.detect_shortcuts("case 1: a. case 2: b. case 3: c. case 4: d.", IR, n, d) == ["assign_enum"] and SC.detect_shortcuts("case 1: a. case 2: b.", IR, n, d) == []
    assert SC.detect_shortcuts("0<1 2<3 restated; hence 0 1 2 3", IR, n, d) == []        # 只复述硬约束不算抄写
    assert SC.detect_shortcuts("try both sides; try both again", IR, n, d) == ["assign_enum"]
    gv = "guess 0 1 2 3 check ok;\nguess 2 3 0 1 check fails;\nguess 0 2 1 3 verify fails"
    assert SC.detect_shortcuts(gv, IR, n, d) == ["guess_verify"]                       # 三个不同候选、之间无推理标记
    restate = "The order is 0 1 2 3. check ok.\nLet me verify: 0 1 2 3 holds.\nFinal: 0 1 2 3 valid."
    assert SC.detect_shortcuts(restate, IR, n, d) == []                               # D9：同一顺序的重述 / 复核不算
    reasoned = "try 2 3 0 1 check fails.\nSo 3<0 is impossible, therefore 1<2; then 0 1 2 3 check ok.\nSince that holds, if we swap we get 0 2 1 3 check fails."
    assert SC.detect_shortcuts(reasoned, IR, n, d) == []                              # D9：换候选之间有传播 / 分情况标记
    assert SC.detect_shortcuts(restate, IR, n, d, legacy_gv=True) == ["guess_verify"]   # 原定义并报用
    assert SC.reasoning_markers("so then » x") == 3 and SC.reasoning_markers("plain text 0 1 2") == 0
    natural = "Suppose 3<0. Then 2<3<0<1, so 0<2 is false and 1<3 is false: contradiction. Hence 1<2, giving 0<1<2<3."
    assert SC.detect_shortcuts(natural, IR, n, d) == []
    assert SC.shortcut_rate([natural, gv, restate], [IR, IR, IR], n, d)["any"] == pytest.approx(1 / 3)

def test_strategy_classes():
    n, S = 4, 1
    eng = "Suppose 3<0. Then 2<3<0<1, so contradiction. Hence 1<2, giving 0<1<2<3."
    assert SC.strategy_class(eng, IR, n, S)["strategy_class"] == "english_case"
    sym = "» 3<0 → 2<3<0<1 ⇒ × ⇒ 1<2 ⇒ 0<1<2<3"
    assert SC.strategy_class(sym, IR, n, S)["strategy_class"] == "symbolic_case"
    prop = "since the chain grows, therefore one event remains, so the order follows."   # 无 a<b 断言、有传播词
    assert SC.strategy_class(prop, IR, n, S)["strategy_class"] == "propagation"
    enum = " ".join(f"case {i}: try side {i % 2}" for i in range(8))
    assert SC.strategy_class(enum, IR, n, S)["strategy_class"] == "enumeration"
    assert SC.strategy_class("so we propagate. " + enum, IR, n, S)["strategy_class"] == "mixed_probe_enum"      # enum 且 propagation 标记
    assert SC.strategy_class("try both sides. " + enum, IR, n, S)["strategy_class"] == "mixed_probe_enum"       # enum 且 try-both 标记
    assert SC.strategy_class("so we propagate, try both", IR, n, S)["strategy_class"] != "mixed_probe_enum"     # 非 enum
    path = "0<1, 1<2, 2<3 so 0<1<2<3."
    r = SC.strategy_class(path, IR, n, S); assert r["strategy_class"] == "path_stitching" and r["asserted_on_ir_frac"] == 1.0
    assert SC.strategy_class("hmm 42", IR, n, S)["strategy_class"] == "other"
    dist = SC.class_distribution([SC.strategy_class(t, IR, n, S) for t in (eng, sym, prop)])
    assert dist["english_case"] == pytest.approx(1 / 3)

def _traj(rng, n=4):
    kind = rng.choice(["english_case", "propagation", "enumeration", "path_stitching", "other"])
    if kind == "english_case": return f"Assume {rng.randint(0,3)}<{rng.randint(0,3)}, then we get a cycle, so contradiction. Therefore the order is 0 1 2 3."
    if kind == "propagation": return f"so the chain grows hence one remains thus order 0 1 2 3."
    if kind == "enumeration": return " ".join(f"case {i}: suppose side {i%2}" for i in range(8))
    if kind == "path_stitching": return "0<1, 1<2, 2<3 gives 0<1<2<3."
    return "the events are numbered; nothing else"

def test_warm_preserves_ordering_strategy_distribution():
    rng = random.Random(0); src = [_traj(rng) for _ in range(100)]
    a = [SC.strategy_class(t, IR, 4, 1) for t in src]; b = [SC.strategy_class(warm.warm_text(t), warm.warm_text(IR), 4, 1) for t in src]
    da, db = SC.class_distribution(a), SC.class_distribution(b)
    # english_case → symbolic_case 是变换的定义（字母被删）；其余类不变
    assert da["english_case"] == db["symbolic_case"] and db["english_case"] == 0
    for k in ("propagation", "enumeration", "mixed_probe_enum", "path_stitching", "other"): assert da[k] == db[k], k
    for x, y in zip(a, b): assert (x["case_splits"], x["propagation"], x["branches"]) == (y["case_splits"], y["propagation"], y["branches"])
    for t in src: warm.assert_letter_free(warm.warm_text(t))

def test_parser_and_labels_independent_of_trajectory():
    assert DC.parse_step("we get 0<1<2 and then", 4) == 3 and DC.parse_step("0 -> 3, 3 -> 2", 4) == 3 and DC.parse_step("nothing", 4) is None and DC.parse_step("0<1<2<3<3", 4) == 4
    lab = DC.labels_for(ITEM, 1)
    assert lab == dict(next="1", ready="0100", decided="11") and DC.labels_for(ITEM, 0) == dict(next="0", ready="1010", decided="00") and DC.labels_for(ITEM, 4)["next"] == "?"
    rng = random.Random(0)
    for t in range(5):                                             # 标签只依赖 (题, t)：把轨迹换成随机符号，标签不变
        assert DC.labels_for(dict(ITEM, think="".join(rng.choice("»→×¿ 0123") for _ in range(50))), t) == DC.labels_for(ITEM, t)
    rows = [dict(id="x", think="0<1 then 2", prompt=IR)]
    prow, pr = DC.prefix_rows(rows, {"x": ITEM})
    assert pr["n_prefixes"] == 4 and all(r["labels"] == DC.labels_for(ITEM, r["t"]) for r in prow)

def test_decoder_learns_next_event_on_synthetic():
    rng = random.Random(1); items, rows = {}, []
    for j in range(80):
        sigma = list(range(4)); rng.shuffle(sigma)
        it = dict(id=f"i{j}", prompt=IR, meta=dict(n=4, d=2, sigma=sigma, steps=[dict(t=t, ready=[sigma[t]] if t < 4 else [], next=(sigma[t] if t < 4 else None), decided=[]) for t in range(5)]))
        items[it["id"]] = it
        # 轨迹：写出前 t 个事件的链，再明文说 "next is X"（可学特征），复制片段无关
        for t in (1, 2, 3):
            chain = "<".join(map(str, sigma[:t])); rows.append(dict(id=it["id"], think=f"{chain} so nxt{sigma[t]} ok", prompt=IR))
    prow, pr = DC.prefix_rows(rows, items, fracs=(1.0,))
    assert pr["parse_rate"] > 0.5
    r = DC.run_ordering_decoder(prow, 4, "next", D=1024, iters=300); tr = DC.run_ordering_decoder(DC.transplant_rows(prow), 4, "next", D=1024, iters=300)
    assert r["acc"] >= 0.9 and tr["acc"] < 0.6 and DC.paired_bootstrap(r["per_item_acc"], tr["per_item_acc"])["lo"] > 0.1
    tr_ids, te_ids = DC.split_items([r_["id"] for r_ in prow]); assert not (tr_ids & te_ids)

def test_soundness_completeness_and_cipher():
    sc = DC.soundness_completeness("0<1 and 3<0 hmm 1<2", ITEM)
    assert sc["n_asserted"] == 3 and sc["soundness"] == pytest.approx(2 / 3)
    r = DC.cipher_spearman_warm(["assume assume then so so so contradiction if"] * 5, ["» » → ⇒ ⇒ ⇒ × ¿"] * 5)
    assert r["n_pairs"] >= 4 and r["spearman"] is not None and r["spearman"] > 0.5
