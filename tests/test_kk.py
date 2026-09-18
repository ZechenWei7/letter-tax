"""v7 主任务 kk 生成器测试：手工 S=0/1/2 实例、S 的独立暴力验证、唯一性、置换不变、答案判定、划分零重叠。"""
import itertools, random
import pytest
from tasks import kk
import tasks

# ---- 手工实例 ----
# S=0（N=3）：1 说 "!2 & 3"，2 说 "3"，3 说 "!1"。唯一解 (0,1,1)。
#   单元传播从空赋值无法推出任何文字（每条约束都有 ≥2 个未知）；探测 B1=1 → c3 给 B3=0 → c2 给 B2=0 → c1 要求 ¬B2∧B3=1 冲突 ⇒ B1=0 → B3=1 → B2=1 完整。
S0 = (3, [(1, "!x&y", 2, 3), (2, "x", 3, None), (3, "!x", 1, None)], (0, 1, 1))
# S=1（N=4，全 knight）：1 说 3=2，3 说 2=4，4 说 1=3，2 说 1=4。每条都是 3 变量 XOR：任一单个文字都推不出别的，
#   探测任一文字（如 B1=0）后每条约束仍剩 2 个未知 → 无冲突 ⇒ S≥1。切 B1：B1=1 分支探测 B2=0 → B3=0,B4=0 与 c3 冲突 ⇒ B2=1 → 完整；
#   B1=0 分支探测 B2 两个值都冲突 ⇒ 叶。故 S=1。
S1 = (4, [(1, "x=y", 3, 2), (3, "x=y", 2, 4), (4, "x=y", 1, 3), (2, "x=y", 1, 4)], (1, 1, 1, 1))
# S=2（N=8，只作算法测试，数据集不含 S=2）：两份互不引用的 S=1 副本（人 1–4 与 5–8）。任何单次分情况只能解掉一份，另一份仍需一次 ⇒ 每条路径两次切分。
S2 = (8, S1[1] + [(s + 4, f, x + 4, y + 4) for s, f, x, y in S1[1]], S1[2] + S1[2])

def brute_min_depth(n, stmts, depth_cap=4):
    """独立实现：对所有变量顺序穷举的最小决策树深度（同样的推理原语：propagate + probe），不带剪枝/预算短路。"""
    cons = kk._constraints(n, stmts)
    def rec(a, d):
        if all(v >= 0 for v in a): return 0
        if d == depth_cap: return None
        best = None
        for v in range(n):
            if a[v] >= 0: continue
            worst = 0
            for b in (0, 1):
                t = list(a); t[v] = b
                c = kk.probe(cons, t)
                if c is None: continue
                r = rec(c, d + 1)
                if r is None: worst = None; break
                worst = max(worst, r)
            if worst is not None and (best is None or worst + 1 < best): best = worst + 1
        return best
    a0 = kk.probe(cons, [-1] * n)
    return rec(a0, 0)

@pytest.mark.parametrize("inst,expect", [(S0, 0), (S1, 1), (S2, 2)])
def test_hand_instances(inst, expect):
    n, stmts, truth = inst
    assert kk.all_solutions(n, stmts) == [truth]
    info = kk.case_split_depth(n, stmts, max_depth=3, want_tree=True)
    assert info["S"] == expect == brute_min_depth(n, stmts)
    labels = kk.decoder_labels(n, stmts, info["tree"], truth)
    assert len(labels) == expect + 1 and all(len(l) == n for l in labels)
    assert all(l[i] in ("?", str(truth[i])) for l in labels for i in range(n))     # 标签与正解一致
    assert info["oracle_literals"] >= n                                                # 至少写出每个人的值

def test_s0_truncated_search_matches_untruncated():
    n, stmts, _ = S1
    assert kk.case_split_depth(n, stmts, max_depth=0)["S"] is None       # 截断 → None（>0）
    assert kk.case_split_depth(n, stmts, max_depth=1)["S"] == 1

def test_forms_semantics():
    assert kk.eval_form("x!=y", 1, 0) == 1 and kk.eval_form("x=y", 1, 0) == 0
    assert kk.eval_form("!x|y", 1, 0) == 0 and kk.eval_form("!x&y", 0, 1) == 1 and kk.eval_form("x&!y", 1, 0) == 1
    assert kk.eval_form("!x", 0, None) == 1 and kk.eval_form("x", 0, None) == 0
    assert kk.render_expr("x&!y", 3, 7) == "3 & !7" and kk.render_expr("!x", 2, None) == "!2"

def test_generated_instances_unique_S1_refcap_and_bruteforce_S():
    """生成器（均匀采样 + 拒绝）：唯一解、S==1（独立暴力核对）、每人恰一条陈述、被引用 ≤3、不引用自己、x≠y。"""
    rng = random.Random(11)
    for _ in range(12):
        it = kk.generate(rng, "kk_n10_s1")
        m = it["meta"]; n = m["n"]; stmts = [tuple(s) for s in m["stmts"]]
        assert kk.all_solutions(n, stmts) == [tuple(m["truth"])]
        assert m["S"] == 1 == brute_min_depth(n, stmts)
        assert kk.probe(kk._constraints(n, stmts), [-1] * n) is not None and not kk._complete(kk.probe(kk._constraints(n, stmts), [-1] * n))   # 探测解不掉
        assert sorted(s for s, *_ in stmts) == list(range(1, n + 1))
        from collections import Counter
        refs = Counter([x for _, _, x, _ in stmts] + [y for _, _, _, y in stmts if y is not None])
        assert max(refs.values()) <= kk.MAX_REF
        for s, f, x, y in stmts:
            assert x != s and y != s and (y is None or x != y)
            assert (y is None) == (f in kk.UNARY)
        assert it["answer"] == " ".join(map(str, m["truth"])) and kk.check(it["answer"], it["answer"])
        assert it["prompt"].startswith(f"persons 1..{n} (1 = knight, 0 = knave)\n") and it["prompt"].endswith(f"answer: {n} digits for persons 1..{n}")
        assert len(it["prompt"].split("\n")) == n + 2
        assert "knight" in it["prompt_gloss"] and "says" not in it["prompt_gloss"].split("\n")[0]
        assert m["oracle_lb"]["2"] == 2 * m["oracle_literals"] and m["oracle_lb"]["3"] == 3 * m["oracle_literals"]
        assert len(m["decoder_labels"]) == 2 and m["decoder_labels"][1].count("?") < n     # k=1：分情况后的部分分配

def test_canonical_form_permutation_invariant():
    rng = random.Random(3)
    for _ in range(5):
        it = kk.generate(rng, "kk_n10_s1"); n = it["meta"]["n"]; stmts = [tuple(s) for s in it["meta"]["stmts"]]
        cf = kk.canonical_form(n, stmts)
        for _ in range(3):
            perm = list(range(1, n + 1)); rng.shuffle(perm); mp = dict(zip(range(1, n + 1), perm))
            st2 = [(mp[s], f, mp[x], None if y is None else mp[y]) for s, f, x, y in stmts]
            if True:   # 对称形式换操作数顺序也不变
                st2 = [(s, f, y, x) if (f in kk.SYMMETRIC and rng.random() < 0.5) else (s, f, x, y) for s, f, x, y in st2]
            assert kk.canonical_form(n, st2) == cf
        # 改一条陈述 → 规范形变
        s, f, x, y = stmts[0]; alt = [(s, "x|y" if f != "x|y" else "x&y", x, y)] + stmts[1:]
        assert kk.canonical_form(n, alt) != cf

def test_answer_rules():
    assert kk.check(" 1 0  1 ", "1 0 1") and not kk.check("1 0", "1 0 1") and not kk.check("1 0 2", "1 0 2")
    assert kk.normalize("1,0,1") is None and kk.normalize("1 0 1") == "1 0 1"
    assert kk.hamming("1 0 1 1", "1 1 1 0") == 2 and kk.hamming("1 0", "1 0 1") is None
    assert kk.lenient_extract("so the answer is 1 0 1 1 0.") == "1 0 1 1 0"
    assert kk.chance("kk_n10_s1") == 2 ** -10 and kk.parse_key("kk_n12_s1") == dict(n=12, s=1)
    with pytest.raises(ValueError):
        kk.parse_key("kk_10")

def test_splits_disjoint_and_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(kk, "DATA_DIR", tmp_path)
    sp = kk.build_splits("kk_n10_s1", seed=0, sizes=dict(train=6, stop=4, report=4), log=None)
    assert [len(sp[k]) for k in ("train", "stop", "report")] == [6, 4, 4]
    kk.assert_disjoint(sp)
    sp2 = kk.build_splits("kk_n10_s1", seed=0, sizes=dict(train=6, stop=4, report=4), log=None)
    assert [x["prompt"] for x in sp["train"]] == [x["prompt"] for x in sp2["train"]]
    assert sp["stop"][0]["id"] == "kk_n10_s1/stop/0/0"
    rep = kk.split_report(sp)
    assert rep["train"]["distinct_canonical"] == 6 and rep["stop"]["S"] == {1: 4}
    assert not ({x["meta"]["canonical"] for x in sp["train"]} & {x["meta"]["canonical"] for x in sp["report"]})

def test_registry_uses_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(kk, "DATA_DIR", tmp_path); kk._CACHE.clear()
    assert tasks.task_for_key("kk_n10_s1") is kk
    with pytest.raises(FileNotFoundError):
        tasks.make_eval_set("kk_n10_s1", 2)
    kk.load_splits("kk_n10_s1", 0, build=True) if False else None
    sp = kk.build_splits("kk_n10_s1", seed=0, sizes=dict(train=4, stop=3, report=3), log=None)
    import json; json.dump(sp, open(kk.split_path("kk_n10_s1", 0), "w"))
    stop = tasks.make_eval_set("kk_n10_s1", 3, split="stop"); rep = tasks.make_eval_set("kk_n10_s1", 3, split="report")
    assert {x["id"].split("/")[1] for x in stop} == {"stop"} and {x["id"].split("/")[1] for x in rep} == {"report"}
    tr = [x["id"] for _, x in zip(range(8), tasks.iter_train("kk_n10_s1", seed=1))]
    assert all("/train/" in i for i in tr) and len(set(tr)) == 4
    kk._CACHE.clear()
