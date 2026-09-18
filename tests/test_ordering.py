"""v8 生成器：手工小例（格统计）、S 独立暴力核对、两个下限独立实现互检、重编号 / 顺序置换不变、答案规则、划分零重叠。"""
import itertools, random
import pytest
from tasks import ordering as O
import tasks

# ---- 手工小例 ----
# n=4，σ = 0 1 2 3；硬 0<1, 2<3；析取 (1<2)|(3<0) 与 (0<2)|(1<3)（各一边一致、一边不一致）。
#   除 0123 外：链 23 整体在前的 2301 违反第二条（0<2、1<3 都假）；任何交错都违反第一条（1<2 与 3<0 都假）→ 唯一。
HAND = dict(n=4, sigma=[0, 1, 2, 3], hard=[(0, 1), (2, 3)], disj=[((1, 2), (3, 0)), ((0, 2), (1, 3))])

def test_hand_example_counts_and_depth():
    n, hard, disj, sigma = HAND["n"], HAND["hard"], HAND["disj"], HAND["sigma"]
    assert O.count_le_hard(n, hard) == 6                      # 两条链 {0<1},{2<3} 交错：C(4,2)=6
    assert O.count_solutions(n, hard, disj) == 1 and O.solutions(n, hard, disj) == [sigma]
    assert O.max_antichain(n, hard) == 2
    assert O.hamiltonian_path_count(n, O.mention_graph_edges(hard, disj)) == 12     # 被提及图 = K4：4!/2
    assert O.hamiltonian_path_count(4, {frozenset((0, 1)), frozenset((1, 2)), frozenset((2, 3)), frozenset((3, 0))}) == 4
    assert O.hamiltonian_path_count(4, {frozenset(p) for p in itertools.combinations(range(4), 2)}) == 12   # K4：4!/2
    info = O.decision_depth(n, hard, disj)
    # 单独试任一边：{0<1,2<3} 加 3<0 / 1<2 / 0<2 / 1<3 都无环 → 根上无强迫 → S ≥ 1；切第一条：1<2 → 链 0123 全序；3<0 → 2301 使第二条两边皆假 → 冲突 ⇒ S = 1
    assert info["S"] == 1 and info["surviving_disj"] == 2
    assert O.redundancy_rate(n, hard, disj) == 0.0
    steps = O.kahn_trajectory(n, hard, disj, sigma)
    # t=0：硬边闭包下无前驱的是 0、2，单独试"0 先"/"2 先"都不成环 → 就绪 [0,2]；t=1：前缀 0<3 与不一致边 3<0 成环 → 1<2 被迫 → 链 0<1<2<3 → 之后每步单元素
    assert [s["ready"] for s in steps[:n]] == [[0, 2], [1], [2], [3]] and [s["next"] for s in steps[:n]] == sigma
    assert steps[1]["decided"] == [0, 1] and steps[0]["decided"] == []      # 1<2 被迫后 0<2 亦被蕴含 → 两条都算已定
    lb = O.lower_bounds(n, len(disj), steps)
    assert lb["decision_lb"] == 6 and lb["kahn_lb"] == 5 and lb["single_ready_frac"] == 0.75 and lb["kahn_lb_tokens"]["2.5"] == 12.5
    assert O.first_split_position(n, hard, disj, sigma) == 0                         # 起点就绪集 {0,2}
    assert O.first_split_position(4, [(0, 1), (1, 2), (2, 3)], [], [0, 1, 2, 3]) == 4  # 链：始终单元素 → n
    assert O.pairwise_accuracy([0, 1, 2, 3], sigma) == 1.0 and O.pairwise_accuracy([3, 2, 1, 0], sigma) == 0.0 and O.pairwise_accuracy([0, 0, 1, 2], sigma) is None
    les = O.linear_extensions(n, hard); ps = {e: i for i, e in enumerate(sigma)}
    exp = sum(O.pairwise_accuracy(le, sigma) for le in les) / len(les)                 # 与枚举定义一致
    assert O.pairwise_baseline(n, hard, sigma) == pytest.approx(exp)

def brute_min_depth(n, hard, disj, cap=4):
    """独立实现：对所有析取顺序穷举的最小决策树深度（同样的 probe 原语）。"""
    def rec(ch, d):
        after, ch2 = O.probe(n, hard, disj, ch)
        if after is None: return 0
        if O.is_total(n, after): return 0
        if d == cap: return None
        best = None
        for k in range(len(disj)):
            if k in ch2: continue
            worst = 0
            for s in (0, 1):
                t = dict(ch2); t[k] = s; r = rec(t, d + 1)
                if r is None: worst = None; break
                worst = max(worst, r)
            if worst is not None and (best is None or worst + 1 < best): best = worst + 1
        return best
    return rec({}, 0)

def closure_dfs(n, pairs):
    """独立闭包实现：DFS 可达；环 → None。"""
    adj = [[] for _ in range(n)]
    for i, j in pairs: adj[i].append(j)
    reach = []
    for s in range(n):
        seen = set(); st = list(adj[s])
        while st:
            v = st.pop()
            if v in seen: continue
            seen.add(v); st += adj[v]
        if s in seen: return None
        reach.append(seen)
    return reach

def kahn_lb_independent(n, hard, disj, sigma):
    """独立 Kahn 下限：就绪 = 剩余中在（硬 ∪ 已定边 ∪ 前缀）DFS 闭包下无剩余前驱；probing = 单独试 e 排下一个是否成环。"""
    total = 0; ch = {}
    for t in range(n):
        placed, rem = sigma[:t], [e for e in range(n) if e not in sigma[:t]]
        prefix = [(p, r) for p in placed for r in rem]
        _, ch = O.probe(n, list(hard) + prefix, disj, ch)
        base = list(hard) + prefix + [disj[k][s] for k, s in ch.items()]
        reach = closure_dfs(n, base)
        ready = [e for e in rem if not any(e in reach[r] for r in rem if r != e)]
        ready = [e for e in ready if closure_dfs(n, base + [(e, r) for r in rem if r != e]) is not None]
        total += len(ready)
    return total

def test_generated_instances_unique_S_and_bounds():
    rng = random.Random(3)
    for key in ("ord_n8_h4_d5", "ord_n9_h5_d7"):
        p = O.parse_key(key)
        for _ in range(6):
            it = O.generate(rng, key); m = it["meta"]; n = m["n"]
            hard = [tuple(x) for x in m["hard"]]; disj = [tuple(map(tuple, x)) for x in m["disj"]]
            assert O.solutions(n, hard, disj) == [m["sigma"]] and len(hard) == p["h"] and len(disj) == p["d"]
            assert m["S"] >= 1 and m["S"] == brute_min_depth(n, hard, disj)
            pos = {e: i for i, e in enumerate(m["sigma"])}
            for s1, s2 in disj:                                             # 一边一致、一边不一致；不与硬约束重合
                assert (pos[s1[0]] < pos[s1[1]]) != (pos[s2[0]] < pos[s2[1]]) and s1 not in hard and s2 not in hard
            assert m["decision_lb"] == p["d"] + n and m["kahn_lb"] == kahn_lb_independent(n, hard, disj, m["sigma"])
            assert m["kahn_lb"] >= n and m["gen"]["mode"] == "guided" and len(m["gen"]["false_side_span"]) == p["d"]
            assert it["answer"] == " ".join(map(str, m["sigma"])) and it["prompt"].startswith(f"n={n}\nhard: ")
            assert len(m["steps"]) == n + 1 and all(s["next"] in s["ready"] for s in m["steps"][:n])
            assert m["canonical_path"] and m["canonical_path"][-1]["outcome"] == "ok"
            assert O.chance(key, it) == pytest.approx(1 / m["stats"]["n_le_hard"])

def test_canonical_form_relabel_and_order_invariant():
    rng = random.Random(5)
    it = O.generate(rng, "ord_n8_h4_d5"); m = it["meta"]; n = m["n"]
    hard = [tuple(x) for x in m["hard"]]; disj = [tuple(map(tuple, x)) for x in m["disj"]]
    cf = O.canonical_form(n, hard, disj)
    for _ in range(4):
        perm = list(range(n)); rng.shuffle(perm); rl = dict(enumerate(perm))
        h2 = [(rl[i], rl[j]) for i, j in hard]; rng.shuffle(h2)
        d2 = [((rl[a], rl[b]), (rl[c], rl[e])) for (a, b), (c, e) in disj]; d2 = [(s2, s1) if rng.random() < 0.5 else (s1, s2) for s1, s2 in d2]; rng.shuffle(d2)
        assert O.canonical_form(n, h2, d2) == cf
    assert O.canonical_form(n, hard[:-1] + [(hard[-1][1], hard[-1][0])], disj) != cf

def test_answers():
    assert O.check("3 0 5 1 2 4 7 6", "3 0 5 1 2 4 7 6") and O.check(" 3, 0 ,5 1 2 4 7 6", "3 0 5 1 2 4 7 6")
    assert not O.check("3 0 5 1 2 4 7", "3 0 5 1 2 4 7 6") and O.normalize("3 3 0") is None and O.normalize("1 2 3") is None
    assert O.hamming("0 1 2 3", "0 2 1 3") == 2
    assert O.lenient_extract("first 0<1<2 then order: 3 -> 0 -> 2 -> 1 ok", 4) == "3 0 2 1" and O.lenient_extract("no order here", 4) is None
    with pytest.raises(ValueError): O.parse_key("ord_8")

def test_cell_stats_and_decision_rules():
    rng = random.Random(7); items = [O.generate(rng, "ord_n8_h4_d5") for _ in range(6)]
    sp = dict(stop=items[:3], train=items[3:], _gen_stats=dict(tries=1000, unique=10, S_ge_1=4))
    rep = O.cell_stats(sp); d = O.cell_decision(rep)
    assert rep["_generation"]["retention_S_ge1"] == 0.4 and rep["_generation"]["cover_exact_frac"] == 1.0
    assert rep["stop"]["iso_classes_subset_of_train"] is False and set(d["checks"]) >= {"n_le_hard_median_ge_20", "first_split_pos_median_le_n/2", "retention_S_ge1_ge_1pct"}
    assert "single_ready_frac_median_le_0.8" not in d["checks"] and rep["stop"]["first_split_pos_median"] <= 4
    sp["_gen_stats"] = dict(tries=1000, unique=1000, S_ge_1=5)
    assert not O.cell_decision(O.cell_stats(sp))["checks"]["retention_S_ge1_ge_1pct"]

def test_splits_disjoint_and_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DATA_DIR", tmp_path); O._CACHE.clear()
    monkeypatch.setattr(O, "uniform_acceptance_rate", lambda key, seed=0, tries=0: 0.0)
    sp = O.build_splits("ord_n8_h4_d5", seed=0, sizes=dict(train=4, stop=3, report=3, direct=3), log=None)
    O.assert_disjoint(sp); assert [len(sp[k]) for k in ("train", "stop", "report", "direct")] == [4, 3, 3, 3]
    import json; json.dump(sp, open(O.split_path("ord_n8_h4_d5", 0), "w"))
    assert tasks.task_for_key("ord_n8_h4_d5") is O
    assert {x["id"].split("/")[1] for x in tasks.make_eval_set("ord_n8_h4_d5", 3, split="direct")} == {"direct"}
    tr = [x["id"] for _, x in zip(range(8), tasks.iter_train("ord_n8_h4_d5", seed=1))]
    assert all("/train/" in i for i in tr) and len(set(tr)) == 4
    O._CACHE.clear()

# ---- 数据文件 / 10k 实例 ----
import os, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_data_files_unique_solution_and_S():
    """data/ordering/*_seed0.json：四划分零重叠、全体唯一解（DFS 计数）、S 独立暴力核对（每格随机 60 题）、格报告存在。文件缺失则跳过。"""
    files = sorted(f for f in (ROOT / "data/ordering").glob("ord_n*_seed*.json") if not f.name.endswith(".report.json"))
    if not files: pytest.skip("no data files (run scripts/ord_build_splits.py)")
    rng = random.Random(0)
    for fp in files:
        sp = json.load(open(fp)); O.assert_disjoint(sp)
        items = [it for k, v in sp.items() if isinstance(v, list) for it in v]
        for it in items:
            m = it["meta"]; assert O.count_solutions(m["n"], [tuple(x) for x in m["hard"]], [tuple(map(tuple, x)) for x in m["disj"]]) == 1 and m["S"] >= 1
        for it in rng.sample(items, min(60, len(items))):
            m = it["meta"]; assert brute_min_depth(m["n"], [tuple(x) for x in m["hard"]], [tuple(map(tuple, x)) for x in m["disj"]]) == m["S"]
        assert fp.with_suffix(".report.json").exists()

@pytest.mark.skipif(not os.environ.get("ORD_FULL"), reason="set ORD_FULL=1: 10k 实例（多进程）")
def test_generator_10k_instances():
    """10k 实例（四格各 2500，独立 seed 流 'full10k'）：唯一解、S 独立暴力核对、重编号不变。"""
    import multiprocessing as mp
    forms = 0
    for key in O.CELLS:
        with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
            items = pool.map(O.generate_indexed, [(key, "full10k", 0, j) for j in range(2500)], chunksize=8)
        for it in items:
            m = it["meta"]; hard = [tuple(x) for x in m["hard"]]; disj = [tuple(map(tuple, x)) for x in m["disj"]]
            assert O.count_solutions(m["n"], hard, disj) == 1 and brute_min_depth(m["n"], hard, disj) == m["S"] >= 1
            forms += 1
    assert forms == 10000
