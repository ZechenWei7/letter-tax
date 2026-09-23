"""第二阶段（不在 OSF 注册的 stage-1 协议之内）：把每题的求解过程渲染成同一段推导的四个版本。

  N1   完整英文句子
  N2   简洁英文：连接词 since / not / forces / holds / suppose / contradiction / so / order
  N2s  乱词对照：N2 的连接词一一换成无关的常见英文词（since→table …）
  N3   纯符号：连接词换成白名单里的 ASCII 标点（since→: …）；全文无字母 token
N2 / N2s / N3 出自同一模板，除连接词外逐 token 相同（每个连接词在 Qwen3 分词器下是带前导空格的单 token，
且模板保证连接词前后都是空格分隔，分词不会跨连接词合并；scripts/phase2_render_check.py 对全部题目逐条核对）。

求解过程 = stage-1 的同一求解器（tasks.ordering：传递闭包 + 单元规则传播，failed-disjunct 单边试探，
规范路径"最低编号未决析取优先、先第一边"），这里只是逐步记录。记录版求解器的终态与 tasks.ordering.probe /
canonical_path 逐题断言一致；不修改 tasks/ordering.py。

步骤（steps）——渲染与往返测试的公共表示：
  force   k, dead, chain      "since b<…<a  not a<b  forces c<d"   链证明 b 在 a 之前 → 该析取的 dead 边不成立 → 另一边被迫
  sat     k, side, chain      "since a<…<b  holds (a<b)|(c<d)"     链证明某一边已被蕴含 → 该析取已满足
  assume  k, side             "suppose a<b"                         决策（规范路径：最低编号未决析取、先第一边）
  cycle   chain               "contradiction a<b<…<a"              冲突：闭包出环
  both    k, chain0, chain1   "contradiction (a<b)|(c<d) since … since …"   冲突：析取两边都被否定
  back    k, side             "so c<d"                              冲突之后：弹出最近一个假设，其析取的另一边被迫
  order   sigma               "order 3 0 7 …"                       全序
每个链的每一环都是当时可用的显式约束（硬约束或已定析取边），verify() 独立重放检查。
"""
from __future__ import annotations
import re
from collections import deque

from tasks.ordering import closure, _implied, probe, is_total, canonical_path

CONNECTIVES = ("since", "not", "forces", "holds", "suppose", "contradiction", "so", "order")
MAPS = {
    "N2":  {c: c for c in CONNECTIVES},
    "N2s": {"since": "table", "not": "green", "forces": "river", "holds": "paper",
            "suppose": "window", "contradiction": "music", "so": "garden", "order": "yellow"},
    "N3":  {"since": ":", "not": "!", "forces": "=>", "holds": ";",
            "suppose": "?", "contradiction": "#", "so": "=", "order": "@"},
}
VERSIONS = ("N1", "N2", "N2s", "N3")
MAX_REFUTE_DEPTH = 4


class ProbeFired(Exception):
    """failed-disjunct 单边试探在传播不动点之后仍发现冲突（理论上不应发生：加边 a<b 成环 ⇔ 闭包已蕴含 b<a ⇔ 单元规则已处理）。"""


def _inst(item):
    m = item["meta"]
    return m["n"], [tuple(p) for p in m["hard"]], [(tuple(a), tuple(b)) for a, b in m["disj"]], list(m["sigma"])


def _edges(hard, disj, ch):
    return list(hard) + [disj[k][s] for k, s in sorted(ch.items())]


def _path(n, edges, x, y):
    """边图上 x → y 的最短路（邻接表排序 → 确定性）。"""
    adj = {i: [] for i in range(n)}
    for a, b in edges: adj[a].append(b)
    for v in adj: adj[v].sort()
    prev = {x: None}; q = deque([x])
    while q:
        u = q.popleft()
        if u == y and u != x: break
        for v in adj[u]:
            if v not in prev: prev[v] = u; q.append(v)
    assert y in prev and x != y, (x, y)
    p = [y]
    while p[-1] != x: p.append(prev[p[-1]])
    return p[::-1]


def _cycle(n, edges):
    for a, b in sorted(set(edges)):
        try: return [a] + _path(n, edges, b, a)
        except AssertionError: continue
    raise AssertionError("no cycle found")


def _trace_propagate(n, hard, disj, ch, ev):
    """逐句照搬 tasks.ordering.propagate（同一遍内用遍首闭包），记录每次单元规则。"""
    ch = dict(ch)
    while True:
        E = _edges(hard, disj, ch)
        after = closure(n, E)
        if after is None:
            ev.append(dict(t="cycle", chain=_cycle(n, E))); return None, ch
        changed = False
        for k, ((a, b), (c, d)) in enumerate(disj):
            if k in ch: continue
            f1 = _implied(after, b, a); f2 = _implied(after, d, c)
            t1 = _implied(after, a, b); t2 = _implied(after, c, d)
            if f1 and f2:
                ev.append(dict(t="both", k=k, chain0=_path(n, E, b, a), chain1=_path(n, E, d, c))); return None, ch
            if t1 or t2:
                s = 0 if t1 else 1; x, y = disj[k][s]
                ch[k] = s; changed = True
                ev.append(dict(t="sat", k=k, side=s, chain=_path(n, E, x, y))); continue
            if f1:
                ch[k] = 1; changed = True; ev.append(dict(t="force", k=k, dead=0, chain=_path(n, E, b, a)))
            elif f2:
                ch[k] = 0; changed = True; ev.append(dict(t="force", k=k, dead=1, chain=_path(n, E, d, c)))
        if not changed: return after, ch


def _trace_probe(n, hard, disj, ch, ev):
    """照搬 tasks.ordering.probe；单边试探若在不动点后仍触发则抛 ProbeFired（实测计数）。"""
    after, ch = _trace_propagate(n, hard, disj, ch, ev)
    if after is None: return None, ch
    base = _edges(hard, disj, ch)
    for k in range(len(disj)):
        if k in ch: continue
        for s in (0, 1):
            if closure(n, base + [disj[k][s]]) is None:
                raise ProbeFired((k, s))
    return after, ch


def _close(n, hard, disj, ch, depth):
    """在当前（已传播、无冲突）的假设下，找一条以冲突结束的推导，关闭最内层的假设。"""
    if depth <= 0: return None
    for k2 in sorted(i for i in range(len(disj)) if i not in ch):
        r0 = _refute(n, hard, disj, ch, k2, 0, depth - 1)
        if r0 is None: continue
        ev = r0["events"] + [dict(t="back", k=k2, side=1)]
        a1, c3 = _trace_probe(n, hard, disj, {**ch, k2: 1}, ev)
        if a1 is None: return ev
        r = _close(n, hard, disj, c3, depth - 1)
        if r is not None: return ev + r
    return None


def _refute(n, hard, disj, ch, k, s, depth):
    ev = [dict(t="assume", k=k, side=s)]
    a, c2 = _trace_probe(n, hard, disj, {**ch, k: s}, ev)
    if a is None: return dict(events=ev, kind="conflict")
    r = _close(n, hard, disj, c2, depth)
    if r is None: return None
    return dict(events=ev + r, kind="refuted_deeper")


def _topo(n, after):
    return sorted(range(n), key=lambda i: -bin(after[i]).count("1"))


def derive(item) -> list[dict]:
    """规范路径的逐步记录。与 tasks.ordering.probe / canonical_path 逐题断言一致。"""
    n, hard, disj, sigma = _inst(item)
    pos = {e: i for i, e in enumerate(sigma)}
    ev = []
    after, ch = _trace_probe(n, hard, disj, {}, ev)
    ref_after, ref_ch = probe(n, hard, disj, {})
    assert after == ref_after and ch == ref_ch, "root probe mismatch"
    decisions = []
    while not is_total(n, after):
        k = min(i for i in range(len(disj)) if i not in ch)
        a, b = disj[k][0]
        if pos[a] < pos[b]:
            ev.append(dict(t="assume", k=k, side=0)); decisions.append(dict(disj=k, side=0, outcome="ok"))
            after, ch = _trace_probe(n, hard, disj, {**ch, k: 0}, ev)
        else:
            r = _refute(n, hard, disj, ch, k, 0, MAX_REFUTE_DEPTH)
            assert r is not None, "side 0 not refutable within depth"
            ev.extend(r["events"]); ev.append(dict(t="back", k=k, side=1))
            decisions += [dict(disj=k, side=0, outcome=r["kind"]), dict(disj=k, side=1, outcome="ok")]
            after, ch = _trace_probe(n, hard, disj, {**ch, k: 1}, ev)
        assert after is not None
    assert decisions == canonical_path(n, hard, disj, sigma), "canonical_path mismatch"
    order = _topo(n, after)
    assert order == sigma, "order mismatch"
    ev.append(dict(t="order", sigma=order))
    return ev


# ---------------------------------------------------------------- rendering
def _e(x): return f"{x[0]}<{x[1]}"
def _ch(c): return "<".join(map(str, c))
def _dj(d): return f"({d[0][0]}<{d[0][1]})|({d[1][0]}<{d[1][1]})"


def render(steps, item, version) -> str:
    n, hard, disj, sigma = _inst(item)
    if version == "N1": return _render_n1(steps, disj)
    C = MAPS[version]; out = []
    for s in steps:
        t = s["t"]
        if t == "force":
            dead = disj[s["k"]][s["dead"]]; forced = disj[s["k"]][1 - s["dead"]]
            out.append(f" {C['since']} {_ch(s['chain'])} {C['not']} {_e(dead)} {C['forces']} {_e(forced)}")
        elif t == "sat":
            out.append(f" {C['since']} {_ch(s['chain'])} {C['holds']} {_dj(disj[s['k']])}")
        elif t == "assume":
            out.append(f" {C['suppose']} {_e(disj[s['k']][s['side']])}")
        elif t == "cycle":
            out.append(f" {C['contradiction']} {_ch(s['chain'])}")
        elif t == "both":
            out.append(f" {C['contradiction']} {_dj(disj[s['k']])} {C['since']} {_ch(s['chain0'])} {C['since']} {_ch(s['chain1'])}")
        elif t == "back":
            out.append(f" {C['so']} {_e(disj[s['k']][s['side']])}")
        elif t == "order":
            out.append(f" {C['order']} " + " ".join(map(str, s["sigma"])))
        else: raise ValueError(t)
    return "\n".join(out)


def _link(x, y): return f"{x} comes before {y}"
def _chain_en(c):
    links = [_link(c[i], c[i + 1]) for i in range(len(c) - 1)]
    return links[0] if len(links) == 1 else ", ".join(links[:-1]) + " and " + links[-1]
def _dj_en(d): return f"{d[0][0]} comes before {d[0][1]} or {d[1][0]} comes before {d[1][1]}"


def _render_n1(steps, disj):
    out = []
    for s in steps:
        t = s["t"]
        if t == "force":
            (a, b), (c, d) = disj[s["k"]][s["dead"]], disj[s["k"]][1 - s["dead"]]
            out.append(f"Since {_chain_en(s['chain'])}, {a} cannot come before {b}, so the other option must hold: {c} comes before {d}.")
        elif t == "sat":
            out.append(f"Since {_chain_en(s['chain'])}, the constraint that {_dj_en(disj[s['k']])} is already satisfied.")
        elif t == "assume":
            a, b = disj[s["k"]][s["side"]]; out.append(f"Suppose that {a} comes before {b}.")
        elif t == "cycle":
            out.append(f"But then {_chain_en(s['chain'])}, which is a contradiction.")
        elif t == "both":
            out.append(f"But then the constraint that {_dj_en(disj[s['k']])} cannot be satisfied, because {_chain_en(s['chain0'])}, and also {_chain_en(s['chain1'])}. This is a contradiction.")
        elif t == "back":
            c, d = disj[s["k"]][s["side"]]; out.append(f"So that assumption was wrong, and the other option must hold: {c} comes before {d}.")
        elif t == "order":
            out.append("Therefore the order, from first to last, is " + " ".join(map(str, s["sigma"])) + ".")
    return "\n".join(out)


# ---------------------------------------------------------------- parsing（文本 → 表层 → 按状态解析出析取下标 → steps）
_EDGE = re.compile(r"^(\d+)<(\d+)$")
_CHAIN = re.compile(r"^\d+(?:<\d+)+$")
_DJ = re.compile(r"^\((\d+)<(\d+)\)\|\((\d+)<(\d+)\)$")
_LINK = re.compile(r"(\d+) comes before (\d+)")


def _pe(w):
    m = _EDGE.match(w); assert m, w; return (int(m[1]), int(m[2]))
def _pc(w):
    assert _CHAIN.match(w), w; return [int(x) for x in w.split("<")]
def _pd(w):
    m = _DJ.match(w); assert m, w; return ((int(m[1]), int(m[2])), (int(m[3]), int(m[4])))
def _pc_en(txt):
    links = [(int(a), int(b)) for a, b in _LINK.findall(txt)]
    assert links and all(links[i][1] == links[i + 1][0] for i in range(len(links) - 1)), txt
    assert _chain_en([links[0][0]] + [y for _, y in links]) == txt.strip(), txt
    return [links[0][0]] + [y for _, y in links]


def _surface(text, version):
    lines = text.split("\n"); out = []
    if version == "N1":
        for L in lines:
            if (m := re.match(r"^Since (.*), (\d+) cannot come before (\d+), so the other option must hold: (\d+) comes before (\d+)\.$", L)):
                out.append(dict(t="force", chain=_pc_en(m[1]), dead=(int(m[2]), int(m[3])), forced=(int(m[4]), int(m[5]))))
            elif (m := re.match(r"^Since (.*), the constraint that (\d+) comes before (\d+) or (\d+) comes before (\d+) is already satisfied\.$", L)):
                out.append(dict(t="sat", chain=_pc_en(m[1]), dj=((int(m[2]), int(m[3])), (int(m[4]), int(m[5])))))
            elif (m := re.match(r"^Suppose that (\d+) comes before (\d+)\.$", L)):
                out.append(dict(t="assume", edge=(int(m[1]), int(m[2]))))
            elif (m := re.match(r"^But then the constraint that (\d+) comes before (\d+) or (\d+) comes before (\d+) cannot be satisfied, because (.*), and also (.*)\. This is a contradiction\.$", L)):
                out.append(dict(t="both", dj=((int(m[1]), int(m[2])), (int(m[3]), int(m[4]))), chain0=_pc_en(m[5]), chain1=_pc_en(m[6])))
            elif (m := re.match(r"^But then (.*), which is a contradiction\.$", L)):
                out.append(dict(t="cycle", chain=_pc_en(m[1])))
            elif (m := re.match(r"^So that assumption was wrong, and the other option must hold: (\d+) comes before (\d+)\.$", L)):
                out.append(dict(t="back", edge=(int(m[1]), int(m[2]))))
            elif (m := re.match(r"^Therefore the order, from first to last, is ([\d ]+)\.$", L)):
                out.append(dict(t="order", sigma=[int(x) for x in m[1].split()]))
            else: raise ValueError(f"N1 unparsable: {L!r}")
        return out
    inv = {v: k for k, v in MAPS[version].items()}
    for L in lines:
        assert L.startswith(" "), L
        w = L[1:].split(" ")
        c = [inv.get(x) for x in w]
        if c[0] == "since" and c[2] == "not" and c[4] == "forces" and len(w) == 6:
            out.append(dict(t="force", chain=_pc(w[1]), dead=_pe(w[3]), forced=_pe(w[5])))
        elif c[0] == "since" and c[2] == "holds" and len(w) == 4:
            out.append(dict(t="sat", chain=_pc(w[1]), dj=_pd(w[3])))
        elif c[0] == "suppose" and len(w) == 2:
            out.append(dict(t="assume", edge=_pe(w[1])))
        elif c[0] == "contradiction" and len(w) == 2:
            out.append(dict(t="cycle", chain=_pc(w[1])))
        elif c[0] == "contradiction" and len(w) == 6 and c[2] == "since" and c[4] == "since":
            out.append(dict(t="both", dj=_pd(w[1]), chain0=_pc(w[3]), chain1=_pc(w[5])))
        elif c[0] == "so" and len(w) == 2:
            out.append(dict(t="back", edge=_pe(w[1])))
        elif c[0] == "order":
            out.append(dict(t="order", sigma=[int(x) for x in w[1:]]))
        else: raise ValueError(f"{version} unparsable: {L!r}")
    return out


def parse(text, item, version) -> list[dict]:
    """文本 → steps。析取下标按求解状态解析（最低编号的未决、匹配的析取；back 取最近弹出的假设）。"""
    n, hard, disj, sigma = _inst(item)
    ch = {}; stack = []; popped = None; steps = []
    def undecided(pred): return min(k for k in range(len(disj)) if k not in ch and pred(k))
    for s in _surface(text, version):
        t = s["t"]
        if t == "force":
            k = undecided(lambda k: (disj[k][0] == s["dead"] and disj[k][1] == s["forced"]) or (disj[k][1] == s["dead"] and disj[k][0] == s["forced"]))
            dead = 0 if disj[k][0] == s["dead"] else 1
            steps.append(dict(t="force", k=k, dead=dead, chain=s["chain"])); ch[k] = 1 - dead
        elif t == "sat":
            k = undecided(lambda k: disj[k] == s["dj"])
            side = 0 if disj[k][0] == (s["chain"][0], s["chain"][-1]) else 1
            steps.append(dict(t="sat", k=k, side=side, chain=s["chain"])); ch[k] = side
        elif t == "assume":
            k = undecided(lambda k: s["edge"] in disj[k]); side = disj[k].index(s["edge"])
            stack.append((k, side, dict(ch))); ch[k] = side
            steps.append(dict(t="assume", k=k, side=side))
        elif t in ("cycle", "both"):
            if t == "both":
                k = undecided(lambda k: disj[k] == s["dj"]); steps.append(dict(t="both", k=k, chain0=s["chain0"], chain1=s["chain1"]))
            else: steps.append(dict(t="cycle", chain=s["chain"]))
            popped = stack.pop(); ch = dict(popped[2])
        elif t == "back":
            k, side0, _ = popped; side = 1 - side0
            assert disj[k][side] == s["edge"], "back edge mismatch"
            ch[k] = side; steps.append(dict(t="back", k=k, side=side)); popped = None
        elif t == "order":
            steps.append(dict(t="order", sigma=s["sigma"]))
    return steps


# ---------------------------------------------------------------- verification（独立重放，检查每条链的每一环都是当时可用的显式约束）
def verify(steps, item) -> tuple[bool, str]:
    n, hard, disj, sigma = _inst(item)
    ch = {}; stack = []; pending_back = None
    def E(): return set(hard) | {disj[k][s] for k, s in ch.items()}
    def chain_ok(c, x, y):
        es = E(); return c[0] == x and c[-1] == y and len(c) >= 2 and all((c[i], c[i + 1]) in es for i in range(len(c) - 1))
    for i, s in enumerate(steps):
        t = s["t"]
        if pending_back is not None and t != "back": return False, f"step {i}: expected back"
        if t == "force":
            k, dead = s["k"], s["dead"]; a, b = disj[k][dead]
            if k in ch or not chain_ok(s["chain"], b, a): return False, f"step {i}: bad force"
            ch[k] = 1 - dead
        elif t == "sat":
            k, side = s["k"], s["side"]; a, b = disj[k][side]
            if k in ch or not chain_ok(s["chain"], a, b): return False, f"step {i}: bad sat"
            ch[k] = side
        elif t == "assume":
            if s["k"] in ch: return False, f"step {i}: assume on decided"
            stack.append((s["k"], s["side"], dict(ch))); ch[s["k"]] = s["side"]
        elif t == "cycle":
            c = s["chain"]
            if not (len(c) >= 3 and c[0] == c[-1] and chain_ok(c, c[0], c[-1])) or not stack: return False, f"step {i}: bad cycle"
            pending_back = stack.pop()
        elif t == "both":
            k = s["k"]; (a, b), (c_, d) = disj[k]
            if k in ch or not chain_ok(s["chain0"], b, a) or not chain_ok(s["chain1"], d, c_) or not stack: return False, f"step {i}: bad both"
            pending_back = stack.pop()
        elif t == "back":
            if pending_back is None: return False, f"step {i}: back without conflict"
            k, side0, saved = pending_back
            if s["k"] != k or s["side"] != 1 - side0: return False, f"step {i}: bad back"
            ch = dict(saved); ch[k] = s["side"]; pending_back = None
        elif t == "order":
            after = closure(n, list(E()))
            if after is None or not is_total(n, after) or _topo(n, after) != s["sigma"] or s["sigma"] != sigma: return False, f"step {i}: bad order"
            if i != len(steps) - 1: return False, "order not last"
    return True, "ok"
