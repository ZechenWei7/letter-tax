"""第二阶段闸门 1：打分、内容检验与判读的纯逻辑（CPU，可单测）。GPU 入口：scripts/phase2_gate1_train.py / phase2_gate1_eval.py；
判读：scripts/phase2_gate1_analyze.py。不在 OSF 注册的 stage-1 协议之内。

定义（在任何 gate-1 数据之前写定）：
  think 段        模型输出中第一个 "</think>" 之前的文本，去掉末尾换行（SFT 目标是 "推导\n</think>…"）。
  解析正确        derivation.parse(think) 成功，且与该题的规范推导 derivation.derive(item) 逐步完全相同。
  可解析 / 可证   parse 成功 / parse 成功且 derivation.verify 通过（副指标）。
  循环            think 段中同一非空行出现 ≥ LOOP_MIN_REPEATS 次（新评估集 500 题的金标推导里，同一行最多重复 4 次）。
  撞 cap          think 段到达 cap（由预算强制收尾），即 evaluate._row 的 capped。
  移植            题 i 的 think 段整段替换为题 π(i) 的金标推导（同一写法），π 为 seed 0 的均匀随机错位；只生成答案。
                  副：π(i) 用该 run 自己在题 π(i) 上生成的 think 段（同 stage-1 E3 的做法）。
  篡改            规范推导去掉末行 order 后长 m 步，前半 h = ⌊m/2⌋ 步作为预填。在前半里按顺序找第一条"推出的关系"
                  （force 步的被迫边，或 back 步的边）c<d，要求：它在最终解 σ 中成立；它在前半其后各步的链里不再出现；且在后半某步的链里出现
                  （即正确续写要用到它）。篡改 = 把该行的 c<d 改写为 d<c；对照 = 不改的同一前半。模型从前半之后续写 think 与答案。
                  找不到这样的关系的题不进篡改检验（报覆盖率）。
                  （"在 σ 中成立"一条是 2026-09-23 用桩生成器测流水线时加的：否则会选到被推翻的假设分支里推出的关系，
                  篡改它不影响正确答案，oracle 桩的对照组里有 13.7% 的题出现"答案层面 d 在 c 之前"。此时尚无任何真实 gate-1 数据。）
                  "续写与被改关系一致" = 续写部分（按行宽松解析）的链中出现 d→c 这一环、且不出现 c→d；另报答案层面的 d 是否排在 c 之前。
                  对照组同样计算这两个量作基线（正确的续写也可能在随后被推翻的假设分支里出现 d→c）。
"""
from __future__ import annotations
import collections, math, random

from cot_compress import derivation as D

LOOP_MIN_REPEATS = 5
TOST_MARGIN = 0.05
ALPHA = 0.05
Z90 = 1.6448536269514722        # 双单侧 α=0.05 的 TOST ⇔ 90% 置信区间落在 ±margin 内


# ---------------------------------------------------------------- 打分
def think_of(completion: str) -> str:
    return completion.split("</think>", 1)[0].rstrip("\n")


def loop_flag(think: str) -> bool:
    lines = [l for l in think.split("\n") if l.strip()]
    return bool(lines) and max(collections.Counter(lines).values()) >= LOOP_MIN_REPEATS


def score_derivation(think: str, item: dict, version: str, gold_steps: list) -> dict:
    try:
        steps = D.parse(think, item, version)
    except Exception:
        return dict(parsed=False, exact=False, sound=False)
    try: sound = D.verify(steps, item)[0]
    except Exception: sound = False
    return dict(parsed=True, exact=(steps == gold_steps), sound=bool(sound))


# ---------------------------------------------------------------- 移植
def transplant_pairing(n: int, seed: int = 0) -> list[int]:
    from cot_compress.admission import derangement
    return derangement(n, random.Random(seed))


# ---------------------------------------------------------------- 篡改
def _links(s: dict) -> set:
    out = set()
    for key in ("chain", "chain0", "chain1"):
        c = s.get(key)
        if c: out |= {(c[i], c[i + 1]) for i in range(len(c) - 1)}
    return out


def _derived_edge(s: dict, disj) -> tuple | None:
    if s["t"] == "force": return tuple(disj[s["k"]][1 - s["dead"]])
    if s["t"] == "back": return tuple(disj[s["k"]][s["side"]])
    return None


def tamper_plan(steps: list, item: dict) -> dict | None:
    disj = [(tuple(a), tuple(b)) for a, b in item["meta"]["disj"]]
    pos = {e: i for i, e in enumerate(item["meta"]["sigma"])}
    body = [s for s in steps if s["t"] != "order"]
    h = len(body) // 2
    for i in range(h):
        e = _derived_edge(body[i], disj)
        if e is None or pos[e[0]] > pos[e[1]]: continue          # 只选在最终解 σ 里成立的关系（被推翻的假设分支里推出的、σ 中不成立的不选）
        if any(e in _links(body[j]) for j in range(i + 1, h)): continue
        if any(e in _links(body[j]) for j in range(h, len(body))):
            return dict(line=i, edge=e, h=h, m=len(body))
    return None


def tamper_texts(steps: list, item: dict, version: str, plan: dict) -> tuple[str, str]:
    """返回 (对照前缀, 篡改前缀)，都是 think 段的前 h 行（不含末尾换行）。"""
    lines = D.render(steps[:plan["h"]], item, version).split("\n")
    c, d = plan["edge"]; old = f"{c}<{d}"
    L = lines[plan["line"]]; assert L.endswith(" " + old), (L, old)
    tl = lines[:]; tl[plan["line"]] = L[: -len(old)] + f"{d}<{c}"
    return "\n".join(lines), "\n".join(tl)


def continuation_follow(cont_think: str, version: str, edge: tuple, pred: str | None) -> dict:
    """续写部分按行宽松解析；统计链中的 c→d / d→c 两个方向。"""
    c, d = edge; uses_orig = uses_flip = False; n_parsed = 0
    for line in cont_think.split("\n"):
        if not line.strip(): continue
        try: sf = D._surface(line, version)
        except Exception: continue
        n_parsed += 1
        for s in sf:
            L = _links(s)
            uses_orig |= (c, d) in L; uses_flip |= (d, c) in L
    order = None
    if pred:
        try:
            p = [int(x) for x in pred.split()]
            if c in p and d in p: order = "flip" if p.index(d) < p.index(c) else "orig"
        except ValueError: order = None
    return dict(n_parsed_lines=n_parsed, uses_orig=uses_orig, uses_flip=uses_flip,
                follows_flip=(uses_flip and not uses_orig), answer_order=order)


# ---------------------------------------------------------------- 判读（配对：键 = (题 id, seed)）
def mcnemar_p(b: int, c: int) -> float:
    """精确双侧 McNemar（二项检验，p=0.5）。"""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(xa: dict, xb: dict) -> dict:
    """xa, xb: {(item, seed): 0/1}。diff = acc_b − acc_a（按配对键）。"""
    keys = sorted(set(xa) & set(xb)); n = len(keys)
    assert n >= 2, "need paired data"
    d = [int(xb[k]) - int(xa[k]) for k in keys]
    b10 = sum(1 for k in keys if xa[k] and not xb[k]); b01 = sum(1 for k in keys if not xa[k] and xb[k])
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    ci90 = (mean - Z90 * se, mean + Z90 * se)
    p = mcnemar_p(b10, b01)
    return dict(n=n, acc_a=sum(int(xa[k]) for k in keys) / n, acc_b=sum(int(xb[k]) for k in keys) / n, diff=mean,
                a_only=b10, b_only=b01, p_mcnemar=p, ci90=ci90,
                equivalent=(ci90[0] > -TOST_MARGIN and ci90[1] < TOST_MARGIN),
                sig_worse=(p < ALPHA and mean <= -TOST_MARGIN), sig_better=(p < ALPHA and mean >= TOST_MARGIN))


def classify(acc: dict, premise: dict) -> dict:
    """acc: {"N2": {...}, "N2s": {...}, "N3m": {...}, "N3u": {...}}，值为 {(item, seed): 0/1}（原生 think 评估；N3m = 禁字母，N3u = 不禁字母）。
    premise: {"N2_acc": float, "N2_exact": float, "N2_transplant_acc": float}（两个 seed 合并）。
    返回 dict(reading, premise_ok, premise_checks, comparisons, subreadings)。"""
    pc = dict(N2_acc_ge_50=premise["N2_acc"] >= 0.50, N2_exact_ge_80=premise["N2_exact"] >= 0.80,
              N2_transplant_lt_10=premise["N2_transplant_acc"] < 0.10)
    out = dict(premise_checks=pc, premise_ok=all(pc.values()), comparisons={}, subreadings={})
    if not out["premise_ok"]:
        out["reading"] = "P_fail"; out["text"] = "干净推导在此训练量下不可学/不可执行；停。"; return out
    C = out["comparisons"]
    C["N3m_vs_N2"] = compare(acc["N2"], acc["N3m"])        # diff = N3m − N2
    C["N2s_vs_N2"] = compare(acc["N2"], acc["N2s"])        # diff = N2s − N2
    C["N3u_vs_N2"] = compare(acc["N2"], acc["N3u"])        # diff = N3u − N2
    C["N2s_vs_N3m"] = compare(acc["N3m"], acc["N2s"])      # diff = N2s − N3m
    C["N3m_vs_N3u"] = compare(acc["N3u"], acc["N3m"])      # diff = N3m − N3u
    main = C["N3m_vs_N2"]
    if main["equivalent"]:
        out["reading"] = "R1"; out["text"] = "给定相同内容，无字母记法执行得和简洁英文一样好。"; return out
    if main["sig_worse"]:
        sub = out["subreadings"]
        sub["R2a"] = C["N2s_vs_N2"]["diff"] <= -TOST_MARGIN and C["N2s_vs_N3m"]["equivalent"]
        sub["R2b"] = C["N2s_vs_N2"]["equivalent"] and abs(C["N3m_vs_N3u"]["diff"]) < TOST_MARGIN
        sub["R2c"] = C["N3u_vs_N2"]["equivalent"]
        hit = [k for k, v in sub.items() if v]
        texts = dict(R2a="起作用的是连接词的含义，不是字母。", R2b="起作用的是字母本身。", R2c="是 mask 本身在干扰。")
        if len(hit) == 1:
            out["reading"] = hit[0]; out["text"] = texts[hit[0]]
        else:
            out["reading"] = "R2_unclear"; out["text"] = f"说不清（R2 子读法命中 {hit or '无'}）；停。"
        return out
    if main["sig_better"]:
        out["reading"] = "R3"; out["text"] = "N3 显著更好；如实报告。"; return out
    out["reading"] = "R4"; out["text"] = "以上都不是；说不清，停。"; return out
