"""v4 奖励（README §2）：
    r = 10 · [ c·(1 − min(λ·L/M, 0.9)) − 0.5·v·(1−c) − 0.05·v·c ]
c ∈ {0,1} 正确；v ∈ {0,1} 格式违规（任一）。**v 只能在答案区发生**：唯一性与答案区只看第一个 </think> 之后（思考内草稿忽略，2026-09-18），
think 段内容本身（包括出现 <answer> 草稿、任何字符）不触发 v；think 段只受 mask（采样侧）约束。违规分两级：
  hard（c := 0 → r = −5）：第一个 </think> 与其后第一个 <answer> 之间有非空白；第二个 think 标签（<think> 已在 prompt 预填，完成段里出现 <think> 或 </think> 多于一次）
  soft（v=1，c 按答案）：</think> 之后出现多个不同答案值；答案区（</think> 后首个 <answer> 到末尾）> 32 token
  答案缺失 / 格式错 → c 必然 0，r = −5。
无组内通过率门控。L = 生成开始 → 最后一个 <answer> 之前的 token 数（强制收尾 / 无 <answer> → cap）；C_rand / D 由 rollout 侧通道给 L。
性质（单测穷举验证）：任意正确样本 r ≥ 0.5 > 0 ≥ 任意错误样本；任意 hard 违规 r = −5 = min，严格低于无违规的错误（0）与任意正确样本。
每步记录"正确样本中 L > 0.9·M/λ 的比例"（长度项饱和区）。
"""
from __future__ import annotations
import json, pathlib, statistics
from .parsing import ANSWER_OPEN, ANSWER_CLOSE, THINK_OPEN, THINK_CLOSE, _ANS_RE

MAX_ANSWER_TOKENS = 32   # 完整终态答案（如 8 个 "6D"）需要更多 token；2026-09-18 由 16 改 32
SIDE = {"L": None}   # C_rand：rollout 把每条完成的 L 放这里（与 completions 同序）
LAST = {}            # 最近一次奖励计算的汇总（train.py 的 StepLogger 在 vLLM 路径下用它取自然结束比例）
HARD = {"text_after_think", "second_think_tag"}          # hard 违规：c := 0（r = −5，与"错且格式差"同级）——封住"</think> 先行再明文推理"的逃逸
SOFT = {"multi_answer", "answer_too_long"}               # soft 违规：v=1，c 按答案判（答对扣 0.05·10）
# no_answer / bad_answer：答案缺失或非法 → c 必然 0，v=1 → r = −5

def is_hard(vio) -> bool:
    return any(x in HARD for x in vio)

def score(c: float, v: float, L: float, M: float, lam: float) -> float:
    return 10.0 * (c * (1.0 - min(lam * L / M, 0.9)) - 0.5 * v * (1.0 - c) - 0.05 * v * c)

def answer_region(text: str) -> str:
    """答案区 = 第一个 </think> 之后的文本（无 </think> 则全文）。思考内的 <answer> 草稿一律忽略（2026-09-18 定）。"""
    return text.split(THINK_CLOSE, 1)[1] if THINK_CLOSE in text else text

def parse_answer(text: str, task):
    """返回 (answer_norm | None, reasons)。只看 </think> 之后的 <answer> 标签：同值多标签 = 一个候选；不同值 = multi_answer。"""
    found = _ANS_RE.findall(answer_region(text))
    if not found:
        return None, ["no_answer"]
    vals = {task.normalize(x) for x in found}
    if len(vals) > 1:
        return None, ["multi_answer"]
    v = vals.pop()
    if v is None:
        return None, ["bad_answer"]
    return v, []

def violations(text: str, task, tok, max_answer_tokens: int = MAX_ANSWER_TOKENS) -> list[str]:
    ans, reasons = parse_answer(text, task)
    if text.count(THINK_OPEN) >= 1 or text.count(THINK_CLOSE) > 1:      # <think> 已在 prompt 预填：完成段里出现即第二个标签
        reasons.append("second_think_tag")
    region = answer_region(text)
    if ANSWER_OPEN in region:
        if THINK_CLOSE in text and region.split(ANSWER_OPEN, 1)[0].strip() != "":   # 第一个 </think> → 其后第一个 <answer> 之间只许空白
            reasons.append("text_after_think")
        if len(tok(region[region.find(ANSWER_OPEN):], add_special_tokens=False)["input_ids"]) > max_answer_tokens:   # 答案区：首个 <answer> 到末尾
            reasons.append("answer_too_long")
    return reasons

def span_length(tok, text: str, cap: int, force_start: int) -> tuple[int, bool]:
    if ANSWER_OPEN not in text:
        return cap, True
    n = len(tok(text.rsplit(ANSWER_OPEN, 1)[0], add_special_tokens=False)["input_ids"])
    return (cap, True) if n >= force_start else (n, False)

def make_reward(task, tok, *, lam: float, M: float, cap: int, force_start: int, log_path=None, length_term: bool = True,
                prefilled: bool = False, archive_path=None, task_n: int | None = None, kk_n: int | None = None, ord_nd: tuple | None = None):
    """archive_path：runs/<run>/rollouts.jsonl.zst，每步追加全部 rollout（prompt id、完整文本、token ids、奖励各分量、L、策略描述量）。"""
    """length_term=False（臂 D）：λ 视为 0。C_rand / D 的 L 来自 SIDE['L']（rollout 写入），否则从文本算。"""
    lam_eff = lam if length_term else 0.0
    log_path = pathlib.Path(log_path) if log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    sat_thr = (0.9 * M / lam) if lam > 0 else float("inf")

    def reward(prompts, completions, completion_ids=None, answer=None, **kw):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        side_L = kw.get("L_side") if prefilled else SIDE.get("L")      # prefilled 臂（C_rand / D）：L 是数据集列，think 在 prompt 里
        use_side = side_L is not None and len(side_L) == len(texts)
        rows, out = [], []
        for i, (t, gold) in enumerate(zip(texts, answer)):
            if use_side:                     # C_rand / D：<answer> 开标签在 prompt 里，完成段只有答案续写
                t = ANSWER_OPEN + t
            ans, _ = parse_answer(t, task)
            c = 1.0 if (ans is not None and task.check(ans, gold)) else 0.0
            vio = violations(t, task, tok)
            v = 1.0 if vio else 0.0
            if is_hard(vio):
                c = 0.0                      # hard 违规：答对也不计分 → r = −5
            if use_side:
                L, capped = int(side_L[i]), False
            else:
                L, capped = span_length(tok, t, cap, force_start)
            r = score(c, v, L, M, lam_eff)
            rows.append(dict(c=c, v=v, hard=is_hard(vio), L=L, capped=capped, violations=vio, r=r, pred=ans, text=t))
            out.append(r)
        LAST.update(n=len(rows), capped=statistics.mean(1.0 if r["capped"] else 0.0 for r in rows),
                    acc=statistics.mean(r["c"] for r in rows), viol=statistics.mean(r["v"] for r in rows),
                    hard=statistics.mean(1.0 if r["hard"] else 0.0 for r in rows), L_mean=statistics.mean(r["L"] for r in rows))
        if use_side and not prefilled:
            SIDE["L"] = None
        st = kw.get("trainer_state"); step = getattr(st, "global_step", None)
        if archive_path:
            from .archive import append_jsonl_zst
            ids_list = completion_ids if completion_ids is not None else [None] * len(rows)
            pids = kw.get("id") or [None] * len(rows)
            style = None
            if task_n:
                from .style import rewrite_style
            if task_n or kk_n or ord_nd:
                from .traj_stats import think_text
            if kk_n:
                from .strategy import annotate
                from .templates import detect
            if ord_nd:
                from .shortcuts import strategy_class as _sc, detect_shortcuts as _ds
            users = kw.get("user") or [None] * len(rows)
            recs = []
            for i, r in enumerate(rows):
                rec = dict(step=step, i=i, prompt_id=pids[i] if i < len(pids) else None, gold=answer[i], text=r["text"],
                           token_ids=list(ids_list[i]) if ids_list[i] is not None else None, n_tokens=(len(ids_list[i]) if ids_list[i] is not None else None),
                           c=r["c"], v=r["v"], hard=r["hard"], violations=r["violations"], L=r["L"], capped=r["capped"], r=r["r"], pred=r["pred"])
                if task_n:
                    rec["style"] = rewrite_style(think_text(r["text"]), task_n)
                if kk_n and i < len(users) and users[i]:
                    th = think_text(r["text"]); rec["strategy"] = annotate(th, users[i], kk_n, tok); rec["template_hits"] = detect(th, users[i], kk_n)
                if ord_nd and i < len(users) and users[i]:
                    th = think_text(r["text"]); S_ = (kw.get("S") or [1] * len(rows))[i]
                    rec["strategy"] = _sc(th, users[i], ord_nd[0], int(S_), tok); rec["shortcut_hits"] = _ds(th, users[i], ord_nd[0], ord_nd[1])
                recs.append(rec)
            append_jsonl_zst(archive_path, recs)
        if log_path:
            corr = [r for r in rows if r["c"] == 1.0]
            with open(log_path, "a") as fh:
                fh.write(json.dumps(dict(step=step, n=len(rows), acc=statistics.mean(r["c"] for r in rows),
                                         viol_rate=statistics.mean(r["v"] for r in rows),
                                         L_mean=statistics.mean(r["L"] for r in rows),
                                         capped=statistics.mean(1.0 if r["capped"] else 0.0 for r in rows),
                                         sat_frac=(statistics.mean(1.0 if r["L"] > sat_thr else 0.0 for r in corr) if corr else None),
                                         reward_mean=statistics.mean(out),
                                         rows=[{k: r[k] for k in ("c", "v", "hard", "L", "capped", "r", "pred")} for r in rows])) + "\n")
        return out
    reward.__name__ = "cot_compress_reward_v4"
    return reward
