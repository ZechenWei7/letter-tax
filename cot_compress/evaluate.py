"""准确率评估核心（校准、臂 0、训练中 eval 共用）。
思考长度 L（README §2 写定）：生成开始 → "<answer>" 之前的全部 token；被预算强制收尾 / 无 <answer> 的轨迹 L = cap。
rows 里的 think_tokens 字段即 L（历史字段名保留，calibration.csv 兼容）。
"""
from __future__ import annotations
import cot_compress  # noqa: F401  unsloth first
import statistics, time
import torch

import tasks
from .generation import chat_prompt, direct_prefill, generate_texts
from .parsing import ANSWER_OPEN, THINK_CLOSE, parse_completion, Parsed
from .span import HOOK

VLLM = {"llm": None}   # 设为 vllm.LLM 后，run_eval / forced_answer_eval / prefilled_answer_eval 的生成全部走 vLLM（准入校准 n≥500 用）

def _gen(model, tok, prompts, *, max_new_tokens, sampling, batch_size, seed, force: bool, progress=None, batch_stats=None,
         logits_processor=None):
    """force=True：思考段生成（预算强制 + HOOK.banned_ids 屏蔽）；False：只生成答案段（无强制、无屏蔽）。"""
    if VLLM["llm"] is not None:
        from .rollout import generate_texts_vllm
        return generate_texts_vllm(VLLM["llm"], tok, prompts, max_new_tokens=max_new_tokens, sampling=sampling, reserve=HOOK.reserve,
                                   mask_mode=(HOOK.mask_mode if force else "none"), force=force, seed=seed)
    was = HOOK.enabled
    if not force:
        HOOK.enabled = False
    try:
        return generate_texts(model, tok, prompts, max_new_tokens=max_new_tokens, sampling=sampling, batch_size=batch_size, seed=seed,
                              logits_processor=logits_processor, progress=progress, batch_stats=batch_stats)
    finally:
        HOOK.enabled = was

MODES = ("think", "guided", "direct")
SPAN_OF = dict(think="tags", guided="pre_answer", direct="none")
CLOSE_OF = dict(think=THINK_CLOSE + "\n\n" + ANSWER_OPEN, guided=ANSWER_OPEN)
# 预算强制 / 截断时追加的收尾。(a) 必须连 <answer> 一起预填：只追加 </think> 时 Qwen3 会在答案区继续算，48 token 内到不了 <answer>。
# think  = (a) Qwen3 原生思考模式（enable_thinking=True），模型自发 <think>…</think>；思考段 = 标记内
# guided = (b) 关闭原生思考（enable_thinking=False，模板注入空 think 块）+ SYSTEM_PROMPT_GUIDED 要求逐步推理
#          （IBM Abstract-CoT 做法）。思考段 = <answer> 之前的全部生成文本；格式只要求 <answer> 存在且唯一。
# direct = enable_thinking=False + 预填 <answer>，max_new_tokens 很小，模型无法在答案外推理

def _system(task, mode):
    return task.SYSTEM_PROMPT if mode == "think" else task.SYSTEM_PROMPT_GUIDED

THINK_PREFILL = "<think>\n"   # v4：<think> 不在白名单（special），所有臂在 prompt 里预填，完成段 = think 体 + </think> + 答案

def _prompt_text(tok, task, item, mode, gen_cfg):
    native = mode == "think"
    prefill = {"think": THINK_PREFILL, "guided": str(gen_cfg.get("guided_prefill") or ""), "direct": direct_prefill()}[mode]
    return chat_prompt(tok, _system(task, mode), item["prompt"], think=native, prefill=prefill), prefill

def span_L(tok, text: str, cap: int | None, forced: bool = False) -> tuple[int, bool]:
    """L = 生成开始 → <answer> 之前的 token 数；forced 或无 <answer> → (cap, True)。cap=None 时不封顶。"""
    if forced and cap is not None:
        return cap, True
    if ANSWER_OPEN not in text:
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        return (cap if cap is not None else n), True
    return len(tok(text.rsplit(ANSWER_OPEN, 1)[0], add_special_tokens=False)["input_ids"]), False   # 最后一个 <answer>

def _row(tok, task, it, mode, text, o=None, extra=None, cap=None):
    """think 模式按 v4 规则打分：c = 答案值正确且无 hard 违规；v = rewards.violations 任一；format_ok = ¬v；correct = c。
    另给 correct_clean = c ∧ ¬v（准入检查 6 / 端点用它）。hard 违规（text_after_think / second_think_tag）使 c=0。"""
    span = SPAN_OF[mode]
    p = parse_completion(text, span=span, validate=task.is_valid, validate_norm=task.normalize)
    if mode != "direct":
        from .rewards import parse_answer, violations, is_hard
        ans, _ = parse_answer(text, task); vio = violations(text, task, tok)
        c = bool(ans is not None and task.check(ans, it["answer"])) and not is_hard(vio)   # hard 违规：c := 0
        p = Parsed(p.think, ans, not vio, ("ok" if not vio else "+".join(vio)), p.tail)
        correct = c
    else:
        correct = bool(p.format_ok and task.check(p.answer, it["answer"]))
        vio = [] if p.format_ok else [p.reason]
    lenient = task.lenient_extract(p.tail if p.tail else text)
    if mode == "direct":
        span_tokens, capped = 0, False
    else:
        span_tokens, capped = span_L(tok, text, cap, forced=bool(o is not None and o.forced))
    r = dict(id=it["id"], key=it["meta"]["key"], mode=mode, gold=it["answer"], pred=p.answer, correct=correct,
             correct_clean=bool(correct and p.format_ok), violations=vio,
             format_ok=p.format_ok, reason=p.reason, lenient_pred=lenient,
             lenient_correct=bool(lenient is not None and task.check(lenient, it["answer"])),
             think_tokens=span_tokens, capped=capped, forced=bool(o is not None and o.forced),
             has_think_close=bool(o is not None and o.has_think_close),
             completion_tokens=len(o.ids) if o is not None else None,
             finished=o.finished if o is not None else True, prompt=it["prompt"], completion=text)
    if extra:
        r.update(extra)
    return r

def run_eval(model, tok, key: str, items: list[dict], mode: str, gen_cfg: dict, *, seed: int = 0,
             logits_processor=None, progress=None, batch_stats: list | None = None,
             backend: str = "hf", llm=None, banned_ids=None) -> list[dict]:
    """backend="vllm" 时用 llm（vLLM LLM 对象，如 TRL colocate 的 trainer.llm）生成，屏蔽/预算强制由 rollout.VLLMSpanProcessor 施加。"""
    assert mode in MODES, mode
    task = tasks.task_for_key(key)
    think = mode != "direct"
    pp = [_prompt_text(tok, task, it, mode, gen_cfg) for it in items]
    prompts, prefill = [x[0] for x in pp], pp[0][1]
    max_new = int(gen_cfg["max_new_tokens_think"] if think else gen_cfg["max_new_tokens_direct"])
    sampling = dict(do_sample=gen_cfg.get("do_sample", True), temperature=gen_cfg.get("temperature", 0.6),
                    top_p=gen_cfg.get("top_p", 0.95), top_k=gen_cfg.get("top_k", 20))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if backend == "vllm":
        from .rollout import generate_texts_vllm
        outs = generate_texts_vllm(llm, tok, prompts, max_new_tokens=max_new, sampling=sampling, seed=seed, reserve=HOOK.reserve,
                                   mask_mode=(HOOK.mask_mode if think else "none"), force=think)
    else:
        outs = _gen(model, tok, prompts, max_new_tokens=max_new, sampling=sampling, batch_size=int(gen_cfg.get("batch_size", 8)), seed=seed,
                    force=think, logits_processor=logits_processor, progress=progress, batch_stats=batch_stats)
    dt = time.time() - t0
    rows = [_row(tok, task, it, mode, (ANSWER_OPEN if not think else "") + o.text, o, cap=max_new if think else None)
            for it, o in zip(items, outs)]
    peak = torch.cuda.max_memory_allocated() / 2**30
    for r in rows:
        r["_sec_total"] = round(dt, 1)
        r["_peak_gib"] = round(peak, 2)
    return rows

@torch.no_grad()
def forced_answer_eval(model, tok, rows: list[dict], cutoff: int | None, gen_cfg: dict, *, seed: int = 0,
                       max_new_tokens: int = 24, batch_size: int = 8, cutoff_frac: float | None = None,
                       shuffle: bool = False) -> list[dict]:
    """预算强制 / 截断敏感性 / 打乱敏感性（内容度量，README §5）：
    - cutoff：把 L 段（生成开始→最后一个 <answer> 前）截到 cutoff 个 token；cutoff_frac：按每条样本自身 L 的比例截（25% / 50%）。
    - shuffle=True：把（截断后的）L 段 token 随机打乱再收尾。
    追加收尾（已含 </think> 则只补 <answer>，否则 "</think>\n\n<answer>"），只重新生成答案段。
    固定 cutoff 且 L 本来就 < cutoff 的自然结束样本原样返回（forced_eval=False）。rows 来自 run_eval（think 模式）。"""
    import random
    rng = random.Random(seed)
    out, todo = [], []
    for r in rows:
        mode = r["mode"]; assert mode in ("think", "guided"), mode
        task = tasks.task_for_key(r["key"])
        pre = r["completion"].rsplit(ANSWER_OPEN, 1)[0] if ANSWER_OPEN in r["completion"] else r["completion"]
        ids = tok(pre, add_special_tokens=False)["input_ids"]
        cut = int(len(ids) * cutoff_frac) if cutoff_frac is not None else cutoff
        if not shuffle and cutoff_frac is None and len(ids) <= cut and ANSWER_OPEN in r["completion"] and not r.get("forced"):
            out.append(dict(r, cutoff=cut, forced_eval=False)); continue
        kept_ids = ids[:cut]
        if shuffle:
            kept_ids = list(kept_ids); rng.shuffle(kept_ids)
        kept = tok.decode(kept_ids)
        close = ANSWER_OPEN if THINK_CLOSE in kept else CLOSE_OF["think"]
        item = dict(id=r["id"], answer=r["gold"], prompt=r["prompt"], meta=dict(key=r["key"]))
        base, _ = _prompt_text(tok, task, item, mode, gen_cfg)
        head = kept + close   # <think> 已在 prompt 预填
        todo.append((r, item, task, head, base + head))
        out.append(None)
    if todo:
        sampling = dict(do_sample=gen_cfg.get("do_sample", True), temperature=gen_cfg.get("temperature", 0.6),
                        top_p=gen_cfg.get("top_p", 0.95), top_k=gen_cfg.get("top_k", 20))
        outs = _gen(model, tok, [t[4] for t in todo], max_new_tokens=max_new_tokens, sampling=sampling, batch_size=batch_size,
                    seed=seed, force=False)      # 答案段很短，不需要预算强制 / 屏蔽
        k = 0
        for i, x in enumerate(out):
            if x is None:
                r, item, task, head, _ = todo[k]; o = outs[k]; k += 1
                row = _row(tok, task, item, r["mode"], head + o.text, None, cap=None)
                row.update(cutoff=len(tok(head, add_special_tokens=False)["input_ids"]), cutoff_frac=cutoff_frac, shuffled=shuffle,
                           forced_eval=True, finished=True, completion_tokens=None,
                           think_tokens=len(tok(head.rsplit(ANSWER_OPEN, 1)[0], add_special_tokens=False)["input_ids"]),
                           answer_tokens=len(o.ids), orig_correct=r["correct"], orig_think_tokens=r["think_tokens"])
                out[i] = row
    return out

def compression_stats_rows(tok, rows: list[dict]) -> dict:
    """L 段的 gzip 字节比（压缩后/压缩前）与 token bigram 熵（bit）。"""
    import gzip, math
    from collections import Counter
    ratios, ents = [], []
    for r in rows:
        pre = r["completion"].rsplit(ANSWER_OPEN, 1)[0] if ANSWER_OPEN in r["completion"] else r["completion"]
        b = pre.encode("utf-8")
        if len(b) >= 64:
            ratios.append(len(gzip.compress(b, 9)) / len(b))
        ids = tok(pre, add_special_tokens=False)["input_ids"]
        if len(ids) >= 32:
            c = Counter(zip(ids, ids[1:])); n = sum(c.values())
            ents.append(-sum(v / n * math.log2(v / n) for v in c.values()))
    return dict(gzip_ratio=statistics.mean(ratios) if ratios else None, bigram_entropy_bits=statistics.mean(ents) if ents else None)

@torch.no_grad()
def prefilled_answer_eval(model, tok, key: str, items: list[dict], think_ids_per_item: list[list[int]], gen_cfg: dict, *,
                          seed: int = 0, max_new_tokens: int = 24, batch_size: int = 8, tag: str = "prefilled") -> list[dict]:
    """通用"给定 think 段 → 强制收尾 → 只生成答案"评估：填充 / 移植 / unigram 重采样 / 臂 D（think 为空）都用它。
    think_ids_per_item[i] 是放进 <think>…</think> 的 token id 列表。"""
    from .rollout import build_prefilled_prompt
    task = tasks.task_for_key(key)
    prompts, heads = [], []
    for it, ids in zip(items, think_ids_per_item):
        base, _ = _prompt_text(tok, task, it, "think", gen_cfg)
        full = build_prefilled_prompt(tok, base, ids)
        prompts.append(full); heads.append(full[len(base):])
    sampling = dict(do_sample=gen_cfg.get("do_sample", True), temperature=gen_cfg.get("temperature", 0.6),
                    top_p=gen_cfg.get("top_p", 0.95), top_k=gen_cfg.get("top_k", 20))
    outs = _gen(model, tok, prompts, max_new_tokens=max_new_tokens, sampling=sampling, batch_size=batch_size, seed=seed, force=False)
    rows = []
    for it, head, o, ids in zip(items, heads, outs, think_ids_per_item):
        row = _row(tok, task, it, "think", head + o.text, None, cap=None)
        row.update(tag=tag, think_tokens=len(ids), prefilled=True, answer_tokens=len(o.ids))
        rows.append(row)
    return rows

def masked_fraction_rows(tok, rows: list[dict], banned: set[int]) -> float | None:
    """L 段里落在屏蔽集的 token 比例（臂 A 也算）。"""
    from .vocab_mask import masked_fraction
    vals = []
    for r in rows:
        pre = r["completion"].rsplit(ANSWER_OPEN, 1)[0] if ANSWER_OPEN in r["completion"] else r["completion"]
        v = masked_fraction(tok, pre, banned)
        if v is not None:
            vals.append(v)
    return statistics.mean(vals) if vals else None

_REF = {}
def ref_ppl_rows(rows: list[dict], ref_name: str = "Qwen/Qwen3-0.6B-Base", max_tokens: int = 1024) -> float | None:
    """参考模型下 L 段（前 max_tokens token）的每 token 困惑度，返回样本均值。
    Unsloth 把 Qwen3 类全局打了 Triton 补丁，CPU 上前向会报错，所以只在算的时候把 0.6B（bf16，~1.2 GB）搬到 GPU，算完搬回 CPU。"""
    import math
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if "m" not in _REF:
        _REF["tok"] = AutoTokenizer.from_pretrained(ref_name)
        _REF["m"] = AutoModelForCausalLM.from_pretrained(ref_name, dtype=torch.bfloat16).eval()
    rt, rm = _REF["tok"], _REF["m"]
    vals = []
    try:
        torch.cuda.empty_cache()
        rm.to("cuda")
        for r in rows:
            pre = r["completion"].rsplit(ANSWER_OPEN, 1)[0] if ANSWER_OPEN in r["completion"] else r["completion"]
            ids = rt(pre, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :max_tokens].to("cuda")
            if ids.shape[1] < 2:
                continue
            with torch.no_grad():
                loss = rm(input_ids=ids, labels=ids).loss.float().item()
            vals.append(math.exp(loss))
    finally:
        rm.to("cpu"); torch.cuda.empty_cache()
    return statistics.mean(vals) if vals else None

def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    tt = [r["think_tokens"] for r in rows]
    ct = [r["completion_tokens"] or 0 for r in rows]
    return dict(
        n=n,
        acc=sum(r["correct"] for r in rows) / n,
        acc_clean=sum(bool(r.get("correct_clean", r["correct"])) for r in rows) / n,
        viol_rate=sum(not r["format_ok"] for r in rows) / n,
        forced_rate=sum(bool(r.get("forced")) for r in rows) / n,
        capped_rate=sum(bool(r.get("capped")) for r in rows) / n,
        lenient_acc=sum(r["lenient_correct"] for r in rows) / n,
        format_err=sum(not r["format_ok"] for r in rows) / n,
        truncated=sum(not r["finished"] for r in rows) / n,
        think_mean=round(statistics.mean(tt), 1), think_median=statistics.median(tt),
        think_p90=sorted(tt)[int(0.9 * (n - 1))],
        completion_mean=round(statistics.mean(ct), 1),
        sec=rows[0].get("_sec_total", 0) if rows else 0, peak_gib=rows[0].get("_peak_gib", 0) if rows else 0,
        tok_per_s=round(sum(ct) / rows[0]["_sec_total"], 1) if rows and rows[0].get("_sec_total") else 0,
    )
