"""GRPO 训练（v7）。臂：
  A      原生，allowed = 全词表
  A2     A″：mask = 白名单 ∪ 单字母 token（带/不带前导空格）（mask.mode=letterfree_plus_single）
  B      letter-free mask（think 阶段）
  Bwarm  B_warm-RL：从 B_warm SFT adapter 初始化（train.init_from=runs/Bwarm_sft_<tag>/final），letter-free mask 继续 RL；
         B_warm-SFT（无 RL 直接评估）= --eval-only 同一 adapter
  Crand  C_rand：think 段上下文 token 从 B 收敛 unigram 抽样、长度从 B 分布抽（train.crand_source），loss 仅答案
  （B_seq / B_eng / D 已删，2026-09-18 v7）
用法:
  python scripts/train.py --config configs/cloud_4b.yaml --arm A --smoke                  # 3 步 smoke（无 eval）
  python scripts/train.py --config configs/cloud_4b.yaml --arm A --dry-run                # 20 步：每步墙钟 / token / 显存 + 400 步 × 矩阵的 GPU 小时与费用投影
  python scripts/train.py --config configs/cloud_4b.yaml --arm B --tag _kk_n10_s1_s1 --set train.seed=1
  python scripts/train.py --config configs/cloud_4b.yaml --arm Bwarm --eval-only --set train.init_from=runs/Bwarm_sft_x/final
  ... --resume                                                                            # 从最新 checkpoint 恢复（云上抢占）
奖励 v4（cot_compress/rewards.py；v 只能在答案区发生）；策略 log-prob 在该臂 mask 下重归一化（cot_compress/masked_logps.py）；
LoRA：attention + MLP + embed_tokens（lm_head 不挂：tied embeddings；embed 按 results/lora_embed_sync_check.json 门控，见 cot_compress/lora.py）；lr 1e-5，10 步 warmup。
采样：训练 T=1.0 无 top-k/p；评估 T=0.6 top-p 0.95 top-k 20（generation.*）。
停止判据：连续两个 50 步区间 ΔL_mean < 5% 且 Δacc < 4pp（stopping-eval 500 题，split=stop），或 max_steps（400）；报告用 reporting-eval（06_diagnose）。
残留检查（r3；C_rand 豁免）：每 50 步在 direct-check 2000 题上测直接作答；带轨迹 acc ≥ 冻结 direct + 10pp 后生效；direct ≥ acc − (acc − 冻结 direct)/2 连续两次 → 停 run 标不可解释。
M = 准入 500 条原生轨迹中位长度（results/admission_<key>.json）。结束时写 designated_ckpt.json（收敛 step / 不可解释 step / 跑满 max_steps）。
B 的命中率（exact、Hamming ≤ 2）每次 eval 记入 eval.jsonl（hit_exact / hit_hamming2）。
NOTE: 第一个 import 必须是 cot_compress。
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import cot_compress  # noqa  unsloth first
import argparse, json, os, statistics, time
import torch
from transformers import TrainerCallback
import tasks
from cot_compress.config import load_config
from cot_compress.span import HOOK, CLOSE_TEXT
from cot_compress.rewards import make_reward
from cot_compress.evaluate import run_eval, forced_answer_eval, summarize, masked_fraction_rows, ref_ppl_rows, THINK_PREFILL
from cot_compress.generation import chat_prompt
from cot_compress.vocab_mask import banned_ids, think_allowed_ids, lift_id
from cot_compress.masked_logps import make_masked_trainer_class
from cot_compress.rollout import EXTRA_KEY
from cot_compress.guard import assert_writable, assert_checkpoint_saved
from cot_compress.lora import lora_targets_for

ARM_MASK = {"A": "none", "A2": "letterfree_plus_single", "B": "letterfree", "Bwarm": "letterfree", "Crand": "none"}   # v7：B_seq / D 已删

LOG_MD = ROOT / "results/overnight_log.md"

def log_md(text: str):
    with open(LOG_MD, "a") as f:
        f.write(f"\n- {time.strftime('%H:%M')} {text}\n")
    print("[log]", text, flush=True)

def build_dataset(task_mod, tok, key, n, seed, prefill_think=None):
    """prefill_think=None：普通臂，prompt 以 "<think>\n" 结尾。
    prefill_think=callable(item_id, rng)->list[int]：C_rand / D——think 段（非策略采样）直接烤进 prompt 并接上收尾，策略只生成答案；
    同一 prompt 的 G 条生成共享同一段 think（组内基线的状态一致）。L 记在 L_side 列给奖励。"""
    import random
    from datasets import Dataset
    from cot_compress.rollout import build_prefilled_prompt
    it = tasks.iter_train(key, seed=seed); rng = random.Random(seed)
    rows = []
    for _ in range(n):
        ex = next(it)
        text = chat_prompt(tok, task_mod.SYSTEM_PROMPT, ex["prompt"], think=True, prefill=THINK_PREFILL)   # 字符串 prompt：模板 + "<think>\n" 预填
        row = dict(prompt=text, user=ex["prompt"], answer=ex["answer"], key=key, id=ex["id"], S=int(ex["meta"].get("S", 1)))
        if prefill_think is not None:
            ids = prefill_think(ex["id"], rng)
            row["prompt"] = build_prefilled_prompt(tok, text, ids); row["L_side"] = len(ids)
        rows.append(row)
    return Dataset.from_list(rows)

class StepLogger(TrainerCallback):
    def __init__(self, path, tag):
        self.path, self.tag, self.t0, self.rows_seen, self.forced_seen = path, tag, None, 0, 0
        self.records = []; self.gen_sec = 0.0
    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        HOOK.collect(); self.rows_seen, self.forced_seen = HOOK.totals["rows"], HOOK.totals["forced"]   # 排除 eval 期间的生成
    def on_step_end(self, args, state, control, **kw):
        dt = time.time() - self.t0
        HOOK.collect()                       # TRL 的 generate 不经过 generate_texts，这里手动汇总本步的强制统计
        rows = HOOK.totals["rows"] - self.rows_seen; forced = HOOK.totals["forced"] - self.forced_seen
        self.rows_seen, self.forced_seen = HOOK.totals["rows"], HOOK.totals["forced"]
        if rows == 0:                                        # vLLM 路径没有进程内控制器：用奖励侧的 capped 比例
            from cot_compress import rewards as _rw
            if _rw.LAST.get("n"):
                rows = int(_rw.LAST["n"]); forced = int(round(_rw.LAST["capped"] * rows))
        gen_sec, self.gen_sec = self.gen_sec, 0.0
        from cot_compress import rewards as _rw
        rec = dict(step=state.global_step, sec=round(dt, 1), gen_sec=round(gen_sec, 1), train_sec=round(dt - gen_sec, 1),
                   reward_acc=_rw.LAST.get("acc"), reward_viol=_rw.LAST.get("viol"), reward_hard=_rw.LAST.get("hard"), L_mean=_rw.LAST.get("L_mean"), peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                   gen_rows=rows, gen_forced=forced, natural_end=(rows - forced) / rows if rows else None,
                   gen_steps_per_s_by_256=HOOK.last.windows(256) if HOOK.last is not None else None)
        self.records.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{self.tag} step {state.global_step}] {dt:.0f}s (gen {gen_sec:.0f}s + train {dt - gen_sec:.0f}s) alloc={rec['peak_alloc_gib']}GiB reserved={rec['peak_reserved_gib']}GiB "
              f"natural_end={rec['natural_end']} acc={rec['reward_acc']} L_mean={rec['L_mean']} viol={rec['reward_viol']} hard={rec['reward_hard']}", flush=True)
        torch.cuda.empty_cache()

class WorkspaceGuard(TrainerCallback):
    """训练开始、每次写 checkpoint 之前探测 run 目录可写；保存后核对 checkpoint 非空。不可写 → 抛错停训（不静默丢数据）。"""
    def __init__(self, run_dir, save_steps):
        self.run_dir, self.save_steps = run_dir, save_steps
    def on_train_begin(self, args, state, control, **kw):
        assert_writable(self.run_dir, "train begin")
    def on_step_end(self, args, state, control, **kw):
        if self.save_steps and state.global_step % self.save_steps == 0:      # HF Trainer 在 on_step_end 之后才保存
            assert_writable(self.run_dir, f"before checkpoint-{state.global_step}")
    def on_save(self, args, state, control, **kw):
        assert_checkpoint_saved(pathlib.Path(args.output_dir) / f"checkpoint-{state.global_step}")

class EvalCallback(TrainerCallback):
    """每 every 步 eval；判定逻辑在 cot_compress/stopping.py（可单测）：
    收敛：连续两个区间 |ΔL_mean|/L_后一次 < 5% 且 |Δacc| < 4pp（stopping-eval 500 题；step-0 eval 计入历史），或 max_steps。
    残留检查（C_rand 豁免）：fn 返回 (L_mean, acc, direct_now)；acc ≥ 冻结 direct + 10pp 后生效（锁存）；
      direct_now ≥ acc − (acc − frozen_direct)/2 连续两次 → 停 run，标不可解释（runs/<run>/uninterpretable.json）。"""
    def __init__(self, fn, every, at_start, dl_rel=0.05, dacc=0.04, frozen_direct=None, run_dir=None, arm=""):
        from cot_compress.stopping import ResidualCheck
        self.fn, self.every, self.at_start, self.dl_rel, self.dacc = fn, every, at_start, dl_rel, dacc
        self.run_dir, self.resid = run_dir, ResidualCheck(frozen_direct, arm)
        self.hist = []; self.stopped_converged = False; self.stopped_residual = False
    def _record(self, res, state, control):
        L, acc = res[0], res[1]; direct_now = res[2] if len(res) > 2 else None
        self.hist.append((L, acc))
        r = self.resid.update(acc, direct_now)
        if r["stop"]:
            json.dump(dict(step=state.global_step, acc_trace=acc, direct=direct_now, threshold=r["threshold"], frozen_direct=self.resid.frozen_direct,
                           reason="direct >= acc_trace - (acc_trace - frozen_direct)/2 twice"), open(self.run_dir / "uninterpretable.json", "w"))
            log_md(f"STOP: uninterpretable at step {state.global_step}: direct {direct_now:.3f} >= {r['threshold']:.3f} twice")
            self.stopped_residual = True; control.should_training_stop = True
    def _converged(self):
        from cot_compress.stopping import converged
        return converged(self.hist, self.dl_rel, self.dacc)
    def on_train_begin(self, args, state, control, **kw):
        if self.at_start and state.global_step == 0:
            self._record(self.fn(0), state, control)
    def on_step_end(self, args, state, control, **kw):
        if self.every and state.global_step % self.every == 0:
            self._record(self.fn(state.global_step), state, control)
            if self._converged():
                log_md(f"STOP: converged at step {state.global_step} (two consecutive intervals ΔL<{self.dl_rel:.0%}, Δacc<{self.dacc*100:.0f}pp)")
                self.stopped_converged = True; control.should_training_stop = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_1p7b.yaml")
    ap.add_argument("--arm", choices=list(ARM_MASK), required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--smoke", action="store_true", help="3 步、无 eval（λ 不变）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-eval-at-start", action="store_true")
    ap.add_argument("--tag", default="", help="run 目录后缀，如 _check（runs/{arm}_{lambda}{tag}）")
    ap.add_argument("--eval-only", action="store_true", help="不训练：加载 train.init_from 的 adapter，在该臂 mask 下跑一次 eval（B_warm-SFT 无 RL 直接评估）")
    ap.add_argument("--dry-run", action="store_true", help="20 步、无 eval；输出每步墙钟 / token / 显存峰值，并按 400 步 × 矩阵投影 GPU 小时与费用")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tcfg, gcfg, rcfg, mcfg, kcfg = cfg["train"], cfg["generation"], cfg["reward"], cfg["mask"], cfg["task"]
    key = kcfg["key"]; task_mod = tasks.task_for_key(key)
    lam = float(rcfg["lambda"])
    cap = int(tcfg["max_completion_length"]); reserve = int(tcfg.get("force_reserve", 24))
    G = int(tcfg["num_generations"])
    max_steps = 3 if args.smoke else (int(tcfg.get("dry_run_steps", 20)) if args.dry_run else int(tcfg["max_steps"]))
    run_name = f"smoke_{args.arm}" if args.smoke else (f"dryrun_{args.arm}{args.tag}" if args.dry_run else (f"evalonly_{args.arm}{args.tag}" if args.eval_only else f"{args.arm}_{lam}{args.tag}"))
    run_dir = ROOT / "runs" / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = ROOT / "samples" / run_name; samples_dir.mkdir(parents=True, exist_ok=True)

    # ---- M ----
    adm_path = ROOT / f"results/admission_{key}.json"
    if rcfg.get("m_override"):            # D8：只用于吞吐测量（--dry-run）的临时 M；正式 run 的 M 必须来自该 cap 下的准入
        M = float(rcfg["m_override"]); log_md(f"TEMPORARY M override = {M} (throughput measurement only; not a result)")
        assert args.dry_run or args.smoke, "reward.m_override is only allowed with --dry-run / --smoke"
    elif key.startswith("ord_"):            # r3：M = 准入 500 条原生轨迹的中位长度（results/admission_<key>.json["M"]），不再用 M.json 的 32 题
        assert adm_path.exists(), f"{adm_path} missing: run 01_calibrate.py --admission --keys {key} first (M comes from the 500 admission trajectories)"
        M = float(json.load(open(adm_path))["M"])
    else:
        Mrec = json.load(open(kcfg["m_file"]))[key]; M = float(Mrec["M"])

    # ---- model ----
    m = cfg["model"]; use_unsloth = bool(m.get("use_unsloth", True))
    lora_targets = lora_targets_for(cfg)                                    # v7：+ embed_tokens / lm_head（门控）
    if use_unsloth:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(model_name=m["name"], max_seq_length=int(m["max_seq_length"]),
                                                       load_in_4bit=bool(m["load_in_4bit"]), fast_inference=False, dtype=m.get("dtype"))
        model = FastLanguageModel.get_peft_model(
            model, r=int(tcfg["lora_r"]), lora_alpha=int(tcfg["lora_alpha"]), lora_dropout=0, bias="none",
            target_modules=lora_targets, use_gradient_checkpointing=tcfg.get("gradient_checkpointing", "unsloth"),
            random_state=int(tcfg["seed"]))
    else:   # 云端 TRL 原生路径：transformers bf16 + PEFT；vLLM 由 TRL colocate 管理
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        tok = AutoTokenizer.from_pretrained(m["name"])
        model = AutoModelForCausalLM.from_pretrained(m["name"], dtype=torch.bfloat16, device_map="cuda")
        if tcfg.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=int(tcfg["lora_r"]), lora_alpha=int(tcfg["lora_alpha"]), lora_dropout=0.0,
                                                 bias="none", target_modules=lora_targets, task_type="CAUSAL_LM"))
    tok.padding_side = "left"
    init_from = tcfg.get("init_from") if (args.arm == "Bwarm" or args.eval_only) else None
    if args.arm == "Bwarm" or args.eval_only:
        assert init_from, "arm Bwarm / --eval-only needs train.init_from=runs/Bwarm_sft_<tag>/final"
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd_path = pathlib.Path(init_from) / "adapter_model.safetensors"
        set_peft_model_state_dict(model, load_file(str(sd_path)))
        print(f"[init] loaded adapter from {init_from}", flush=True)

    # ---- mask（按臂）+ span hook ----
    mask_mode = ARM_MASK[args.arm]
    V = int(model.get_input_embeddings().weight.shape[0])
    lf_banned = banned_ids(tok, "letterfree", vocab_size=V)             # 描述量"禁集占比"统一用 letter-free 集
    banned = banned_ids(tok, mask_mode, vocab_size=V) if mask_mode != "none" else None
    allowed_for_logps = think_allowed_ids(tok, mask_mode, vocab_size=V) if mask_mode != "none" else None
    banned_all = lf_banned
    close_len = len(tok(CLOSE_TEXT, add_special_tokens=False)["input_ids"])
    HOOK.configure(tok, reserve=reserve, banned_ids=banned, mask_mode=mask_mode).install(model)
    force_start = cap - reserve - close_len
    gen_len = cap
    if args.arm == "Crand":                  # think 段在 prompt 里（抽样 think），策略只生成 ≤16 token 的答案
        from cot_compress.rollout import ANSWER_MAX_TOKENS
        gen_len = ANSWER_MAX_TOKENS; force_start = 10**9; HOOK.enabled = False

    # ---- reward ----
    reward = make_reward(task_mod, tok, lam=lam, M=M, cap=cap, force_start=force_start,
                         log_path=run_dir / "reward_log.jsonl", length_term=True, prefilled=(args.arm == "Crand"),
                         archive_path=run_dir / "rollouts.jsonl.zst", task_n=(task_mod.parse_key(key)["n"] if key.startswith("cups") else None),
                         kk_n=(task_mod.parse_key(key)["n"] if key.startswith("kk_") else None),
                         ord_nd=((task_mod.parse_key(key)["n"], task_mod.parse_key(key)["d"]) if key.startswith("ord_") else None))

    # ---- data ----
    prefilled_arm = args.arm == "Crand"
    crand_stats = None
    if args.arm == "Crand":
        from cot_compress.traj_stats import TrajStats, load_rows
        src = tcfg.get("crand_source"); assert src, "arm Crand needs train.crand_source=runs/<B run>"
        crand_stats = TrajStats(tok, load_rows(ROOT / src), lift_id(tok))
    prefill_fn = None
    if args.arm == "Crand":
        prefill_fn = lambda iid, rng: crand_stats.sample_unigram(crand_stats.sample_length(iid, rng), rng)
    ds = build_dataset(task_mod, tok, key, n=max(2 * max_steps, 16), seed=int(tcfg["seed"]) + 1000, prefill_think=prefill_fn)
    ds_meta = [dict(user=ex["user"], id=ex["id"]) for ex in ds] if "id" in ds.column_names else []

    from trl import GRPOConfig, GRPOTrainer
    grpo = GRPOConfig(
        output_dir=str(run_dir), max_steps=max_steps, learning_rate=float(tcfg["learning_rate"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        num_generations=G, max_prompt_length=(512 if args.arm not in ("Crand",) else 512 + cap), max_completion_length=gen_len,
        temperature=1.0, top_p=1.0, top_k=None,                       # 无 warper：屏蔽/强制不与 top-k 冲突
        loss_type=str(tcfg["loss_type"]), scale_rewards="none" if not tcfg["scale_rewards"] else "group",
        beta=float(tcfg["beta"]), logging_steps=1, save_strategy="steps", save_steps=int(tcfg.get("save_every", 25)),
        save_total_limit=int(tcfg.get("save_total_limit", 2)), bf16=True,    # 只留最近 2 个 checkpoint（省持久盘） optim="adamw_8bit", lr_scheduler_type="constant", warmup_steps=0,
        max_grad_norm=0.1, weight_decay=0.0, report_to=["tensorboard"], logging_dir=str(run_dir / "tb"),
        warmup_steps=int(tcfg.get("warmup_steps", 10)), lr_scheduler_type="constant_with_warmup",     # v7：10 步 warmup
        seed=int(tcfg["seed"]), remove_unused_columns=False, log_completions=False, mask_truncated_completions=False,
        **({"use_vllm": True, "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": float(tcfg.get("vllm_gpu_memory_utilization", 0.3)),
            # TRL 0.25.1 colocate：args.generation_kwargs 原样并入 SamplingParams → 每请求配置交给引擎级 SpanLogitsProcessor
            "generation_kwargs": ({"extra_args": {EXTRA_KEY: dict(cap=cap, reserve=reserve, mode=mask_mode)}}
                                  if args.arm != "Crand" else None)} if tcfg.get("use_vllm") else {}),
    )
    step_logger = StepLogger(run_dir / "steps.jsonl", run_name)
    callbacks = [WorkspaceGuard(run_dir, int(tcfg.get("save_every", 25))), step_logger]

    # ---- eval ----
    trainer_ref = {}
    eval_gen = dict(gcfg, batch_size=int(gcfg.get("eval_batch_size", 4)), max_new_tokens_think=cap)
    eval_items = tasks.make_eval_set(key, int(kcfg["eval_n"]), seed=int(kcfg["seed"]), **({"split": "stop"} if key.startswith(("kk_", "ord_")) else {}))   # 停止判据只用 stopping-eval
    direct_items = tasks.make_eval_set(key, int(kcfg.get("direct_check_n", 2000)), split="direct") if key.startswith("ord_") else None   # v8 残留检查：direct-check 2000 题
    frozen_direct = None
    adm = ROOT / f"results/admission_{key}.json"
    if direct_items is not None and adm.exists():
        frozen_direct = float(json.load(open(adm))["direct"])                 # 冻结 direct（direct-check 2000 题，准入第 1 条同一数）
    force_at = int(kcfg.get("eval_force_at", 1024))
    eval_backend = str(gcfg.get("backend", "hf"))
    import cot_compress.evaluate as _ev
    def do_eval(step):
        t0 = time.time()
        if use_unsloth:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        else:
            model.eval()
        try:
            tr = trainer_ref.get("t"); llm = getattr(tr, "llm", None) if eval_backend == "vllm" else None
            if eval_backend == "vllm" and llm is None:
                print("[warn] eval backend=vllm but trainer.llm unavailable; falling back to HF generate", flush=True)
            if llm is not None:
                try:
                    tr._move_model_to_vllm()            # eval 前把最新 LoRA 权重同步进 vLLM
                except Exception as e:
                    print("[warn] weight sync before eval failed:", repr(e)[:160], flush=True)
            _ev.VLLM["llm"] = llm
            was_enabled = HOOK.enabled; HOOK.enabled = True
            if args.arm == "Crand":                      # 与训练同分布地评估：think 抽样预填，策略只作答
                import random as _r
                rr = _r.Random(1234)
                th = [crand_stats.sample_unigram(crand_stats.sample_length(it["id"], rr), rr) for it in eval_items]
                from cot_compress.evaluate import prefilled_answer_eval
                rows = prefilled_answer_eval(model, tok, key, eval_items, th, eval_gen, seed=0, batch_size=eval_gen["batch_size"], tag=args.arm)
                for r in rows:
                    r.setdefault("forced", False); r.setdefault("capped", False); r.setdefault("_sec_total", 0); r.setdefault("_peak_gib", 0)
            else:
                rows = run_eval(model, tok, key, eval_items, "think", eval_gen, seed=0)
            HOOK.enabled = was_enabled
            s = summarize(rows)
            fr = forced_answer_eval(model, tok, rows, force_at, eval_gen, seed=0, batch_size=eval_gen["batch_size"])
            fs = summarize(fr)
            mf = masked_fraction_rows(tok, rows, set(banned_all))
            samp = rows[:int(tcfg.get("samples_per_eval", 10))]
            with open(samples_dir / f"step{step:04d}.jsonl", "w") as f:
                for r in samp:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            from cot_compress.archive import append_jsonl_zst          # 全量留档：本次 eval 的全部完整输出 + @force_at 强制后的输出
            append_jsonl_zst(run_dir / f"eval_step{step:04d}.jsonl.zst", rows)
            append_jsonl_zst(run_dir / f"eval_step{step:04d}_force{force_at}.jsonl.zst", fr)
            try:
                ppl = ref_ppl_rows(samp)
            except Exception as e:
                ppl = None; print("[warn] ref ppl failed:", repr(e)[:200])
            Ls = [r["think_tokens"] for r in rows]
            met = dict(step=step, acc=s["acc"], acc_clean=s["acc_clean"], viol_rate=s["viol_rate"], acc_force=fs["acc"], force_at=force_at, L_mean=round(statistics.mean(Ls), 1),
                       L_median=statistics.median(Ls), capped_rate=s["capped_rate"], fmt_err=s["format_err"],
                       mask_frac=mf, ref_ppl=ppl, eval_min=round((time.time() - t0) / 60, 1), n=len(rows), split="stop")
            from cot_compress.lengths import letter_fraction
            from cot_compress.traj_stats import think_text as _tt
            met["letter_frac"] = round(statistics.mean(letter_fraction(_tt(r["completion"])) for r in rows), 4)     # A 的字母占比曲线
            if key.startswith(("kk_", "ord_")):                           # 命中率（exact、Hamming ≤ 2）内部日志
                hd = [task_mod.hamming(r["pred"] or "", r["gold"]) for r in rows]
                met["hit_exact"] = s["acc"]; met["hit_hamming2"] = sum(1 for h in hd if h is not None and h <= 2) / len(rows); met["lenient_acc"] = s["lenient_acc"]
            direct_now = None
            if direct_items is not None and args.arm != "Crand":          # 残留检查：direct-check 2000 题直接作答（当前策略）；C_rand 豁免
                dr = summarize(run_eval(model, tok, key, direct_items, "direct", eval_gen, seed=0))
                direct_now = dr["acc"]; met["direct_acc"] = direct_now; met["direct_n"] = len(direct_items); met["frozen_direct"] = frozen_direct
                if frozen_direct is not None: met["residual_threshold"] = s["acc"] - (s["acc"] - frozen_direct) / 2
            with open(run_dir / "eval.jsonl", "a") as f:
                f.write(json.dumps(met) + "\n")
            flag = " **FLAG forced_close>20%**" if s["forced_rate"] > 0.20 else ""
            met["forced_rate"] = s["forced_rate"]
            log_md(f"eval {run_name} step {step}: acc={met['acc']:.2f} acc_clean={met['acc_clean']:.2f} viol={met['viol_rate']:.2f} acc@{force_at}={met['acc_force']:.2f} "
                   f"L_mean={met['L_mean']} L_med={met['L_median']} capped={met['capped_rate']:.2f} forced={s['forced_rate']:.2f}{flag} "
                   f"fmt_err={met['fmt_err']:.2f} mask_frac={mf if mf is None else round(mf, 3)} ppl={ppl if ppl is None else round(ppl, 1)} ({met['eval_min']} min)")
            return (met["L_mean"], met["acc"], direct_now)
        finally:
            _ev.VLLM["llm"] = None
            if use_unsloth:
                FastLanguageModel.for_training(model)
            else:
                model.train()
            torch.cuda.empty_cache()
    if not args.smoke and not args.dry_run and not args.eval_only:
        eval_cb = EvalCallback(do_eval, int(kcfg["eval_every"]), at_start=not args.no_eval_at_start, dacc=float(tcfg.get("stop_dacc", 0.04)),
                               frozen_direct=frozen_direct, run_dir=run_dir, arm=args.arm)
        callbacks.append(eval_cb)

    trainer_kwargs = dict(model=model, processing_class=tok, reward_funcs=[reward], args=grpo, train_dataset=ds, callbacks=callbacks)
    _holder = {}
    if tcfg.get("use_vllm"):
        # vLLM 0.10.2：per-request logits_processors 在 V1 / V0 都不可用；TRL 0.25.1 的 rollout_func 只接在 server 模式。
        # → colocate + 引擎级 SpanLogitsProcessor（patch LLM.__init__ 注入）+ generation_kwargs.extra_args 传每请求配置。
        from cot_compress.rollout import patch_vllm_llm_init
        patch_vllm_llm_init()
    elif args.arm == "Crand":
        pass   # HF 路径同样可用：think 已烤进 prompt
    # 策略 log-prob 在该臂 mask 下重归一化 + 强制 close 段不进损失（allowed=None → 全词表）
    TrainerCls = make_masked_trainer_class(GRPOTrainer, allowed_ids=allowed_for_logps, vocab_size=V, lift_id=lift_id(tok),
                                           force_start=force_start, close_ids=tok(CLOSE_TEXT, add_special_tokens=False)["input_ids"])
    trainer = TrainerCls(**trainer_kwargs)
    trainer_ref["t"] = trainer
    # 生成前后立刻归还缓存：4 × 3k 生成的 KV/临时块留在 reserved 里会把训练前向挤到 OOM（step1 reserved 6.55 GiB vs alloc 2.93）
    import gc
    _orig_generate = trainer._generate
    def _generate_with_cache_release(*a, **k):
        gc.collect(); torch.cuda.empty_cache()
        _tg = time.time()
        out = _orig_generate(*a, **k)
        step_logger.gen_sec += time.time() - _tg           # 每步耗时拆分：生成（含权重同步到 vLLM）vs 其余（logp 前向 + 反传 + 奖励）
        r0 = torch.cuda.memory_reserved() / 2**30
        gc.collect(); torch.cuda.empty_cache()
        print(f"[mem] after generate: reserved {r0:.2f} -> {torch.cuda.memory_reserved()/2**30:.2f} GiB, alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
        return out
    trainer._generate = _generate_with_cache_release
    from cot_compress.memory import tune_allocator
    print("[mem]", tune_allocator(), flush=True)
    log_md(f"start {run_name}: arm={args.arm} key={key} λ={lam} M={M} cap={cap} force_start={force_start} G={G} beta={grpo.beta} "
           f"max_steps={max_steps} mask={mask_mode}({len(banned) if banned else 0} banned) logp_renorm={'on' if allowed_for_logps else 'full-vocab'} "
           f"scale_rewards={grpo.scale_rewards} loss={grpo.loss_type}")
    if args.eval_only:                                                     # B_warm-SFT：无 RL 直接评估（该臂 mask 下）
        if tcfg.get("use_vllm"):
            trainer._move_model_to_vllm()
        do_eval(0); trainer.save_model(str(run_dir / "final")); log_md(f"eval-only done {run_name} (adapter {init_from})"); return
    resume = args.resume and any(run_dir.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    recs = step_logger.records
    if recs:
        ne = [r["natural_end"] for r in recs if r["natural_end"] is not None]
        log_md(f"done {run_name}: steps={len(recs)} sec/step={statistics.mean(r['sec'] for r in recs):.0f} "
               f"peak_alloc={max(r['peak_alloc_gib'] for r in recs)}GiB peak_reserved={max(r['peak_reserved_gib'] for r in recs)}GiB "
               f"natural_end={statistics.mean(ne) if ne else None}")
    if args.dry_run:
        from cot_compress.cost import dry_run_report
        rep = dry_run_report(recs, cfg, arm=args.arm, matrix_path=ROOT / str(tcfg.get("matrix_for_projection", "configs/matrix_core.yaml")))
        out = ROOT / f"results/dry_run_{args.arm}{args.tag}.json"; json.dump(rep, open(out, "w"), indent=1)
        log_md(f"dry-run {run_name}: {json.dumps(rep['summary'])}"); print(json.dumps(rep, indent=1)); return
    if not args.smoke:
        trainer.save_model(str(run_dir / "final"))
        from cot_compress.stopping import designated_ckpt           # r3：指定 checkpoint（未收敛跑满 → step = max_steps）
        dc = designated_ckpt(int(trainer.state.global_step), max_steps, eval_cb.stopped_converged, eval_cb.stopped_residual)
        json.dump(dc, open(run_dir / "designated_ckpt.json", "w")); log_md(f"designated_ckpt {run_name}: {json.dumps(dc)}")

if __name__ == "__main__":
    main()
