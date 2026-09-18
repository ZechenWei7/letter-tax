"""LoRA 目标模块（v7 §7，用户 2026-09-18 定）：attention + MLP + embed_tokens，r=32 α=64 dropout 0。
lm_head **不挂**：Qwen3-4B tie_word_embeddings=true，vLLM 加载时跳过 lm_head.weight，lm_head 的 LoRA 增量同步不进采样引擎；不做 untied 副本。
embed_tokens 只在 scripts/10_check_lora_embed_sync.py 验证"增量能同步到 vLLM 引擎"后启用：
  results/lora_embed_sync_check.json 里 variants["q+embed"].ok=True → 启用；否则全臂去掉 embed_tokens，并把原因写进 results/cloud_log.md。
train.lora_embed_lm_head: auto（默认，按检查结果）| true（强制，检查未通过则报错）| false（不挂）。
"""
from __future__ import annotations
import json, pathlib, time

BASE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EXTRA_TARGETS = ["embed_tokens"]          # lm_head 不挂（tied）
ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECK_PATH = ROOT / "results" / "lora_embed_sync_check.json"

def embed_sync_ok(check: dict | None, var: str = "q+embed") -> tuple[bool, str]:
    if not check:
        return False, "no sync check result (run scripts/10_check_lora_embed_sync.py)"
    v = (check.get("variants") or {}).get(var)
    if v is None:
        return False, f"variant {var} missing in check"
    if not v.get("ok"):
        return False, f"variant {var} failed: max|Δlogp| after sync = {v.get('vllm_vs_hf_after_sync', [None])[0]}"
    return True, f"variant {var} ok"

def lora_targets_for(cfg: dict, check_path=CHECK_PATH, log_path=ROOT / "results" / "cloud_log.md") -> list[str]:
    mode = str(cfg["train"].get("lora_embed_lm_head", "auto")).lower()
    if mode == "false":
        return list(BASE_TARGETS)
    check = json.load(open(check_path)) if pathlib.Path(check_path).exists() else None
    ok, why = embed_sync_ok(check)
    if ok:
        return BASE_TARGETS + EXTRA_TARGETS
    if mode == "true":
        raise RuntimeError(f"train.lora_embed_lm_head=true but sync check not passed: {why}")
    with open(log_path, "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} LoRA embed_tokens dropped for ALL arms: {why}\n")
    print(f"[lora] embed_tokens dropped: {why}", flush=True)
    return list(BASE_TARGETS)
