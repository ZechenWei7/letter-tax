# LoRA on embed_tokens / lm_head × TRL 0.25.1 colocate × vLLM 0.10.2 —— 权重能否同步（v7 硬依赖）

**结论（源码分析，待 `scripts/10_check_lora_embed_sync.py` 在 pod 上实测）**

| 情形 | 预期 | 依据 |
|---|---|---|
| LoRA 只挂注意力/MLP 线性层 | 同步 OK | TRL 逐参数 `load_weights`，名字去 `.base_layer` 后与 vLLM 权重名一致 |
| + `embed_tokens`（peft `lora.Embedding`） | 预期 OK | merge 后权重在 `model.embed_tokens.base_layer.weight` → 去 `.base_layer` → `model.embed_tokens.weight`，vLLM `VocabParallelEmbedding` 可加载；vocab 151936 已是 padded 大小 |
| + `lm_head`，基座 **tied**（Qwen3-4B `tie_word_embeddings: true`） | **同步不进去** | vLLM Qwen3 的 `load_weights` 在 tied 时跳过 `lm_head.weight`；训练侧却应用了 lm_head 的 delta → 采样器/trainer 分布不一致（importance ratio 会系统性偏离）。另：PEFT 在同一共享张量上 merge 两个 adapter（embed 的 delta 会直接出现在 lm_head 里，再叠 lm_head 的 delta）——同一张量的双重修改 |
| + `lm_head`，基座 **untied 副本**（`tie_word_embeddings=false`，`lm_head.weight` 单独物化，HF 与 vLLM 都指向该副本） | 预期 OK | 两个权重独立，`lm_head.weight` 正常加载。代价：+0.78 GB 参数（151936×2560×bf16）、训练显存与 vLLM 各多一份 |

**TRL 0.25.1 `_move_model_to_vllm`（PEFT、非 FSDP 分支）逐行要点**：`merge_adapter()` → `named_parameters()` → `name.removeprefix("base_model.model.").replace(".base_layer", "")` → 跳过含 `self.model.prefix`（"lora_"）与 `"original_module"` 的名字 → 去 `modules_to_save.default.` → colocate：`self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param.data)])` → `unmerge_adapter()` → `reset_prefix_cache()`。
注意：`modules_to_save`（全量训练 embed/lm_head 而非 LoRA）走 `original_module` 跳过 + 前缀去除，同样受 tied 限制。

**v7 定稿（用户 2026-09-18）**：Qwen3-4B 的 embed / lm_head 是 tied → LoRA **只挂 embed_tokens，lm_head 不挂，不做 untied 副本**（`cot_compress/lora.py`：EXTRA_TARGETS = ["embed_tokens"]；门控读 `variants["q+embed"].ok`）。
GPU 上仍要跑 `scripts/10_check_lora_embed_sync.py` 验证 embed_tokens 的 LoRA 增量能同步到 vLLM（变体 q_only / q+embed 须 ok；q+embed+lmhead_tied 保留为阴性演示，预期 fail）。
判定阈值：max |Δlogp| < 0.05（bf16 跨引擎噪声；未同步时为 O(1)）。`make_untied_copy` 保留在脚本里但不再调用。
