# Deviations log（预注册之后的全部代码改动）

预注册：OSF https://osf.io/usycb（2026-09-19 02:23:34），引用 freeze r4 `d207c263df938f6003e51ff3da90f13367b40d99`（README 填 hash 的提交 `471f0561`）。
规则：此后任何代码 / 配置改动都记在这里，带 commit hash、原因、是否影响预注册的分析或判定。

| # | 日期 | commit | 改动 | 原因 | 对预注册的影响 |
|---|---|---|---|---|---|
| D1 | 2026-09-19 | `832ec52` | `setup/sync_cloud.sh`：默认目标改为新 pod `root@195.26.233.96:27805`；push 规则放行 `results/whitelist_extra_banned.json` 与 `results/whitelist_audit.md`，并排除 `archive/` | RunPod 自动迁移换了机器（/workspace 迁移后为空，需重推重装）；原 push 规则排除整个 `results/`，会让 pod 缺少 287 个 id 的审计禁集 → 白名单变成 8638 / 8742（与预注册的 8351 / 8455 不符） | 无（基础设施；保证 pod 上的白名单与预注册一致） |
| D2 | 2026-09-19 | `347ab06` | `scripts/10_check_lora_embed_sync.py`：设置 RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT（与 TRL 0.25.1 colocate 相同）；每个变体改为单独子进程；删掉未使用的 untied 变体入口 | 首次在 GPU 上运行即 `KeyError: 'RANK'`（vLLM external_launcher 需要分布式环境变量）；一个进程内多次建引擎会重复初始化进程组 | 无（验证脚本；判定阈值与被测同步逻辑 `trl_sync` 未改） |
| D3 | 2026-09-19 | `ee45149` | `scripts/10_check_lora_embed_sync.py`：追加诊断量 noise_floor（vLLM 基座 vs HF 基座）、delta_agreement / delta_corr（两个引擎各自看到的改动量是否一致）；加两条任务风格的 prompt；**原判定 `ok`（max Δlogp < 0.05）不变** | 首次运行三个变体（含只挂 q_proj 的对照）都不满足 0.05：max 0.22 来自跨引擎 bf16 噪声（对照同步后 mean 0.035，q+embed 同步后 mean 0.031），阈值无法区分“没同步”与“噪声” | 无代码路径影响；embed LoRA 的去留由用户依据两套数字决定（`cot_compress/lora.py` 的门仍读原 `ok`） |
| D4 | 2026-09-19 | `7fd0ec0` | `scripts/10_check_lora_embed_sync.py`：追加 true_noise_floor（vLLM 基座 vs HF `disable_adapter`）、sampler_vs_trainer（同步后的 vLLM vs 训练端“adapter 激活未合并”前向）、hf_merged_vs_trainer_forward | D3 的“HF 基座”实为 adapter 激活未合并的前向（= 训练端算 logp 的前向）；要判断的量是采样端与训练端是否同分布，而不只是权重有没有拷进 vLLM | 无代码路径影响；用于决定 embed LoRA 去留 |
| D5 | 2026-09-19 | （本提交） | `configs/cloud_4b.yaml`：`train.lora_embed_lm_head: false` —— **全臂去掉 embed_tokens LoRA**，LoRA 目标 = attention + MLP（q/k/v/o/gate/up/down），r=32 α=64 不变 | `scripts/10_check_lora_embed_sync.py` 在 A100 / vLLM 0.10.2 / TRL 0.25.1 / peft 0.21.0 上未通过：权重确实拷进了 vLLM（同步后 vLLM vs HF 合并 = 噪声底 mean 0.033），但 Qwen3-4B tied embeddings 下，TRL 的 merge→同步把 embed 的增量写进共享张量，采样端在输出头也带上了增量，而训练端（同步后 unmerge、adapter 激活未合并）的前向只在输入嵌入处有：采样端 vs 训练端 max / mean \|Δlogp\| = 0.965 / 0.186（只挂 q_proj 的对照 0.208 / 0.031；纯跨引擎噪声底 0.19–0.25 / 0.035）。原判定 `ok`（max<0.05）对三个变体都为 False（该阈值低于噪声底，对照也不过）；结论依据 D4 的 sampler_vs_trainer | 属预注册写明的门控分支（“不通过 → 全臂去掉 embed LoRA 并记日志”）；对所有臂一致，不影响臂间比较；臂 B 少了嵌入层的可训练自由度（可能更难冷启动，B_warm 条件分支不变） |
