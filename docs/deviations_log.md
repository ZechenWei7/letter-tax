# Deviations log（预注册之后的全部代码改动）

预注册：OSF https://osf.io/usycb（2026-09-19 02:23:34），引用 freeze r4 `d207c263df938f6003e51ff3da90f13367b40d99`（README 填 hash 的提交 `471f0561`）。
规则：此后任何代码 / 配置改动都记在这里，带 commit hash、原因、是否影响预注册的分析或判定。

| # | 日期 | commit | 改动 | 原因 | 对预注册的影响 |
|---|---|---|---|---|---|
| D1 | 2026-09-19 | `832ec52` | `setup/sync_cloud.sh`：默认目标改为新 pod `root@195.26.233.96:27805`；push 规则放行 `results/whitelist_extra_banned.json` 与 `results/whitelist_audit.md`，并排除 `archive/` | RunPod 自动迁移换了机器（/workspace 迁移后为空，需重推重装）；原 push 规则排除整个 `results/`，会让 pod 缺少 287 个 id 的审计禁集 → 白名单变成 8638 / 8742（与预注册的 8351 / 8455 不符） | 无（基础设施；保证 pod 上的白名单与预注册一致） |
| D2 | 2026-09-19 | `347ab06` | `scripts/10_check_lora_embed_sync.py`：设置 RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT（与 TRL 0.25.1 colocate 相同）；每个变体改为单独子进程；删掉未使用的 untied 变体入口 | 首次在 GPU 上运行即 `KeyError: 'RANK'`（vLLM external_launcher 需要分布式环境变量）；一个进程内多次建引擎会重复初始化进程组 | 无（验证脚本；判定阈值与被测同步逻辑 `trl_sync` 未改） |
| D3 | 2026-09-19 | （本提交） | `scripts/10_check_lora_embed_sync.py`：追加诊断量 noise_floor（vLLM 基座 vs HF 基座）、delta_agreement / delta_corr（两个引擎各自看到的改动量是否一致）；加两条任务风格的 prompt；**原判定 `ok`（max Δlogp < 0.05）不变** | 首次运行三个变体（含只挂 q_proj 的对照）都不满足 0.05：max 0.22 来自跨引擎 bf16 噪声（对照同步后 mean 0.035，q+embed 同步后 mean 0.031），阈值无法区分“没同步”与“噪声” | 无代码路径影响；embed LoRA 的去留由用户依据两套数字决定（`cot_compress/lora.py` 的门仍读原 `ok`） |
