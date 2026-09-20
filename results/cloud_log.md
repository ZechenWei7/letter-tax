# Cloud log — RunPod A100 80GB, 2026-09-17（机器 1：216.81.248.126:10729 global volume，已弃；机器 2：185.216.23.206:28309 legacy volume disk）

## 环境
- A100-SXM4-80GB 81920 MiB，driver 580.159.04（CUDA 13.0），Python 3.12.3，uv 已装，镜像自带 torch 2.8.0+cu128；256 核 / 2 TB RAM；容器盘 50 GB。
- **/workspace 挂载失效**：`fuse.geesefs`（S3 后端 global volume）报 "Transport endpoint is not connected"，容器内无 geesefs 进程 / 凭据，无法自行重挂；需在 RunPod 控制台重启 pod（会清空容器盘）。
- 临时决定：为不让 GPU 空转，先在容器盘 `/root/cot-compress`（HF_HOME=/root/hf）走安装→环境检查→审计→vLLM 验证→准入校准；每步结果 pull 回本地。卷恢复后迁回 /workspace。
- 提醒：geesefs 是 S3-FUSE，venv 放上面 import 会很慢；建议 venv 留容器盘、uv 缓存与 HF 缓存放 /workspace（待用户定）。

## 代码改动（云端 bug / 适配）
- `setup/sync_cloud.sh`：默认目标 / 密钥 / REMOTE_DIR；push 排除 `results/*`（本地 1.7B 的 M.json、校准结果不得带上云）；新增 `ssh` 子命令。
- `scripts/00_check_env.py`：unsloth 可缺省；按实际 compute capability 检查 arch（原来写死 sm_120）；新增 `--hf-model`（transformers bf16 加载 + 生成）；版本表加 vllm / zstandard；`--skip-model` 也写 env_check.json。
- pod restart 后 /workspace（geesefs）恢复：可读写，200 小文件写 0.21 s / 读 0.09 s。按原方案全部放 /workspace（项目、venv、HF_HOME、uv 缓存）。
- `scripts/00_check_env.py`：unsloth 可缺省；按实际 compute capability 查 arch（原写死 sm_120）；新增 `--hf-model`；版本表加 vllm/zstandard；`--skip-model` 也写 env_check.json。`setup/install_cloud.sh`：HF_HOME=/workspace/hf、UV_CACHE_DIR=/workspace/.uv-cache、UV_LINK_MODE=copy。新增 `setup/env_cloud.sh`。
- global volume（geesefs）不支持 chmod / 硬链接 → uv/pip 建不了 venv（`failed to set permissions … Operation not permitted`）。用户决定换 **legacy volume disk** 重新部署（SSH 会变）。失败的安装残留已清。
- 保护（用户要求）：`cot_compress/guard.py`（assert_writable 写-读-删探针 + fsync；assert_checkpoint_saved）；`train.py` 加 `WorkspaceGuard`（训练开始、每次写 checkpoint 前探测，保存后核对非空，不可写即抛错停训）、`save_total_limit=2`；`run_matrix.py` 每个 run 启动前探测 runs/ samples/ results/，不可写 exit 3；`setup/auto_pull.sh`（本地 watcher，云端 eval.jsonl / matrix_log / admission 变化即自动 pull）。`sync_cloud.sh` rsync 改为 `-rlvz --checksum --no-perms --no-owner --no-group --no-times`（FUSE / 块设备通吃）。

## 机器 2（185.216.23.206:28309，legacy volume disk）
- /workspace = /dev/vdb xfs 100 GB，chmod / 硬链接 / 符号链接均可，500 小文件 14 ms；A100-SXM4-80GB，driver 570.172；Python 3.12.3，uv 0.9.0。全部放 /workspace（项目、venv、HF_HOME、uv 缓存）。去掉 UV_LINK_MODE=copy。
- 安装 #1 失败：`vllm>=0.10` 解析到 vllm 0.29.0 + torch 2.12.1（CUDA 13 构建），driver 570（CUDA 12.8）报 "driver too old"；且 unsloth 把 trl 1.13.0 降到 0.24.0。TRL 1.13 的 vllm extra 本身也要求 vllm<=0.28。
- 改 `setup/install_cloud.sh`：固定 **torch 2.8.0(cu128) / vllm 0.10.2 / trl 0.25.1 / transformers<5**，不装 unsloth（`COT_NO_UNSLOTH=1`）。vllm 0.10.2 仍有 V0 引擎，per-request logits_processors 的退路可用。
- 准入校准加 `--backend vllm`：`evaluate._gen` 统一分发（VLLM["llm"] 设置后 run_eval / forced_answer_eval / prefilled_answer_eval 全走 vLLM；屏蔽与 reserve 以 HOOK 配置为准）。理由：n≥500 × 4 格 × (原生 + 屏蔽原生 + 12 次强制/填充/移植) 用 HF generate 约 3 h，vLLM 约 30–40 min。标准不变。另存每格原生轨迹 results/calib/admission_<key>__native.jsonl，结果里加 native_clean / viol_rate / forced_rate。
- 新增 `scripts/09_check_vllm_processor.py`（§9.2 第 4 步：per-request logits_processors + 强制收尾 + 臂 B 零字母 + vLLM vs HF 的 logp 一致性，并判断 vLLM 返回的是 processed 还是 raw logprobs）。
- §9.2 步 2–3：环境检查通过（torch 2.8.0+cu128 / A100 79.25 GiB / trl 0.25.1 `rollout_func: True` / vllm 0.10.2；Qwen3-4B bf16 加载 7.50 GiB，HF 单流 27 tok/s）。白名单审计：原始 8638，标记 287（NFKC 含字母 236、带圈/括号拉丁 80、区域指示符 26、leet 3：` !$`/`@$`），无数字 token 被误标。`00_check_env.py` 的 bnb 小节改为可选。
- §9.2 步 4（vLLM per-request logits_processors）：**V1 拒绝**（"V1 does not support per request user provided logits processors"）；**退路 1 级 VLLM_USE_V1=0 也不可用**——0.10.2 的 V0 引擎无条件拒绝（llm_engine.py:676，报错文案是 "multi-step" 但没有条件分支；`num_scheduler_steps` 参数已不存在）。
- 在退到 2 级（HF generate）之前改走 **vLLM V1 引擎级自定义 LogitsProcessor**（0.10.2 自带接口）：`rollout.SpanBatchState`（引擎无关状态机，单测覆盖）+ `vllm_v1_proc.SpanLogitsProcessor`（薄封装）；每请求配置走 `SamplingParams.extra_args["cot_span"]={cap,reserve,mode}`；`engine_kwargs()` 给 LLM 注册处理器并设 `logprobs_mode=processed_logprobs`；TRL colocate 的 `LLM(...)` 不透传参数 → `patch_vllm_llm_init()` 包一层注入。语义与 SpanController 完全一致；速度仍是 vLLM 级。
- §9.2 步 4 通过（V1 引擎级处理器）：臂 A 4/4 到顶强制收尾；臂 B think 段禁集 token 0、自然 `</think>` 后格式正确作答。logp：vLLM 返回 processed(masked) logprobs，与 trainer 侧 mask 重归一化的 |Δ| 均值 0.070 / 最大 0.27（n=12），对未屏蔽 HF logp 的 |Δ| 均值 11.9 nat。**未退到 HF generate**，每步耗时不受影响。
- 准入校准启动：cups_n6_k12 / n6_k6 / n8_k16 / n8_k8 / n8_k24 / n8_k12，n=500，vLLM 后端。
- 准入 #1 秒崩：`run_admission` 定义在 `if __name__` 之后（NameError），已把 main 守卫移到文件末尾并重启。
- TRL 0.25.1 源码核对后的训练路径改动：(1) `rollout_func` 只接在 **server** 模式分支，colocate 不调用，且它收到的是去重后的 prompts → 弃用；(2) colocate 分支把 `args.generation_kwargs` 原样并入 `SamplingParams` → 用 `generation_kwargs={"extra_args": {"cot_span": {cap,reserve,mode}}}` 把每请求配置交给引擎级处理器，`patch_vllm_llm_init()` 给 TRL 自建的 `LLM(...)` 注入 `logits_processors` 与 `logprobs_mode`；(3) `vllm_max_model_len` 不是 0.25.1 的字段（TRL 自己用 max_prompt+max_completion）→ 删；(4) C_rand / D 改为**数据集预填**：think 段（D 为空、C_rand 抽样）烤进 prompt 并接 `</think>\n\n<answer>`，策略只生成 ≤16 token 的答案，L 走数据集列 `L_side`，同一 prompt 的 G 条共享同一段 think（组内基线状态一致）；(5) 训练内 eval 经 `evaluate.VLLM["llm"]=trainer.llm` 全走 vLLM，eval 前 `_move_model_to_vllm()` 同步权重；(6) micro-batch 2×8（mask 重归一化要物化 [B,T,V] fp32 logits）。`vllm_importance_sampling_correction` 默认 True：采样器 logprob（processed）与 trainer 的 masked logp 同分布。
- `admission.py` 判定修正：预算曲线上无任何点到 native−5pp 时，最小预算 >1.0M，应判**满足** ≥0.5M（原先误判为不满足）。标准文字未变；已在跑的准入进程输出的 verdict 会在汇报时用修正后的函数从指标重算。

## 准入校准结果（冻结 Qwen3-4B，n=500，cap 3072，vLLM V1 + 引擎级处理器；19:32–21:01 UTC，6 格 89 min）
| cell | direct | native | M | forced@cap | budget 0.1/0.25/0.5/0.75/1.0 ×M | filler max(≥0.5M) | transplant | mask 冻结 acc_clean | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| cups_n6_k6 (K/2) | 0.080 | 0.976 | 1404 | 0.032 | .070/.186/.476/.818/.924 | 0.074 | 0.082 | 0.002 | 拒 #2 |
| cups_n6_k12 | 0.068 | 0.868 | 2156 | 0.158 | .066/.126/.376/.716/.806 | 0.068 | 0.102 | 0.084 | 拒 #2（native>80%） |
| cups_n8_k8 (K/2) | 0.058 | 0.946 | 1991 | 0.134 | .058/.168/.536/.802/.888 | 0.048 | 0.096 | 0.092 | 拒 #2 |
| cups_n8_k12 | 0.064 | 0.844 | 2347 | 0.248 | .066/.136/.440/.700/.810 | 0.070 | 0.068 | 0.086 | 拒 #2（native>80%） |
| **cups_n8_k16** | 0.052 | **0.776** | 2874 | **0.438** | .080/.124/.386/.670/.778 | 0.058 | 0.076 | 0.062 | **通过（#7 待 D 臂）** |
| cups_n8_k24 | 0.060 | 0.568 | 3072 | 0.714 | .060/.084/.186/.390/.570 | 0.066 | 0.052 | 0.070 | 拒 #2（native<60%） |
- 注意：cups_n8_k16 的原生轨迹 43.8% 到顶被强制收尾（L p50=2876，p75=p90=3072）；强制的准确率 0.64 vs 自然结束 0.88 → native 与 M 都受 cap 影响，臂 0 就会触发 ">20% 强制收尾" flag。
- GPU 空档利用：启动可选的准入第二批（cups_n10_k12 / n12_k12 / n10_k14：用 n 而不是 k 提难度，轨迹更短、触顶更少；mul_4x3 / mul_4x4），日志 /workspace/admission2.log，可随时 kill。

## 配置变更（用户，2026-09-17）：云端默认 cap 3072 → **4096**
- cap 3072 是本地 8GB 的遗留；4B 云端阶段 eval 与训练统一 cap 4096（`configs/cloud_4b.yaml`：`train.max_completion_length=4096`、`generation.max_new_tokens_think=4096`、`model.max_seq_length=5120`）。第一、二批准入（cap 3072）的结果保留作筛选；入选格按 cap 4096 重测 native / M / 强制收尾率（`01_calibrate.py --admission --native-only`，写 `results/retest_<key>_cap4096.json`），M 取重测的原生中位数。
- 判定规则：第二批里 native∈[60,80] 且其余标准通过的格 → cap 4096 重测，仍在窗口且强制收尾 <20% 则选它；否则 cap 4096 重跑 cups_n8_k16 与 cups_n8_k20；都不行再讨论接受 n8_k16。mul 两格里选 native 在窗口内的作 mask 覆盖率对照，同样按 cap 4096 记 M。
- 训练配置（用户）：每步 2 个 prompt × G=16 = 32 条 → `per_device_train_batch_size=2`、`gradient_accumulation_steps=16`（8B：1×32）。`steps.jsonl` 每步加 `gen_sec`（生成 + 权重同步）/ `train_sec`（logp 前向 + 反传 + 奖励）拆分，A seed 1 启动后据此报每步耗时构成。候选格若多个通过，优先 M 更短的。
- mul_4x3 在 4B 上 native 99.8%、direct 12.2%（太易）；追加 cap 4096 的更难 mul 探测（mul_5x5 / 6x6 / 7x7，native-only + direct），排在 cups 重测之后，GPU 不空转。native-only 记录里加 direct。
- 第二批准入（cap 3072）：cups_n10_k12 86.0% / n12_k12 83.2% / n10_k14 81.0%（都 >80%，且提 n 只让轨迹变长：M 2793–3019）；mul_4x3 99.8%（direct 12.2%）、mul_4x4 98.4%（direct 2.4%），**mul 在冻结 mask 下 acc_clean 69.0% / 57.4%** → mask 对乘法几乎不构成约束（§2.1 的低语言参与端），两格都不在窗口。
- cap 4096 重测（规则 3）：cups_n8_k16 native **84.4%**（出窗口）M 2822.5 强制 13.4%；cups_n8_k20 native **72.4%**（在窗口）M 3213.5 **强制 24.0%（≥20%）**。两格都不干净。追加：更难 mul 探测（5x5/6x6/7x7）→ cups_n8_k18 重测；若 k18 在窗口且强制 <20%，自动跑 cups_n8_k18 + K/2（cups_n8_k9）的 cap 4096 完整准入。

## 格选择判定树（用户，2026-09-17）
1. cups_n8_k18（cap 4096）在窗口且强制 <20% → 选它，跑完整 7 项准入 + K/2（cups_n8_k9），M 取 cap 4096 原生中位数。
2. 否则选 cups_n8_k20，cap 提到 5120 重测 native-only（k20 自然结束准确率 77.6%，放宽 cap 不会推出窗口，只会压低强制收尾；与 k16 相反）；仍 ≥20% → 6144。cap 是配置参数，记日志即可。选定后同样跑完整准入 + K/2（cups_n8_k10）。
3. 都不行才回到 n8_k16 讨论。"强制 <20%"是当日定的配置门槛，不是预注册标准。
- 实现：pod 上 `/workspace/chain5.sh`（排在 mul 探测之后自动执行，日志 chain5.log）。
- **mul 对照**：冻结模型在 letter-free mask 下 acc_clean 69.0%（mul_4x3）/ 57.4%（mul_4x4）= mask 对乘法不构成约束，直接写进结果表。mul 的训练 run 只在 mul_5x5–7x7 有格进窗口时才跑；都不在窗口 → 砍掉 mul 两个 run（裁剪顺序第一位），只报告冻结数据。
- mul 探测（cap 4096）：mul_5x5 native 96.0%（强制 40%）、mul_6x6 86.8%（强制 77%）、mul_7x7 68.8%（**强制 94%**，M=cap；自然结束 90%）→ 7x7 名义在窗口但纯属 cap 伪象；建议砍掉 mul 两个训练 run，只报告冻结数据（mask 下 69.0% / 57.4%）。
- 判定树结果：[1] cups_n8_k18 cap 4096：native 79.2%、强制 22.2%（不干净，自然结束 83.8% 会随 cap 出窗口）；[2] **cups_n8_k20 cap 5120：native 74.2%、强制 11.8%、M 3247、direct 4.6% → 选定**。随后自动跑 cups_n8_k20 + cups_n8_k10（K/2）的 cap 5120 完整准入。
- **配置：云端 cap 4096 → 5120**（`cloud_4b.yaml`：max_completion_length / max_new_tokens_think 5120，max_seq_length 6144，vllm_gpu_memory_utilization 0.35→0.45 以容纳 32×5.6k 的 KV）；task.key=cups_n8_k20；matrix 的 cups key 改为 cups_n8_k20。
- **选定 cups_n8_k20，cap 5120，完整准入通过（#7 待 D 臂）**：direct 4.6%，native 74.8%（clean 74.8%，viol 0.8%，强制 12.4%），M 3295.5；预算曲线 6.8/10.0/34.4/58.2/67.8%（最小预算 >1.0M）；filler ≤6.4%；transplant 6.6%（降 68pp，需 ≥34pp）；冻结+mask 10.8%（≤14.6%）。K/2 = cups_n8_k10：native 94.6%，M 2215.5，filler ≤6.6%，transplant 8.4%，mask 12.2%。
- **M = 3278.0**（cap 5120 两次原生运行合并 n=1000 的中位数：准入运行 3295.5、重测运行 3247.0；均值 3396，p25 2580，p75 4108；合并强制收尾 12.1%，acc 74.5%），已写 pod 上 results/M.json（cups_n8_k20）。smoke 未启动，等确认。

## 2026-09-18 设计改动（用户确认）+ smoke
- 确认 cups_n8_k20 / cap 5120 / M=3278（合并 n=1000）；mul 训练 run 砍掉，7x7 不算进窗口，只报冻结数据。
- **删臂 D**：结果不改变读法（能否内化是任务—模型匹配问题），其风险由诊断 1 逐臂检测。准入第 7 条 → 每臂收敛后诊断 1（force-close at 0 的直接作答）> 冻结 direct+10pp 则该臂长度结果标记不可解释（`analyze.py::diag1_flags`）；匹配准确率中止规则去掉 "within 10pp of D"。代码：train.py 删 D 分支、admission.py、analyze.py、matrix v5（A s1 门 → B×5 → A s2–5 → A″×3 → B_seq×2 (init=auto: 预注册端点下最佳 A seed) → C_rand×2 (crand_source=auto, B 全部完成后) → B s1 β=0.01 复跑）、裁剪 C_rand → A″→2 → A/B→3；eval n=1000 每 50 步；vLLM `max_num_seqs` 提到 128 供 eval。单测 42/42。
- smoke A 启动（3 步，λ=0.5，2×16=32 条/步，cap 5120）。

## 2026-09-18 任务定义改动（用户，训练前）
- cups 答案改为**完整终态**："6D 4U 8D 5D 3D 1D 2D 7D"（位置 1–n 顺序，杯号+U/D，空格分隔），exact match；答案区上限 32 token（n=8 终态实测 29 token，只余 3 个）；chance = 1/(n!·2^n) ≈ 1e-7。理由：单点查询允许"只反向追踪查询位置"的捷径。K/2 格同样。`tasks/cups.py`、`rewards.MAX_ANSWER_TOKENS=32`、`rollout.ANSWER_MAX_TOKENS=40`、诊断 5/6 改为全状态比对。
- 新描述量 `cot_compress/style.py`：每条轨迹"每步重写全状态 / 只写改动位置 / 混合"（按每步提到的位置数：≥0.75n 全写，≤0.25n 只写改动），进准入结果与诊断表。
- smoke A（旧任务）保留作参考：step 1 = 178 s（gen 70 + train 108），micro-batch 2 时训练峰值 73.9 GiB / reserved 78.4 → 已改 1×32。任务改后 smoke A/B 各再跑 1 步确认解析与奖励。
- 重跑准入：cups_n8_k12 / k16 / k20 + K/2（k6 / k8 / k10），cap 5120，n=500，第 7 条不在此时判。单测 44/44。
- **留档改为全量**（用户）：`cot_compress/archive.py`（jsonl.zst 逐 frame 追加，stream 读）；奖励函数每步把全部 rollout（prompt id、完整文本、token ids、c/v/hard/violations/L/capped/r/pred、重写风格）追加到 `runs/<run>/rollouts.jsonl.zst`；每次 eval 的全部输出与 @1024 强制输出存 `eval_step<N>.jsonl.zst` / `eval_step<N>_force1024.jsonl.zst`；诊断中间产物（native 测试/选择集、5 个预算点、移植、unigram、填充、前缀 0.25/0.5/0.75、反事实）存 `diag_*.jsonl.zst`；`sync_cloud.sh pull` 全部拉回。准入原生轨迹已在 results/calib/*.jsonl。单测 45/45。
- 准入第 4 条操作化：只在 native 预算曲线 ≥ chance+10pp 的预算点比较填充（完整终态下 native 在 0.5M/0.75M 处本身 ≈0，"10pp 以内"空成立导致 k12 误拒）。标准文字不变；正在跑的进程 verdict 事后用修正函数重算。
- cups_n8_k12（完整终态）原生违规 27.2% 拆解：multi_answer 117（23.4%，其中 109 条最终答案宽松提取正确——思考内写了与终态不同的草稿状态）；no_answer 19（3.8%，全部是到顶强制后答案被截断：force_reserve=24 装不下 30 token 的终态答案 → 改 40）。无逃逸类、无长度类；答案区全部恰 30 token。待用户决定"唯一性"是否只看 `</think>` 之后的标签（若是，k12 native ≈87% 出窗口，需重跑准入）。

## 2026-09-18 用户决定：唯一性只看 `</think>` 之后；force_reserve=40；准入重跑 → A s1 → 门 → B s1
- `rewards.parse_answer/violations`、`parsing.pre_answer`：答案区 = 第一个 `</think>` 之后；思考内 `<answer>` 草稿一律忽略；multi_answer = `</think>` 后多个不同值；text_after_think = `</think>` 与其后第一个 `<answer>` 之间非空白；answer_too_long = `</think>` 后首个 `<answer>` 到末尾 >32 token。单测 45/45。
- `06_diagnose.py --backend vllm`（合并 adapter → /workspace/tmp_merged/<run> → vLLM 生成；HF 副本只用于线性探针）；run_matrix 的诊断命令带 `--backend vllm`。
- 链：准入 cups_n8_k{12,16,20}+K/2（cap 5120，n=500）→ 自动选格（窗口内、强制<20%、M 最短）→ 写 M.json、改 matrix key → `run_matrix --max-runs 2`（A s1 → 操纵检查 → B s1，各自收敛 + n=1000 诊断）。
- 02:10 UTC：chain6 启动（/workspace/chain6.sh，日志 chain6.log）。杀旧准入时其 vLLM EngineCore 子进程成为孤儿占 70 GB，已 kill -9；此后 kill 流程只按 PID / "EngineCore" 匹配，不用会自匹配的 pkill 模式。

## 2026-09-18 cups 退出结果表（用户，v7 设计前的准备）
- 已停 pod 上全部 GPU 任务（chain6：准入 v2 跑到 k16；A seed 1 / B seed 1 未启动）。所有 cups 结果归档到 `results/cups_archive/`（`local/`：本地 1.7B 阶段的校准表、M、overnight_log、calib；`cloud/`：4B 准入表 v1/v2、cap 重测、mul 冻结数据、白名单审计、smoke 日志、runs 曲线；`local_runs/`、`local_samples/`）。
- **cups 退出结果表，原因：状态类内容的无字母编码由构造决定，见 v7 设计。** 降为附录里的流水线验证。
- 准入 v2（新唯一性规则）已得两格：cups_n8_k12 native 90.4%（M 3029），cups_n8_k16 native 83.8%（M 3823）——都在窗口之上；k20 未跑完。存档供附录。
- 2026-09-18 准备项 2/3 完成（不占 GPU）：docs/kk_prior.md（K&K + Logic-RL 生成器/难度/模板总结，源码核对 AlphaPav/mem-kk-logic lib_kk.py）；scripts/10_check_lora_embed_sync.py + docs/lora_embed_sync.md（LoRA embed_tokens/lm_head → vLLM 同步验证脚本，未跑；源码分析：Qwen3-4B tie_word_embeddings=true 时 lm_head delta 不会同步进 vLLM，需 untied 副本）。pod 未能从内部 Stop（runpodctl 无 API key），待用户在控制台 Stop。
- 2026-09-18 v7 实现（CPU，未训练）：tasks/kk.py（两组 S=1 → S=2 构造，均匀拒绝采样 S=2 接受率 <0.03%，见文件头）、data/kk/kk_n10_s2_seed0.json / kk_n12_s2_seed0.json（train 2000 / stop 500 / report 500，规范形零重叠，S 全为 2，oracle 文字数均值 12 / 14，形式分布 x=y / x!=y 各约 40%）；templates.py / strategy.py / warm.py（docs/warm_map.md）/ decoder.py / endpoints.py / lora.py / cost.py；admission v7 八条；train.py 臂 A/A2/B/Bwarm/Crand + --eval-only + --dry-run；run_matrix 门（含 2× oracle）与 λ=0.3→1.0→G=32 重调；analyze E1/E2/E3；configs 切到 kk_n10_s2；79 项单测通过。待 GPU：scripts/10（embed/lm_head 同步）、01_calibrate --admission、--dry-run。
- 2026-09-18 v7 改：主格改为天然 S=1（均匀构造 + 拒绝，删两簇拼接与 S=2 数据），key kk_n10_s1 / kk_n12_s1，重建三划分；准入八条按用户定稿（含 M ≥ 5× oracle(c=2.5)）；解码器 k∈{0,1}，k=1 仪器检查；LoRA 只挂 embed_tokens（Qwen3-4B tied，lm_head 不挂，不做 untied 副本；GPU 上仍需 scripts/10 验证 embed 增量同步）；操纵门 oracle c=2.5。
- 2026-09-18 v8 实现（CPU，未训练）：主任务 ordering（tasks/ordering.py，引导式生成：一致边取未被蕴含的覆盖边、不一致边均匀；纯均匀接受率 ≈ 0），四格划分 data/ordering/*（格统计：(8,4,5)(8,4,6)(9,5,6) 因单元素步占比 >0.8 拒，(9,5,7) 过）；shortcuts.py（捷径 / 策略类 + 审计脚本）；warm 扩展；解码器前缀对齐 + 真值标签；准入八条 + 格预筛；train.py 残留检查；矩阵 v8；analyze E1 按题配对 bootstrap 三单位。K&K 退出主线（代码 / 测试保留）。
- 2026-09-19 新 pod（195.26.233.96:27805；/workspace 迁移后为空，已重推重装，HEAD 5d97d3d）。00_check_env 通过（torch 2.8.0+cu128 / vllm 0.10.2 / trl 0.25.1 / transformers 4.57.6 / peft 0.21.0）。LoRA 同步检查：权重能拷进 vLLM（同步后 vs HF 合并 mean|Δlogp| 0.033 = 噪声底 0.035），但 tied embeddings 下采样端 vs 训练端前向 q+embed = 0.186 / max 0.965（对照 q_only 0.031 / 0.208）→ 全臂去掉 embed_tokens LoRA（D5）。白名单复核 B=8351、A″=8455（raw 8638 − 审计 287）。
- 2026-09-19 准入 ord_n8_h4_d5（冻结 Qwen3-4B，stop 500，cap 5120）：未通过。direct 0.000（2000 题；**无效**：max_new_tokens_direct=16 截断，D6 已修为 32，待 --direct-only 重测）、宽松 0.0045；native 0.324（严格 = 宽松）；M=5120=cap，强制收尾 91.6%；自然结束 42 条全对，长度 min/p25/median/p75/max = 3243/3888/4458/4754/5065；被截断 458 条准确率 0.262。预算曲线 0.1/0.25/0.5/0.75/1.0 = 0.000/0.004/0.054/0.178/0.320；填充 0.004/0.000/0.002/0.004/0.004；移植（derangement）0.000；letter-ban 0.010；捷径命中 0.288（guess_verify 0.280、ir_copy_order 0.008、perm_enum 0.006、assign_enum 0.004）；Kahn / 决策下限中位 30 / 32.5 token（c=2.5）；逐对基线 0.666。判定：第 2、4、7 条不过（第 4 条为 0.5M 处 native≈0 的空成立）。
- 2026-09-19 (8,4,5) 被截断的 458 条在 5120 处的状态（文本特征 + 抽 10 条人工看）：78.6% 已写出过至少一个完整候选顺序（中位 2 个、去重 1 个）；金标准顺序出现过的占 25.3%（末 1500 字符内 19.2%）；末 1500 字符内有完整顺序的 50.2%；"wait" 中位 13 次。被截断且答对的 120 条里 70% 金标准顺序就在末尾（在逐条验证 / 复核时被截断）；被截断且答错的 338 条里只有 4% 见过金标准顺序——多数还在逐条析取分情况 / 传播，或在改第一个错误候选。抽样 10 条：2 条处于"已得到正确顺序、在逐条验证"，1 条在验证一个错误候选，7 条仍在传播 / 分情况 / 重建顺序。样本存 samples/admission_8_4_5/（3 条自然结束：最短 / 中位 / 最长；3 条 guess_verify 命中；10 条截断样本的尾部）。
- 2026-09-19 用户指示：链继续跑 (8,4,6)；若其截断率 > 80% 则停链、不跑 (9,5,6)/(9,5,7)，改跑 (8,4,5) native-only cap 8192 n=200（/workspace/logs/chain2.sh）。不改准入规则、不改检测器。
- 2026-09-19 (8,4,5) guess_verify 命中的 140 条按"候选顺序之间是否有传播 / 分情况标记"分类（只报不改；标记 = strategy.ROLES 的 case_splits ∪ propagation 词与其符号；顺序正则同检测器）：
  A 相邻"带验证词的完整顺序"（检测器自己计数的那些）之间：每个间隔都有标记 96 条 / 至少一个间隔无标记 44 条（无标记间隔共 53 个，中位仅 44 字符，79% 两侧是同一个顺序的重述）。
  B 相邻所有完整顺序之间：全有标记 65 / 至少一个无标记 75。
  C 同 A 但只看两侧顺序不同的间隔：无此类间隔（只有一个候选被反复验证）67 / 全有标记 65 / 部分无标记 7 / 全无标记 1。
  D 同 B 但只看两侧顺序不同：无此类间隔 42 / 全有标记 76 / 部分无标记 20 / 全无标记 2。
- 2026-09-19 准入 ord_n8_h4_d6（stop 500，cap 5120，direct 已用修复后的 32 token）：未通过。direct 0.0015（2000 题，严格 = 宽松，格式错 0；stop500 0.002；阈值 2×2^−6 = 0.0312）；逐对 direct 0.606 vs 基线 0.670；native 0.226（严格 = 宽松）；M=5120=cap，强制收尾 94.2%；自然结束 29 条准确率 0.966，长度 min/median/max = 3238/4588/5066；被截断准确率 0.18。预算曲线 0.010/0.004/0.022/0.124/0.222；填充 0.002/0/0/0/0.002；移植 0.000；letter-ban 0.006；捷径命中 0.158；Kahn / 决策下限中位 30 / 35 token。第 2、4、7 条不过。截断率 > 80% → 按用户指示停链，未跑 (9,5,6)/(9,5,7)。
- 2026-09-19 (8,4,5) native-only 重测，cap 8192，n=200：自然结束 86/200 = 43.0%（强制收尾 57.0%）；准确率 全部 0.535 / 自然结束 0.977 / 被截断 0.202；严格 = 宽松，无格式违规。自然结束长度 min/p10/p25/median/p75/p90/max = 3590/4155/5211/6191/7262/7841/8134；含截断的全体 L_mean 7340，p10 5081，p25 6680，p75 = p90 = M = 8192（仍是 cap）。累计自然结束率：≤4096 3.5%，≤5120 10.5%，≤6144 21.5%，≤7168 29.5%，≤8192 43.0%。
- 2026-09-19 (8,4,5) direct 重测（D6，--direct-only，32 token）：direct 0.0045（2000 题，严格 = 宽松，格式错 0；stop500 0.002；阈值 0.0625 → 第 1 条通过）；逐对 0.612 vs 基线 0.666。准入 json 已写回，判定仍为未通过（第 2、4、7 条）。
- 2026-09-19 D8–D11 已记（cap 10240；guess_verify 定义 C；第 4 条只比 native > chance+10pp 的点；sleep mode 透传）。pod 流水线 09:56–15:43 UTC，HEAD c40af4f（T1/T2 用 4f43f03）。
- 2026-09-19 吞吐测量（不是结果）：A --dry-run 20 步，(8,4,5)，cap 10240，临时 M=8192，vLLM 0.45，micro-batch 1 × GA 32（32 条 / 步）。去掉第 1 步后 19 步均值：494 s / 步（gen 218 s + train 276 s），L_mean 8106，约 25.9 万 token / 步；峰值 allocated 74.43 GiB / reserved 78.41 GiB（整进程，含 vLLM 常驻约 36 GiB）。vLLM KV 27.37 GiB = 199,296 token，10752 token / 请求下最大并发 18.5。投影（$1.6/h）：200 步 = 27.4 h / run；× 11 run = 302 GPU·h ≈ $483（纯训练）；eval 估 1.0 h / 次（stop 500 原生 ≈ 405 万 token ÷ 约 1190 tok/s + direct 2000）× 5 次 / run × 11 = 55 h ≈ $88；合计约 357 GPU·h ≈ **$571**（> cost.budget_usd 300）。cost.py 按矩阵原样（400 步 × 16 run）给 890 GPU·h / $1424。
- 2026-09-19 准入 ord_n8_h4_d5 @ cap 10240（冻结 Qwen3-4B，stop 500，direct-check 2000）：**未通过，只差第 7 条**。1 direct 0.0045（严格 = 宽松，格式错 0，stop500 0.002；阈值 0.0625）过；逐对 0.612 vs 基线 0.666；2 native 0.608（严格 = 宽松）过；M = 8599.5，强制收尾 37.8%，自然结束 311 条准确率 0.891（长度 min/median/max 3423/6863/10187），被截断 189 条准确率 0.143；全体 L p10/p25/p50/p75/p90 = 5115/6366/8588/10240/10240；3 预算曲线 0.004/0.026/0.226/0.440/0.550，未到 native−5pp=0.558 → ">1.0" 过；4 填充 0.004/0/0.002/0.002/0.004，比较点 0.5/0.75/1.0 均过；5 移植（derangement）0.000，掉 0.608 ≥ 0.302 过；6 letter-ban 0.006 过；7 捷径命中 **0.258** 不过（perm_enum 0.148、guess_verify 定义 C 0.072、ir_copy_order 0.042、assign_enum 0.020；原 r4 定义 any 0.922、guess_verify 0.906）；8 M / (5×Kahn 30) = 286.7× 过（决策下限 32.5 → 264.6×）。perm_enum 74 条里 73 条是"≥8 个不同的完整顺序"触发（1 条短语、0 条 8! / 40320），62 条被截断、13 条答对。第 2 条通过 → cap 12288 重测按规则跳过。策略类（纯描述）100% mixed_probe_enum。
- 2026-09-19 吞吐调优（D11，5 步 dry-run，cap 10240，vLLM 0.6）：T1（仅 0.6）KV 39.26 GiB = 285,872 token、并发 26.6，第 1 步训练阶段 OOM（要 5.79 GiB，剩 3.83 GiB）。T2（0.6 + sleep mode）5 步跑完：步 2–5 均值 476 s（gen 208 s + train 268 s），L_mean 7786；步 1–5 的 gen/train = 236/266、178/272、229/264、194/272、232/265；torch 报的峰值 allocated 86.31 / reserved 93.22 GiB 超过物理显存——sleep mode 的 cumem 分配器把已让出的 vLLM 池也计入，不能当真实占用；进程退出时 abort（rc=134，5 步已完成、报告已写）。与基线比：每步约快 4%（同期 L_mean 也短 4%），基本无收益。

## 2026-09-19 D12 重算第 7 条：(8,4,5) cap 10240（已存 500 条原生轨迹，不重跑生成；本地计算）
- 定义：perm_enum 的"≥8 个不同完整顺序"只计入无推理切换（相邻两个不同完整顺序之间 0 个传播 / 分情况标记）里的顺序；关键词各支不变。
- D12（判定用）：any **0.132**（66/500）｜ir_copy_order 0.042、perm_enum **0.006**、assign_enum 0.020、guess_verify 0.072
- 仅 D9（D12 之前）：any 0.258｜perm_enum 0.148，其余同上
- r4 原定义：any 0.922｜perm_enum 0.148、guess_verify 0.906，其余同上
- 命中 66 条的构成：只 guess_verify 32、只 ir_copy_order 20、只 assign_enum 8、只 perm_enum 2、assign_enum+guess_verify 2、ir_copy_order+guess_verify 1、perm_enum+guess_verify 1
- 命中 × 结局：自然结束且答对 31、自然结束答错 6、强制收尾答对 8、强制收尾答错 21
- **第 7 条仍不过（13.2% ≥ 5%）→ 准入未通过；未写正式 admission json；未开机、未启动 A1。** 其余 7 条不变（M = 8599.5）。

## 2026-09-19 第二次迁移 + D13 + 正式准入
- pod 迁移：新 pod `2dy8hd8l24yo24`（letter-tax-migration，A100 SXM 80GB，195.26.233.65:20967）；/workspace 完整（cot-compress 含 .venv 15G、hf 7.6G、.uv-cache、.runpod、bin/runpodctl、logs），无需重建；~/.runpod 软链重建；`runpodctl get pod` 可见新 pod RUNNING、旧 pod jhlv0ja6xynmy3 EXITED；pod 上 HEAD 是 4f43f03（D12 / D13 是迁移后才推上去的）。
- 审计（66 条被 D12 标中的轨迹全读）：模板 / 抄答案 0、正常 50、瞎搭 16；未标中的 434 条未读。audit_template_rate = 0.000。
- 检测器命中率（描述量，500 条）：r4 any 0.922（ir 0.042 / perm 0.148 / assign 0.020 / gv 0.906）｜D9 0.258（gv 0.072）｜D12 0.132（perm 0.006）｜D13 **0.060**（ir 0.000 / perm 0.006 / assign 0.020 / gv 0.040）
- **(8,4,5) cap 10240 准入通过（八条全过）**，正式 `results/admission_ord_n8_h4_d5.json`，**M = 8599.5**。

## 2026-09-19 A1 正式 run（阶段 1 链，D14）—— 第 1 个训练步崩溃，链按 --strict 停机
- 启动 22:22 UTC：`scripts/chain_stage1.sh`（setsid nohup），A_0.5_ord_n8_h4_d5_s1，λ=0.5，M=8599.5，cap 10240，G=16，max_steps 400，HEAD 37d49d2。
- step-0 eval（stop 500，55.5 min）：acc **0.624**（clean 0.624，宽松 0.624，viol 0），L_mean **8219.0**，L_median **8586.5**，capped / forced 0.394，acc@1024 0.008，fmt_err 0，letter_frac 0.787，mask_frac 0.407，ref_ppl 2.75，hit_hamming2 0.634；direct（2000 题）**0.004**，frozen_direct 0.0045，残留阈值 0.314。与准入（native 0.608、M 8599.5）一致。
- eval 后显存：reserved 43.47 GiB、alloc 42.70 GiB。
- **第 1 个训练步的 backward 报 `torch.AcceleratorError: CUDA error: invalid argument`**（23:25 UTC，train_rc=1；不是 OOM）。没有写出任何 steps.jsonl 行，没有 checkpoint。`run_matrix --strict` → `results/matrix_halt.json`（"A_0.5_ord_n8_h4_d5_s1: train_rc=1 diag_rc=None"）→ `pod_stop.sh --delay 300`。未重试、未改任何设置。
- 环境差异：迁移后的新宿主机 driver **595.91.07**（上一台 580.159.04，那台上同一代码 / 同一 venv 的 20 步 dry-run 与 T2 都跑通）。代码差异：正式 run 先做 step-0 eval 再训练（dry-run 无 eval）、M 来自准入（dry-run 是 m_override 8192）。`None of the inputs have requires_grad=True` 警告在跑通的 dry-run 日志里同样出现，不是原因。根因未定位（需要 CUDA_LAUNCH_BLOCKING=1 复现）。

## 2026-09-19 23:50 UTC 接管：planning-led 成为唯一会话（D17）
- 用户指示接管。execution 的链已结束（A1 崩停）；它的最后一个任务 = 排查第 1 步（3 步 dry-run、不带 eval），报完即退休。在该结果转来之前，planning-led 不对 pod 做写操作、不 push。接管记录与常设规则见 `ops/state.md`。
- 提交重叠期积压的文档改动：`docs/paper.md`（七处与现行协议对齐，逐处行内标注 deviation 并保留预注册原文；§6.1 与 §6.2 成本部分按本日志与准入 json 填入；§9 决策史）、`ops/`（state.md、to_execution.md、只读看板 dashboard.py）、`docs/roles.md`。不涉及代码与判定。
- **更正（本日志上文的笔误）**：2026-09-19 准入 (8,4,5)@cap 10240 一条里写的 "M / (5×Kahn 30) = 286.7×" 应为 **M / Kahn = 8599.5 / 30 = 286.7×**（M / (5×Kahn) = 57.3×）；"决策下限 32.5 → 264.6×" 同为 M / 决策下限。第 8 条判据是 M ≥ 5×Kahn，结论（通过）不变。原文保留不改，以本条为准。
- **费率（D18）**：`configs/cloud_4b.yaml` `cost.usd_per_hour` 1.9 → 1.6。pod 实际单价 $1.59/h；用户定 stage-1 的 $200 线与 $300 上限的投影都按 $1.6/h 判（本日志的 $571 投影原本就按 1.6 算）。`budget_usd: 300` 未动。`run_matrix.py` / `cost.py` 里配置缺省时的回退值 1.9 未改（配置里有值，走不到）。
- **A1 重跑的锚点（用户 2026-09-19 定，先于任何重跑数字）**：崩溃那次的 run 目录另存备查；重跑用干净目录；操纵门以重跑自己的 step-0 eval 为锚点；两次 step-0 的数都报（崩溃那次：acc 0.624 / L_mean 8219 / L_median 8586.5 / 强制收尾 39.4%）。
- **修复方式的边界（用户定）**：vLLM sleep mode / colocate 下 eval↔训练切换、驱动 / 库版本一类 → 工程修复，记 log；“训练前不跑 step-0 eval”“改 eval 的 n”或改 eval 划分 / 温度 / 频率 → 动到操纵门锚点与停止判据的数据来源，先停下问用户。

## 2026-09-19/20 排查第 1 步：3 步 dry-run、无 eval、CUDA_LAUNCH_BLOCKING=1（pod 2dy8hd8l24yo24，重启后 SSH 端口 39125，driver 595.91.07）
- 命令：`train.py --arm A --dry-run --tag _dbg1_noeval --set reward.lambda=0.5 task.key=ord_n8_h4_d5 train.seed=1 train.dry_run_steps=3`，HEAD f46e47a；日志 `/workspace/logs/dbg1_noeval.log`。
- **失败（rc=1），与 A1 同一处**：第 1 个训练步 `loss.backward()` → `torch.AcceleratorError: CUDA error: invalid argument`（同步模式下栈仍落在 autograd 引擎 `_engine_run_backward`，没有更具体的 kernel 帧）；生成阶段正常（after generate: reserved 43.47 GiB、alloc 42.70 GiB）；无 steps.jsonl。
- 结论：不带 eval 也崩 → 不是"eval 之后接训练"的问题；按排查顺序判为**宿主机 / 驱动问题**（同一代码与 venv 在上一台 driver 580.159.04 的机器上跑通过 20 步）。下一步（未做）：控制台 migrate 换机器后重跑本命令确认；第 2 步（`train.dry_run_eval_n=32`，D16）未跑。
- **交接（2026-09-20）**：execution 会话到此收尾、不再接任务；pod 仍在运行（未停机）、其上无任务在跑；后续全部操作由 planning 会话接管。本地后台等待器已全部停止。

## 2026-09-20 A1 backward 崩溃排查（planning-led 接手；pod 2dy8hd8l24yo24，driver 595.91.07，port 39125）
- 00:18 UTC pod 空闲（0 MiB），pod 上 HEAD f46e47a。execution 的第 1 步（3 步 dry-run、不带 eval、CUDA_LAUNCH_BLOCKING=1、seed 1）失败于同一处（`logs/dbg1_noeval.log`，0/3 步，05:55 处崩）。
- **宿主机 backward 阶梯检查**（新增 `ops/host_check.py`：不走 TRL、不 import 项目代码；`results/host_check_595.json`，日志 `logs/host_check_595.log`；CUDA_LAUNCH_BLOCKING=1）：**11 级全部 PASS**——R1 bf16 matmul；R2 SDPA 因果 backward seq 2048 / 10752；R3 non-reentrant checkpoint(SDPA)；R4 Qwen3-4B bf16 + LoRA(r32, attn+MLP) + 梯度检查点 backward seq 2048 / 10752（峰值 16.5 GiB）；R5 同上但带左填充 attention_mask（走带 mask 的 SDPA kernel）；R6 同进程起 colocate 方式的 vLLM（external_launcher，util 0.45，max_model_len 11264）生成后，再做 seq 2048 与带 mask 的 seq 10752 backward（峰值 50.8 GiB）。
  → "这台宿主机 / 驱动做不了 backward"不成立；崩溃需要训练栈里阶梯没覆盖的东西（TRL GRPO 的 loss 路径 / 全词表 fp32 logits [T≈10.3k, V=151936] 的 logsumexp+gather backward / 真实 rollout 的形状）。
- **发现的混淆变量**：旧宿主机上跑通的 20 步 dry-run 命令（`logs/pipeline3.sh`）是 `--set task.key=… reward.m_override=8192 cost.usd_per_hour=1.6`，**train.seed 取配置默认 0**；两次崩溃（A1 正式 run、dbg1）都是 **train.seed=1**、M=8599.5（准入值）。即"旧机器通过 / 新机器崩"同时混着 seed 0 / seed 1。
- 00:23 UTC 启动对照 `logs/dbg2.sh`：在**这台**机器上原样重跑旧命令（seed 0、m_override 8192），3 步，tag `_dbg2_seed0_oldcmd`。通过 → 宿主机被排除、触发条件在 seed 1 的首个 batch；崩 → 宿主机 × 训练栈的交互。
- 00:39 UTC **dbg2 结果：seed 0 在这台机器（595.91.07）上第 1 步跑通**：step 1 = 622.5 s（gen 324.8 + train 297.7；CUDA_LAUNCH_BLOCKING=1 所以偏慢），reward_acc 0.50，L_mean 8845，自然结束 53%，峰值 alloc 73.93 / reserved 78.48 GiB。→ **宿主机 / 驱动被排除**；迁移不解决问题。第 1 步通过后按 PID 停掉该 run（不等第 2–3 步，省 GPU 时间）。操作失误记录：第一次用 `pkill -f "logs/dbg2.sh"`，模式匹配到自己的 ssh shell，把会话杀了、训练进程没杀到；改为只按 PID kill（本日志 09-18 已写过这条教训）。
- 三次 run 的首个 batch 对比（rollouts 归档）：A1（seed 1，prompt 1355 / 609，16/32 到顶）崩；dbg1（seed 1，prompt 1138 / 634，8/32 到顶）崩；dbg2（seed 0，prompt 476 / 784，15/32 到顶）通过。三者 padding 后的最大长度相同（10219 token），token id 上限相同 → **不是张量形状、也不是某一个特定 batch**；两次 seed-1 的 batch 不同却都崩。dbg1 与 dbg2 之间剩下的差异：`train.seed` 1 vs 0、M 8599.5 vs m_override 8192（只进奖励）。
- 00:42 UTC 启动 dbg3（`logs/dbg3.sh`）：dbg1 的原样复现（seed 1、准入 M、1 步），经 `ops/train_anomaly.py` 包装打开 autograd anomaly 模式（不改训练代码），目的 = 拿到出错 backward 算子对应的 forward 调用栈。
- 01:02 UTC **dbg3 结果：seed 1 + 准入 M + anomaly 模式，第 1 步跑通**（rc=0；1119 s = gen 306 + train 813，anomaly 模式拖慢 backward；reward_acc 0.81，L_mean 7738，峰值 alloc 73.93 / reserved 78.48 GiB）。→ **崩溃不是 seed 决定的，是间歇性的**。上一条"宿主机 / 驱动被排除"说过头了，更正：这台机器能做 backward（阶梯 11 级全过），但在**这台机器上首个训练步 4 次里崩 2 次**（A1 崩、dbg1 崩、dbg2 过、dbg3 过），旧机器（580.159.04）上同一训练栈 **30 步 0 次**（20 + 5 + 5 步）。两者差异不像偶然（Fisher 精确检验单侧 p ≈ 0.01），但 4 次的样本小，且 dbg3 多了 anomaly 模式这个变量。共同点：所有 run 的训练侧峰值 reserved 78.4–78.5 GiB / 79.25 GiB，贴着显存上限——"显存吃紧时分配器回收 / 重试路径 × 新驱动"是目前的首要嫌疑，未证实。
- 01:03 UTC 启动 dbg4（`logs/dbg4.sh`）：seed 1、准入 M、**6 步**、`CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1`（不带 anomaly）。目的：① 拿到 C++ 调用栈，看出错的是哪个 CUDA API（kernel / cuBLAS / 分配器）；② 多几步，估计这台机器上的每步崩溃率。上限约 1 小时 ≈ $1.6。
- 02:08 UTC **dbg4 结果：6 步全过**（rc=0；每步 602–686 s，同步模式；reward_acc 0.81 / 0.66 / 0.50 / 0.63 / 0.56 / 0.34；L_mean 7738–9315）。没有崩 → 没拿到 C++ 栈。dbg3 与 dbg4 的首个 batch 完全相同（prompt 1138 / 634，sumL 247,604）；dbg1（同一条命令）的 prompt 相同但采样不同（sumL 250,353）→ vLLM 采样在不同进程间不是逐位可复现的。
- **这台机器（595.91.07）到此的账**：进程首个训练步 5 次里崩 2 次（A1、dbg1 崩；dbg2、dbg3、dbg4 过）；非首步 5 步 0 崩。旧机器（580.159.04）30 步 0 崩（其中首步 3 次）。根因未定位；训练侧峰值 reserved 78.4–78.5 / 79.25 GiB 是唯一有实测支持的嫌疑。
- **用户预先授权（2026-09-20）**：`train.vllm_gpu_memory_utilization` 0.45 → 0.35~0.40 属工程修复，直接做、记 log；批大小 2×16、cap、reward、LoRA、eval 划分不动。验收 = 这台机器上连续 10 步不崩，且带一次小 eval（D16 开关）确认 eval→训练切换；降 util 仍崩或栈指不出原因 → 迁移，新机器用 host_check + 10 步（带小 eval）验收。
- 02:09 UTC 启动验收 acc1（`logs/acc1.sh`）：seed 1、准入 M、`train.vllm_gpu_memory_utilization=0.40`（命令行覆盖，配置文件暂不改，过了再改）、`train.dry_run_steps=10`、`train.dry_run_eval_n=32`、生产条件（不开 CUDA_LAUNCH_BLOCKING；开 TORCH_SHOW_CPP_STACKTRACES 以便万一崩了有栈）。选 0.40 而不是 0.35：训练侧多出约 4 GiB 余量，vLLM KV 少约 4 GiB（并发约 18 → 15 条），生成变慢的代价较小；实测步时见结果。
- **验收方案（用户 2026-09-20 定，取代上面"单次连续 10 步"的执行方式）**：acc1（10 步 + 小 eval）之后再做 3 次"只跑 1 步"的独立启动，专测进程首步；加上 acc1 的首步共 4 个首步。判据：4 个首步全过 → 认为降 util 有效，util 0.40 写进配置并记 deviation，直接重跑 A1（首步再崩 → 视为无效，转迁移）；任何一次崩 → 降 util 无效，转迁移（新机器：host_check + 同样的验收）。依据：不降 util 时首步 5 崩 2，"无效"假设下 4 次全过的概率约 13%。pod 上 `logs/acc2.sh` 已排在 acc1 之后自动执行（acc1 失败则跳过；三次里任何一次非零退出即停）。acc1 实测：util 0.40 → KV 170,448 token、并发 15.85 条（0.45 时 199,296 / 18.5）。
- **watcher + 看板（D19，本地，不占 GPU）**：`setup/auto_pull.sh` 增加每 600 s 固定拉一次曲线文件（`runs/*/steps.jsonl`、`eval.jsonl`、`designated_ckpt.json`、`uninterpretable.json`、`results/matrix_log.md`、`matrix_halt.json`、`CHAIN_DONE`、`results/*.json`；不含 `*.zst`、`reward_log.jsonl`、checkpoint），原有"eval / matrix_log / 准入 json 变化 → 完整 pull"不变。02:18 UTC 起本地 watcher 与 `ops/dashboard.py --watch 120` 以 setsid nohup 常驻（日志 `cloud_pull/auto_pull.log`、`cloud_pull/dashboard.log`）。看板加第 5 块：链以外的排查 / 验收 run 的逐步表。新增排查工具 `ops/host_check.py`、`ops/train_anomaly.py`（均不改训练代码）。
- 03:41 UTC **acc1 通过**（rc=0）：训练前小 eval（n=32：acc 0.5625，L_mean 8678，4.6 min）→ 10 个训练步全过。util 0.40、生产条件。每步 455–552 s，均值 **503 s**（gen 237 + train 266；util 0.45 的 dry-run 是 494 s = 218 + 276）→ 降 util 的吞吐代价约 2%。峰值 alloc 70.0–70.5 GiB、**reserved 77.2–77.3 GiB**（0.45 时 78.4–78.5）→ 离 79.25 的余量从约 0.8 GiB 变成约 2.0 GiB。KV 23.41 GiB = 170,448 token，并发 15.85。第 1 个首步：过。acc2（3 次单步启动）已自动接上。
- 04:13 UTC **acc2：3 次单步启动全过**（rc=0 ×3；每次 477 s；reserved 76.82 GiB）。按用户定的判据，4 个首步全过。**但这 4 个首步不构成对"降 util 有效"的检验，如实记录：**
  1. 三次 acc2 的首个 batch 逐位相同（reward_acc 0.8125、L_mean 7737.625），且与 dbg3、dbg4（util 0.45）的首个 batch 相同——同一 seed、不带 eval 时采样是确定的。所以它们不是 3 个独立的首步样本。
  2. 同一条命令的 dbg1 采样却不同（sumL 250,353 vs 247,604）。查日志找到了原因，也找到了把全部 9 次启动完全分开的变量——**vLLM 的 torch.compile 缓存是冷还是热**（缓存在容器盘 `/root/.cache/vllm/torch_compile_cache`，pod 每次 stop 都会清空）：
     | run | util | 编译缓存 | 结果 |
     |---|---|---|---|
     | A1 正式（开机后第一个进程） | 0.45 | 冷（"Compiling a graph … 32.6 s"） | **崩** |
     | dbg1（再次开机后第一个进程） | 0.45 | 冷（33.4 s） | **崩** |
     | dbg2 / dbg3 / dbg4 | 0.45 | 热（"Directly load the compiled graph(s) … from the cache"） | 过 ×3 |
     | acc1 / acc2 ×3 | 0.40 | 热 | 过 ×4 |
     冷 2/2 崩，热 7/7 过；util 0.45 在热缓存下 3/3 过 → **util 不是解释变量，冷编译才是**。（旧宿主机上通过的 20 步 dry-run 日志里是 "Compiling a graph … 9.01 s"，属于部分冷编译且通过，所以不排除"冷编译 × 这台机器"的交互；T1 / T2 是热缓存。）
  3. 含义：现在直接重跑 A1 是热缓存，预期能过——但这和 util 无关；而链每次 `pod_stop` 之后再开机，第一个进程又是冷缓存，会再崩。
- 04:14 UTC 启动冷缓存对照 cold1（`logs/cold1.sh`）：把 `torch_compile_cache` 改名移开（`…/torch_compile_cache.warm_0920`，不删），util 0.40、seed 1、1 步。崩 → 冷编译是触发条件，且降 util 不治本；过 → 冷编译假说不成立（或 util 0.40 确实有帮助）。
- 04:25 UTC **cold1 结果：过**（rc=0，1 步 478 s，batch 与热缓存的 run 逐位相同）。**但 cold1 不够冷**：日志是 "Compiling a graph … 9.15 s"，而两次崩溃是 32.6 / 33.4 s——cold1 只移开了 vLLM 的 `torch_compile_cache`，inductor（`/tmp/torchinductor_root`）与 triton（`~/.triton/cache`）的缓存仍是热的（三处缓存的创建时间都是 09-19 23:46–23:48 = dbg1 启动时，即那次开机后的全冷编译）。旧宿主机上通过的那次 dry-run 同样是 9 s 的"半冷"。所以 cold1 只说明"半冷不崩"，没有检验"全冷"。
- **用户批示（2026-09-20）**：① 修复方向 = 编译缓存持久化到 /workspace + 开机后预热，两者都做，工程修复记 log；② 验收 = 冷缓存下先复现崩溃，上修复后从冷启动连过 3 次独立启动、其中一次带小 eval；冷缓存下不崩 → 冷编译假说不成立，回到原判据直接重跑 A1，持久化缓存照做；③ util 保留 0.40，记 deviation。看板：默认只显示在跑的与最近一条、历史折叠、崩溃标红（已做：状态读 pod 上 `logs/*.rc` / `*.start`，watcher 每 10 分钟一并拉回）。
- 04:26 UTC 启动 cold2（`logs/cold2.sh`）：**全冷**——vLLM `torch_compile_cache`、`/tmp/torchinductor_root`、`~/.triton/cache` 三处全部改名移开（`*.warm_0920` / `*.cold1_0920`，不删），util 0.40、seed 1、1 步、TORCH_SHOW_CPP_STACKTRACES=1。这才是对"冷编译触发崩溃"的检验，也是用户判据里"冷缓存下先复现崩溃"那一步。
- 04:38 UTC **cold2 结果：全冷 + util 0.40 → 过**（rc=0；编译 32.57 s = 与两次崩溃同等的全冷；1 步 511 s；reward_acc 0.75、L_mean 8619——采样与热缓存的 run 不同，印证"新编译出的 kernel 会改变采样的数值"，也解释了 dbg1 为何与 dbg3 / dbg4 采样不同）。reserved 76.67 GiB。
  到此的 2×2 表（这台机器，进程首个训练步）：
  | | util 0.45 | util 0.40 |
  |---|---|---|
  | 全冷编译 | **崩 2/2**（A1、dbg1） | 过 1/1（cold2） |
  | 热 / 半冷缓存 | 过 3/3（dbg2–4） | 过 6/6（acc1、acc2×3、cold1；均含热或半冷） |
  与两个嫌疑都相容的读法：**全冷编译（多占一块显存 / 编译期的分配）× 显存余量只有约 0.8 GiB → 崩**；任一条件解除都不崩。全冷 × 0.40 只有 1 次，证据弱。
- 04:38 UTC 启动 cold3（`logs/cold3.sh`）：全冷（三处缓存再次移开）+ **util 0.45**（配置原值）、seed 1、1 步、带 C++ 栈。= 用户判据里的"冷缓存下先复现崩溃（确认触发条件）"。崩 → 触发条件确认（全冷 × 0.45 = 3/3），并应能拿到栈；过 → 崩溃不可由"全冷 × 0.45"稳定复现，如实报告。
- 04:46 UTC **cold3 结果：全冷 + util 0.45 → 崩**（rc=1；编译 32.33 s；第 1 个训练步 backward，`CUDA error: invalid argument`）。**触发条件确认：全冷编译 × util 0.45 = 3/3 崩**（A1、dbg1、cold3）；全冷 × 0.40 过（cold2）；热 / 半冷 × 0.45 过 3/3。
- **拿到 C++ 栈（根因定位到分配器的显存回收路径）**：`LogsumexpBackward0` → `mul` → 申请输出张量 → `CUDACachingAllocator::malloc` → **`DeviceCachingAllocator::release_block`** → `c10_cuda_check_implementation`（CUDA invalid argument）。即：backward 里全词表 fp32 logits [T≈10.3k × V=151936]（约 6 GiB）的 logsumexp 反传要新分配一块大张量时，显存不够，缓存分配器走"释放已缓存块再重试"的路径，其中某个块的 `cudaFree` 返回 invalid argument。只有显存顶到上限时才会走到这条路径：util 0.45 下余量约 0.8 GiB，全冷编译（inductor / triton 现场编译）再多占一点就触发；热缓存或 util 0.40（余量约 2 GiB）时 malloc 直接成功，走不到 release_block。为什么 cudaFree 会对那个块报 invalid argument（与 colocate 的 vLLM 内存池 / CUDA graph 私有池 / 新驱动的关系）没有查；修复不依赖它。
  用户最初的显存假说成立。planning-led 途中两个说过头的结论（"宿主机排除""util 不是解释变量"）均已在上文更正。
- **修复（D20，用户批准）**：① `configs/cloud_4b.yaml` `train.vllm_gpu_memory_utilization` 0.45 → **0.40**；② `setup/env_cloud.sh`：`VLLM_CACHE_ROOT` / `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` 指到 `/workspace/.cache/`（持久盘；容器盘每次 stop 清空）；③ `scripts/chain_stage1.sh`：run_matrix 之前先跑一个可丢弃的 1 步 dry-run 预热（退出码不计，tag `_warmup`，不是实验数据）。批大小 2×16、cap、reward、LoRA、eval 划分未动。04:47 UTC 已 push 到 pod。
- 04:47 UTC 启动验收 acc3（`logs/acc3.sh`）：修复就位（util 取自配置，不再命令行覆盖），**3 次独立启动、每次启动前把全部编译缓存（持久盘新位置 + 容器盘旧位置）改名移开 = 每次都全冷（最坏情况）**，第 1 次带小 eval（n=32），各 1 步；任何一次非零退出即停。判据（用户）：3 次全过 → 重跑 A1；任何一次崩 → 停下报告。
