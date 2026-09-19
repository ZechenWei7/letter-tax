# 给 execution 会话的指令（由用户转发；planning-led 写）

## 2026-09-19 #1 — 链的存活状态 + watcher 拉取频率

背景：本地 `cloud_pull/runs/` 里只有 dry-run / smoke，没有 A1 的任何文件；watcher 只在 `eval.jsonl` / `matrix_log.md` / 准入 json 变化时 pull，而 eval 每 50 步（≈7 h）才有一次，所以用户现在无法判断链是否还活着。

**第 1 步（立刻做，先于其它）：报一次 A1 的当前状态。** 只读操作，直接回复给用户：
- A1 的 run 目录名；当前跑到第几步（`steps.jsonl` 行数 / 最后一行的 step）。
- 最近一步的时间戳（`steps.jsonl` 的 mtime，UTC）与 pod 当前 UTC 时间；据此说明链是否在正常前进（正常约 8 min 一步）。
- `steps.jsonl` 最后 5 行原文；`eval.jsonl` 已有的全部行（若有）。
- `results/matrix_log.md` 末尾 20 行；是否存在 `results/matrix_halt.json` / `results/CHAIN_DONE` / 任何 `uninterpretable.json`。
- `nvidia-smi` 一行摘要（GPU 利用率、显存），以及训练进程 / chain_stage1.sh 是否还在。
- 到目前为止的 pod 运行时长与估算花费。
如果发现链已经停了或出错：**不要重启、不要重调、不要改任何东西**，把事实报给用户。

**第 2 步：改本地 watcher（`setup/auto_pull.sh`），让每步的曲线文件也流回本地。** 二选一，取实现更简单的：
- (a) 变化签名里加入 `runs/*/steps.jsonl`（例如把各 run 的 `steps.jsonl` 行数或 mtime 并入 md5 的输入），或
- (b) 不管有无变化，每 10 分钟固定 pull 一次。
要求：
- 高频这一路**只拉小的曲线文件**：`runs/*/steps.jsonl`、`runs/*/eval.jsonl`、`runs/*/designated_ckpt.json`、`runs/*/uninterpretable.json`、`results/matrix_log.md`、`results/matrix_halt.json`、`results/CHAIN_DONE`、`results/*.json`。**排除所有 `*.zst`**（`rollouts.jsonl.zst`、`eval_step*.jsonl.zst`、`diag_*`）和 `reward_log.jsonl`；大归档仍走原来"eval 变化时才 pull"的那一路，行为不变。
- 落点不变（`cloud_pull/`）。
- 只是本地发起的 ssh 读 + rsync：**不改训练代码、不改 pod 上任何正在跑的东西、不重启链。**
- 改完重启本地 watcher，确认 `cloud_pull/runs/<A1 run>/steps.jsonl` 出现且在增长，回报一句。

**留痕：** `results/cloud_log.md` 记一行（做了什么、为什么）；`docs/deviations_log.md` 记一条——基础设施，对预注册无影响（拉取频率不是预注册页写死的内容，不改任何规则 / 判定 / 训练 / 评估）。

**不要做的：** 不改 eval 频率、不改 `train.py` / `run_matrix.py` / 配置、不动 `docs/paper.md` 与 `ops/`。planning-led 会在 `ops/` 下放一个只读 `cloud_pull/` 的本地看板脚本，与你的文件不重叠。

## 2026-09-19 #2 — 费率改为 $1.6/h（**排查结束后再转发**；用户指示现在不打断排查）
- `configs/cloud_4b.yaml`：`cost.usd_per_hour: 1.9` → `1.6`。原因：pod 实际单价 $1.59/h；用户决定 stage-1 的 $200 线与 $300 上限的投影都按 $1.6/h 判；cloud_log 里的 $571 投影本来就是按 1.6 算的，配置里的 1.9 是旧值。`cost.budget_usd: 300` 不动。
- 留痕：`results/cloud_log.md` 一行；`docs/deviations_log.md` 一条——影响 `run_matrix.budget_projection`（B_warm-RL 前的预算检查）与 `--dry-run` 投影的美元数；不改任何规则 / 阈值 / 训练 / 评估；预注册写死的是 $300 上限，不是费率。
- 若还有别的配置文件带同一字段（`cloud_8b.yaml` 等），一并列出，改不改由用户定。

## 2026-09-19 #3 — A1 崩溃排查的回报格式（planning-led 需要的信息）
排查结束后请回报：完整 traceback 与崩溃时的显存数字；3 步无 eval dry-run 与"加小 eval"两次的结果；根因判断；拟用的修复**逐项列出改了什么**（planning-led 要对照预注册页判断是否构成 deviation）。**修复确定之前不要重启 A1**；**重跑时 A1 的 run 目录（用户已定）**：崩溃那次的目录改名另存（如 `runs/A_0.5_ord_n8_h4_d5_s1__crash0919`，不删，曲线文件照常 pull 回本地），重跑用干净目录，操纵门以重跑自己的 step-0 eval 为锚点，两次 step-0 的数都报。原因：`eval.jsonl` 是追加写，`first_run_gate` 取第一条 step-0 记录，两条并存会让锚点含糊。
**修复方式的边界（用户已定）**：vLLM sleep mode / colocate 下 eval↔训练切换一类的工程修复 → 记 log 即可；但**不得**用"训练前不跑 step-0 eval"、"改 eval 的 n"或改 eval 划分 / 温度 / 频率的方式绕开崩溃——那会动到操纵门锚点与停止判据的数据来源，若排查指向只能这样修，停下报给用户。

---
**2026-09-19 23:50 UTC 存档说明**：planning-led 已接管，execution 退休。#1 第 2 步（watcher 纳入 steps.jsonl）若 execution 未做，由 planning-led 做；#2（费率 1.9 → 1.6）已由 planning-led 执行（D18）；#3 的锚点与修复边界继续有效，执行者改为 planning-led。
