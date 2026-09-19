# 会话分工（2026-09-19 起）

本仓库同时有两个 Claude Code 会话，职责不重叠：

| 会话 | 做什么 | 不做什么 |
|---|---|---|
| **execution** | 跑 GPU 任务、改代码与配置、管 pod（同步、启动、监控、`scripts/pod_stop.sh`）、写 `results/cloud_log.md` 与 `docs/deviations_log.md`、提交与推送 | **不改 `docs/paper.md`**（可以读、可以随其它改动一起提交，但不动内容） |
| **planning** | 只读仓库；写 `docs/paper.md` 与分析 | **不碰代码、配置和 pod**（不改 `cot_compress/`、`scripts/`、`tasks/`、`configs/`、`setup/`、`tests/`，不 ssh、不启动或停止任何任务） |

规则：
- 任何代码 / 配置改动仍然只由 execution 做，并按既有规则记入 `docs/deviations_log.md`（带 commit hash）。
- planning 需要新的数字或分析脚本时，写明需求，由用户转给 execution；execution 产出的数字以 `results/` 与 `results/cloud_log.md` 为准。
- 两个会话的意见或改动有冲突时，**以用户为准**。

## 2026-09-19 23:50 UTC 起：单会话（D17）
execution 会话在 stage-1 链结束（A1 首个训练步崩溃、停机）并报完最后一个排查任务后退休。此后只有一个会话（planning-led），上表两行的职责合并由它承担：读结果与写 `docs/paper.md`，同时负责 ssh、跑脚本、管 pod、改代码、写 `results/cloud_log.md` 与 `docs/deviations_log.md`、提交。会话状态与接管记录见 `ops/state.md`；给 execution 的历史指令见 `ops/to_execution.md`（存档）。
用户定的规则不变：规则和数据打架时停下问用户；不扩 scope；协议冻结，改动走 deviation；每次操作记 cloud_log，每次决策记 paper §9；预算上限 $300 写死，超了停下问用户。
