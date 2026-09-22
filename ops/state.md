# ops/state.md — planning-led 会话状态

## 阶段
- **重叠期**（2026-09-19 起）：execution 会话持有 pod 与后台 watcher，正在跑 stage-1 链 A1 → B_warm-SFT(A1) → B1 → warm 判定（D14，`--strict`）。planning-led 不碰 pod、不 ssh、不改 `cot_compress/ tasks/ scripts/ configs/`，只读本地已 pull 的结果；给 execution 的指令写 `ops/to_execution.md`，由用户转发。
- **接管时间：2026-09-19 23:50 UTC**（用户指示“接管”；execution 的链已结束——A1 首个训练步崩溃、`--strict` 停机）。此后 ssh、跑脚本、管 pod、改代码、提交都由 planning-led 做；操作记 `results/cloud_log.md`，决策记 `docs/paper.md` §9。上面“重叠期”一条自此失效。
- **交接尾巴**：execution 的最后一个任务是排查第 1 步（3 步 dry-run、不带 eval），报完结果即退休；结果由用户转来。**在那之前 planning-led 不对 pod 做任何写操作、不 push 代码到 pod**（避免干扰它正在跑的 dry-run）。排查第 2 步（dry-run + 小 eval，`train.dry_run_eval_n`，D16）由 planning-led 接着做。
- pod：`ssh root@195.26.233.65 -p 39125 -i ~/.ssh/id_ed25519`；runpodctl 与 API key 只在 pod 的 /workspace 上 → **开机要用户在控制台点**，停机用 pod 上的 `scripts/pod_stop.sh`。

## 常设规则（用户，2026-09-19）
- 改任何判定标准前先对照预注册页 https://osf.io/usycb（freeze r4 `d207c263`）：页面写死的 → 只能以 deviation 形式改，记 `docs/deviations_log.md` 并在论文交代；页面没写死的（eval 频率、归档粒度、跑的顺序）→ 可调，但也记 log。
- **stage-1 预算线 $200，按 $1.6/h 判**（用户 2026-09-19 定；pod 实际单价 $1.59/h）：按 A1 / B1 各跑满 400 步（494 s/步，每 run ≈ $88 + eval）计的实际上限。`configs/cloud_4b.yaml` 里的 `cost.usd_per_hour: 1.9` 是旧值，改为 1.6 的指令已写进 `ops/to_execution.md` #2（重叠期 planning-led 不改 configs/）。第一批 eval 落地时用实测步时与步数重新投影；投影 > $200 → 停下问用户。总上限 $300 仍写死在配置里，超了不自动砍 seed。

## 待读结果时的判读清单（全部为预注册阈值，不改）
| 决策点 | 数据来源 | 判据 | 结果 → 动作 |
|---|---|---|---|
| A1 操纵门 | `results/matrix_log.md`、A1 `eval.jsonl` | 相对 step-0 L_mean 减少 ≥30%；acc 损失 ≤5pp；收敛 L_median ≥ 2×Kahn = 60 token | 全过 → 链继续；任一不过 → `--strict` 已停机，摆事实与选项（预注册的 λ=0.3 → 1.0 → G=32 重调 vs 其他）给用户 |
| A1 kill | 同上 | 收敛 L_median ≤ 1.5 × 决策下限 32.5 = 48.75 token | 触发 → 研究按 §5.3 停，最小可发表版；问用户 |
| A1 残留检查 | `uninterpretable.json` | acc ≥ 冻结 direct(0.45%)+10pp 后生效；direct ≥ acc − (acc − 0.45%)/2 连续两次 | 触发 → 该 run 不可解释，停下问用户 |
| A 作为比较器 | 06_diagnose | A 的移植与 unigram 重采样各掉 ≥ 一半 | 不过 → 无匹配论文（§5.3） |
| E2 早读 | Bwarm-SFT(A1) eval-only | 总 acc 在 A1 的 5pp 内 | 有效 → 长度比 = 编码税（A seed 1）；否则读法 = "words were load-bearing or the map was lossy" |
| warm 判定 | B1 ≤200 步 eval | B1 最佳 acc < A1 最佳 − 15pp | 触发 → B_warm-RL ×2 进入待跑（受预算检查）；不触发 → 不跑 |
| 继续与否 | 实测花费 + 重新投影 | 总投影 vs $300 | 用户决定：加预算 / 机构算力 / 停在最小可发表版；不自动砍 seed |

## 操作记录（planning-led；重叠期不写 `results/cloud_log.md` 以免与 execution 的提交冲突，接管后并入）
- 2026-09-19：读完 paper / README §2 / deviations_log / cloud_log / roles。本地无链结果（`runs/` 空，无 `matrix_log.md` / `CHAIN_DONE` / `matrix_halt.json`）。
- 2026-09-19：`docs/paper.md` 三处对齐现行协议（§4.4 cap → 10240 [D8]；§4.2 第 7 条 → 人工审计 [D13]；§7.4 → ordering IR 措辞），§6.1 与 §6.2 成本部分按 cloud_log / 准入 json 填入，§9 加两条记录。未提交（提交权重叠期在 execution）。
- 2026-09-19：`docs/paper.md` 第二轮对齐四处（§4.3 LoRA 去 embed [D5]；§4.2 第 4 条 [D10]；§4.4 门失败停机 [D14]；§4.1 / §4.2 原生长度改实测），§9 记一行；顺带更正 §6.1 第 8 条一行的标签（286.7× 是 M / Kahn，不是 M / (5×Kahn)；cloud_log 里的同一标签有同样的笔误，接管后在 cloud_log 加更正注记）。**用户指示：文档改动不提交，等链结束、接管后一起提交。**
- 2026-09-19：写 `ops/to_execution.md` #1（先报 A1 存活状态；再改 watcher 让 steps.jsonl 每 ~10 分钟流回本地，排除 .zst）。用户批准并转发。
- 2026-09-19：新增 `ops/dashboard.py`（只读 `cloud_pull/` → `ops/dashboard.html`；四块：在哪 / 每步曲线 / eval + 预注册阈值线 + 停止判据与残留检查表 / B1 对 A1）。约束：只显示已记录数字与预注册阈值线，不产生判定。`ops/dashboard_demo_FAKE.html` 是合成数据的版式预览（页面顶部有红色标注），可删。更正：此前"本地无链结果"查的是 `runs/`，watcher 的落点其实是 `cloud_pull/`；重查后 `cloud_pull/runs/` 里同样没有 A1（只有 dry-run / smoke）。

## 事件：A1 首个训练步崩溃（2026-09-19，用户转述 execution 的报告；本地尚无原始日志）
- **现象**：A1（`A_0.5_ord_n8_h4_d5_s1`）step-0 eval 正常完成；第一个训练步的 backward 报 `CUDA error: invalid argument`；`--strict` 下子进程非零退出 → 链已停机。没有任何训练步完成，没有 checkpoint。
- **step-0 eval（冻结策略，stopping 500 题，T=0.6，cap 10240）**：acc **0.624**、L_mean **8219**、强制收尾 **39.4%**。
  与准入（同一划分的另一次采样）一致：准入 native 0.608、强制收尾 37.8%、L 中位 8599.5；acc 差 1.6pp，在 n=500 的抽样误差内（SE ≈ 2.2pp）。dry-run 的训练采样 L_mean 8106（T=1.0）也同量级。
- **这组数对后面意味着什么（只是算术，不是判定）**：若它成为 A1 的 step-0 锚点，操纵门的两条线是 L_mean ≤ 0.7 × 8219 = **5753** 与 acc ≥ 0.624 − 0.05 = **0.574**。
- **排查（execution 进行中）**：① 3 步 dry-run、不带 eval；② 通过后加一次小 eval 复现。已知对照：同配置的 20 步 dry-run（无 step-0 eval）此前跑通（cloud_log），所以"训练前先做了一次 vLLM eval"是首要嫌疑，execution 的两步正是在分离这一点。
- **planning-led 此时不做任何事**，等排查结果。结果回来后需要对照的点：
  1. 修复是否触及预注册写死的内容（奖励、批大小 2×16、LoRA、cap、eval 划分 / 温度、停止判据）→ 触及则是 deviation，先停下问用户；只是工程修复（显存、eval 与训练的切换顺序、库版本）→ 记 log 即可。
  2. **重跑 A1 的 step-0 锚点 —— 用户已定（2026-09-19，在看到任何重跑数字之前）**：崩溃那次的 run 目录另存备查（不删）；重跑用干净目录；操纵门以**重跑自己的 step-0 eval** 为锚点；两次 step-0 的数都报（崩溃那次：acc 0.624 / L_mean 8219 / 强制收尾 39.4%）。
  3. 这次崩溃的花费（step-0 eval ≈ 1 h + 启动）计入 stage-1 已花。
- **修复方式的分类（用户 2026-09-19 定）**：
  - 指向 vLLM sleep mode / colocate 下 eval↔训练的切换（显存让出 / 唤醒、权重同步顺序、库版本）→ 工程修复，记 log 即可。
  - 修复方式是"训练前不跑 step-0 eval"或"改 eval 的 n"（或任何改变 eval 划分 / 温度 / 频率以绕开崩溃的做法）→ 动到操纵门的锚点与停止判据的数据来源（step-0 计入判定历史），**先停下问用户**，不自行采纳。
- 费率 1.9 → 1.6 的指令留在 `ops/to_execution.md` #2，**排查结束后**再随其它指令一起转发，现在不打断排查。

## 用户预先定的修复授权与验收（2026-09-20，dbg4 结果出来之前）
1. **授权**：`train.vllm_gpu_memory_utilization` 0.45 → 0.35~0.40 属工程修复，planning-led 可直接做，记 log。**不动**：批大小 2×16、cap、reward、LoRA、eval 划分。理由：预注册没写显存划分；reserved 一直贴 78.4 / 79.25 GiB 是唯一有实测支持的嫌疑。
2. **验收**：栈指向分配器 / 显存 → 修完后在这台机器上**连续 10 步不崩**（带一次小 eval，D16 的 `train.dry_run_eval_n`，确认 eval→训练切换在新 util 下也稳）→ 再重跑 A1。降了 util 还崩，或栈指不出原因 → 迁移；新机器用 `ops/host_check.py` + 10 步 dry-run（同样带小 eval）验收。

## 2026-09-20 05:33 UTC：stage-1 链已重启（D20 修复后）
- 根因：显存顶到上限时缓存分配器的回收路径出错（全冷编译 × util 0.45 崩 3/3）。修复：util **0.35**（用户定）、编译缓存持久化、链前预热。验收：util 0.35 全冷启动 3/3 过。详见 cloud_log / D20 / paper §9。
- 链：预热 → A1（先 step-0 eval ≈ 1 h）→ 门 → B_warm-SFT → B1 → warm 判定 → 停机。崩溃那次的目录在 pod 上为 `…_s1__crash0919`。
- **待办**：首批 eval 落地 → 用实测步时 / eval 时长重算 stage-1 投影并上报（最坏情形含 eval 已是 $220–225，高于 $200 线）；A1 结束 → 判读操纵门（清单见上表）。

## 2026-09-20：stage-1 预算线 $200 → **$240**（用户定；取代上文"$200 线"的所有表述）
- 理由（用户）：$200 是按漏算 eval 的投影划的内部线，不是协议数字；stage-1 的真实上界 $220–225 在 $300 内；这两个 run 正是决定后续是否继续的依据，不为 $20 中断。
- 规则：首批 eval 落地时照常用实测重算投影并上报；**只有投影 > $240 才停下问用户**。预注册的硬上限 **$300 不变**（写死在配置里，超了停下问用户，不自动砍 seed）。按 $1.6/h 判。

## 2026-09-20 08:15 UTC：夜间无人值守规则（用户睡前定）
- **需用户拍板的决策出现时**：planning-led 写 `/workspace/logs/DECISION_PENDING`（内容 = 决策摘要 + 选项），同时在这里和 cloud_log 记一条；`scripts/pod_watchdog.sh`（D21）在 90 min 后若该文件仍在即 `pod_stop --force`。用户回复后删除该文件取消倒计时。
- **不空转超过 1.5 h**：watchdog 另有一条 GPU 连续空转 ≥ 90 min 即停机。链自身在 `--strict` 停机 / 正常结束后已 `pod_stop --delay 300`，watchdog 只是兜底。
- **停机的代价**：checkpoint 每 25 步，`run_matrix` 下次自动 `--resume`，最多丢 25 步；重新开机需用户在 RunPod 控制台点（pod 上的 runpodctl 只能 stop 自己）。
- **不需要拍板、链自己继续的**：A1 收敛 → 操纵门由 run_matrix 判 → 过则自动进 B_warm-SFT → B1 → warm 判定 → 停机。门失败 / kill / 残留触发 / 崩溃 → `--strict` 停机 + 自动 pod_stop，等用户醒来再定。
- **可能需要拍板的**：每次 eval 后的预算重算若投影 > $240（当前最坏 ≈ $235–238，余量小）。

## 待办（2026-09-21）
- step-250 eval 落地后：用 `ops/analysis/acc_decomp.py` 更新 `ops/notes_accuracy_decomposition.md` 的分解表。
- **B1 跑起来后：对 B 用同一脚本**（`acc_decomp.py`、`group_deg.py`、`cot_scan.py`），脚本里写死的 run 路径需改。B 在字母屏蔽下正确率可能低得多 → 组退化率与截断率都要重算，"加 G 无依据"的结论对 B 不自动成立。

## 待用户裁决（2026-09-22 10:20 UTC 起）：B_warm-SFT / E2 的实现测试失败
- **事实**：warm 变换后 1435 条里 **1 条**策略类改变（mixed_probe_enum → symbolic_case，0.070%）。根因是 `branches = case_splits + count("try")` 而 warm 映射没有 `try` 的符号条目，该轨迹的 branches 由 8 降到 6，恰好卡在 `> 3×2^S = 6` 的严格大于边界上。其余字段全部不变。**算法未变，是检测器的字母依赖 + 边界效应。**
- **实现比注册文本更严**：§5.5 test 8 原文是"在 **100** 条 A 轨迹上分布相符"，实现是在全部 1435 条上精确相等断言。
- **选项（需用户定，planning-led 不自行采纳）**：
  (a) 按注册原文比对（100 条样本）或加容差（如 ≤1%）——只改"实现比注册更严"这一点，**warm 映射与检测器都不动**（planning-led 推荐）
  (b) 给 warm 映射加 `try → 符号` —— 改已提交的 warm_map，会改变 E2 的数值
  (c) 让分支计数识别 warm 符号 —— 改检测器（纯描述量）
  (d) 去掉该 assert —— 理由：策略类已是纯描述量、类别迁移读法已撤回
  (e) 不跑 E2，报"E2 未执行"
- **现状**：B1 已先跑（D24），E2 等裁决后用同一份 A1 `final/` 重跑即可，无任何损失。**未启用 DECISION_PENDING**（会在 90 min 后停机、打断 B1）。
