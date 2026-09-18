# v8 实现：改动 → 测试对照（2026-09-18；全部 CPU，未训练）

| # | 模块 | 改动（文件） | 测试（tests/） |
|---|---|---|---|
| 1 | 任务 | `tasks/ordering.py` 新：n 事件、h 硬约束、d 析取；**引导式生成**（一致边 = σ 的未被硬约束闭包蕴含的覆盖边，先无放回后有放回；不一致边在所有与 σ 矛盾的对里均匀；`GEN_MODE="uniform"` 只报接受率）；暴力唯一解（DFS 计数 + 子集 DP #LE(hard)）；S = 闭包传播 + 单元规则 + failed-disjunct probing（单独试一边、只算闭包）的最小决策树深度，只保留 S≥1；决策下限 d+n、Kahn 下限（就绪集轨迹 + failed-element probing）× c；格统计与拒格规则（`cell_stats` / `cell_decision`：#LE ≥20、闭包传播下首次非单元素就绪集位置中位 ≤ n/2（替代已作废的单元素步占比）、反链 ≥3、哈密顿路径 ≥3、存活析取 ≥3、保留率 ≥1%、同构类 eval ⊄ train）、`pairwise_accuracy` / `pairwise_baseline`（直接作答逐对诊断）；同构规范形（颜色细化 + 平局枚举）；四划分（train/stop/report/direct）独立 seed 流、零重叠断言；重编号与顺序随机；IR 题面；答案严格 / 宽松抽取；chance = 1/#LE(hard)；每题存 σ、S、两下限、规范路径、steps[t]（就绪集 / 下一事件 / 已定析取）。`tasks/__init__.py` 注册（make_eval_set split ∈ stop/report/direct）；`scripts/ord_build_splits.py`；`data/ordering/ord_n{8,9}_*_seed0.json` + `.report.json`（含引导前后接受率、不一致边跨越距离分布、cover_exact、重复覆盖边数、格判定） | `test_ordering.py`：手工小例（#LE、反链、哈密顿路径 K4=12、S=1 附推理、Kahn 轨迹逐步就绪集、两下限）；生成实例唯一解 / S 与独立暴力最小深度一致 / 一边一致一边不一致 / Kahn 下限与独立 DFS 闭包实现一致；规范形对重编号 + 顺序置换不变；答案规则；格统计与拒格；划分零重叠 + 注册表；数据文件（四格 5000 题全体唯一解、S≥1、每格 60 题 S 暴力核对）；`ORD_FULL=1` 10k 实例（四格各 2500：唯一解 + S 暴力核对，已跑 1 次通过，8.8 min） |
| 2 | 捷径 / 策略 | `cot_compress/shortcuts.py`：捷径 ir_copy_order（整 token 抄写 ≥50% 且含析取项 + 尾随排列）/ perm_enum / assign_enum（位串 ≥4、编号 case ≥ max(4, 2^(d−1))、Gray、try-both ≥ d）/ guess_verify；策略类 english_case / symbolic_case / propagation / enumeration（分支 > 3×2^S）/ mixed_probe_enum / path_stitching / other（优先级见文件头）；断言对解析（链展开、"a before b"）；复制率；`scripts/14_strategy_audit.py`（每臂 50 条 CSV + 精确率 / 召回率）。冻结原生轨迹（01_calibrate）与各臂收敛正确轨迹（06_diagnose）都跑 | `test_v8_detectors_decoder.py::test_shortcut_detectors / test_strategy_classes`（各类正例 / 反例、小 n 反例、命中率） |
| 3 | B_warm | `cot_compress/warm.py` 词典加 before/after/cycle/first/last/next/ready/order（`docs/warm_map.md`）；`scripts/11_sft_warm.py` 对 ordering 用 shortcuts 策略类（english+symbolic 合并比较） | `test_warm_preserves_ordering_strategy_distribution`（100 条：english→symbolic 为变换定义，其余类与逐条计数不变，无字母）；既有 warm 测试 |
| 4 | 解码器 | `cot_compress/decoder.py`：`parse_step`（发布的对齐解析器：前缀最长事件链）、`labels_for`（只由题目真值 steps[t] 决定）、`prefix_rows`（前缀 0.25/0.5/0.75/1.0，解析率）、`run_ordering_decoder`（next / ready / decided）、`soundness_completeness`、`cipher_spearman_warm`；`scripts/08_decoder.py`（仪器门：任一臂 next ≥90% 且在该臂的移植轨迹和 C_rand 轨迹上都掉 ≥10pp，配对 bootstrap；B 条款需解析率 ≥50%） | `test_parser_and_labels_independent_of_trajectory`（轨迹换随机符号标签不变；prefix_rows 标签 = labels_for）、`test_decoder_learns_next_event_on_synthetic`（≥0.9、移植 <0.6、bootstrap lo>0.1、按题划分无重叠）、`test_soundness_completeness_and_cipher` |
| 5 | 准入 | `cot_compress/admission.py` 按 §4.2 原文：(1) direct ≤ 2×chance（严格；宽松并报；逐对准确率 vs 均匀扩展诊断）(2) native ∈ [60,80] (3) 最小预算 ≥0.5M (4) 填充 (5) 移植向 direct 掉一半 (6) letter-ban (7) 捷径 <5% (8) M ≥ 5× Kahn 下限（c=2.5）；CELL_ORDER = 四格；`prescreen`；`01_calibrate.py --admission`：捷径命中率、Kahn / 决策下限中位、chance、严格与宽松、逐对诊断、原生策略类分布 | `test_admission.py`（八条逐条、边界、pending、预筛、格顺序） |
| 6 | 残留检查 | `scripts/train.py`：每次 eval 在 direct-check 2000 题上测直接作答；residual < 冻结差距/2 连续两次 → 停 run + `uninterpretable.json`；eval.jsonl 记 direct_acc / residual / lenient；数据集带 S 列（归档策略类用） | 逻辑在 `EvalCallback._record`（无 GPU 测试；analyze 读 uninterpretable.json 有测试） |
| 7 | 臂与矩阵 | `configs/matrix_core.yaml` v8（A1 → Bwarm_sft s1 → B1 → … ；A×3、A″×2、B×4、C_rand×2、Bwarm_sft×3（source: A_seed，各自跟在同 seed A 之后）、Bwarm×2 条件）；`run_matrix.py`：`--key`、speed_test（先 --dry-run 20 步）、warm 判定（B1 ≤200 步最佳 vs A1 最佳 − 15pp → results/warm_decision_<key>.json）、Bwarm_sft 用同 seed A；配置 key 改 ord_n8_h4_d5、direct_check_n | `test_analyze_matrix.py::test_run_matrix_prune_gate_retune`；dry-run 走通 |
| 8 | 端点 | `cot_compress/endpoints.py`：`curve_from_items`、`tax_from_curves`、`bootstrap_tax`（按题配对、seed 等权）、`zstd_bits_per_token`（并集 / 各组字典）、`tier_v8`；`scripts/analyze.py`：E1 y=总准确率（外化并列）、x=正确轨迹 token（无条件并列）、三单位 × bootstrap、MW 敏感性、三单位同档 claim、无匹配表、对两下限的比、direct 差 ≤3pp 可比性、残留停 run 标记；E2 按 §4.5（总准确率、按 A seed、"词承载结构或映射有损"）；E3 (i) B > C_rand 按题配对 seed 分块 bootstrap、(ii) 移植 / unigram 掉一半、(iii) 复制率、解码器门 + B 条款（解析率 ≥50%）、描述量；`06_diagnose.py`：预算行截断 token、mean_tokens_all、ordering 描述量（捷径 / 策略类 / 复制率 / 字母占比 / 命中率 / 下限中位） | `test_endpoints.py`（bootstrap 税：点估计在区间内、seed 配对、同比例换算不变、无匹配 None；档位；zstd 两种字典）、`test_analyze_matrix.py`（合成 run + diag_budget 文件：E1 全流程、不可计算表、direct 可比性） |
| 9 | 测试 | 见各行；沿用 mask / logprob / C_rand / B_warm / 奖励穷举 | `python -m pytest tests -q`：94 通过、2 跳过（opt-in 10k），7.3 min；ORD_FULL 10k 通过 |
| 10 | 文档 | README §2 v8 + 预注册 hash 占位；本文件；`scripts/13_instrument_control.py` 改用排序题（英文 152 token / 符号 84 / warm 107；比 0.55 / 0.70；无捷径命中）→ `results/instrument_control.json`（K&K 版存 `_kk.json`） | 脚本已运行 |

## 已按用户 2026-09-18 决定落实
1. "单元素步占比 ≤0.8" 作废，改为"闭包传播下首次非单元素就绪集位置中位 ≤ n/2"；四格重算（表见下与 README §2.1），四格全部通过。
2. 引导生成按现状，结论（cover_exact 100% 是构造必然；均匀接受率 0）已写进 README。
3. probing = 单边试、仅闭包；README 记录"probe 内加单元传播则全部 S=0（一层 lookahead 可解）"。
4. §4.2 / §4.5 原文已替换 v7 措辞（admission.py、analyze.py、README §2.5 / §2.7）。

## 四格统计表（stop 划分 500 题；`data/ordering/*_seed0.report.json`）
| 格 | #LE(hard) 中位 | 首次非单元素就绪集位置中位（直方图） | 单元素步占比中位（只报） | 反链 | 哈密顿路径 | 存活析取 | 冗余率 | S 分布 | Kahn 下限中位（元素） | 决策下限 | 逐对基线 | 均匀接受率 | 引导接受率 | S≥1 保留率 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (8,4,5) | 1680 | 0（0:499, 2:1） | 0.875 | 5 | 24 | 5 | 0.031 | 1:473 2:27 | 12 | 13 | 0.666 | 0/20000 | 1.11% | 30.6% | 通过 |
| (8,4,6) | 2240 | 0（0:499, 1:1） | 0.875 | 5 | 52 | 6 | 0.079 | 1:464 2:36 | 12 | 14 | 0.672 | 0/20000 | 3.29% | 41.9% | 通过 |
| (9,5,6) | 7560 | 0（0:497, 1:3） | 0.889 | 5 | 60 | 6 | 0.048 | 1:474 2:26 | 13 | 15 | 0.670 | 0/20000 | 1.33% | 32.3% | 通过 |
| (9,5,7) | 10080 | 0（0:496, 1:2, 2:2） | 0.889 | 5 | 116.5 | 7 | 0.097 | 1:451 2:49 | 13 | 16 | 0.677 | 0/20000 | 3.29% | 41.6% | 通过 |

注：首次非单元素就绪集几乎总在位置 0（h 条硬约束下起点就绪集通常 ≥2），该规则在这些格上不构成约束。Kahn 下限中位 ≈ n+4 元素 → c=2.5 时 30 / 30 / 32.5 / 32.5 token，第 8 条要求 M ≥ 150–163 token。
