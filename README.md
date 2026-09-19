# cot-compress：非语言中间表示的推理效率上限

## 1. Motivation

**核心命题。** 在必须靠推理才能做对的任务上（不思考时准确率 < 30%），非语言的中间表示达到基线准确率所需的最少 token 数，是否显著低于**经过长度优化的**自然语言中间表示；并且该非语言表示是否真的携带串行计算状态，而不是充当填充 token。

**已有工作的缺口。**

- IBM Abstract-CoT（arXiv 2604.22709）、ORION（arXiv 2511.22891）、CLSR（arXiv 2606.29354，ICML 2026）都声称非语言 / 符号化表示能把 token 数减少 3–16×。但它们全部：(a) 在不需要推理的 benchmark 上做（IBM 自己的对照显示不写 CoT 也有 86%）；(b) 只和从未压缩过的冗长 CoT 比；(c) 没有验证表示携带内容（IBM 截断到 32 token 几乎不掉分）。
- Little（arXiv 2607.09786）和 Kaufmann et al.（arXiv 2603.30036）在推理必需的任务上做了长度惩罚 RL，但输出始终是自然语言，没有测试非语言表示。
- 两组工作从未交叉。本实验的三臂设计正好补上：推理必需的任务 + 压缩过的语言对照 + 内容度量。

## 2. 实验设计 v8（2026-09-18 定稿；主任务 = 唯一线性序 + 析取先后约束）

**预注册 commit hash（freeze r4）：`d207c263df938f6003e51ff3da90f13367b40d99`**（r1 `eceff232`、r2 `288c874`、r3 `fd5c656` 作废；本行由 r4 之后的单独 commit 填入，预注册引用 r4）（生成器、检测器、白名单、B_warm 映射、配置、seed 与划分报告一并进 git）。历史设计（v4 cups、v7 K&K）见 §10 与 git 历史；K&K 代码与测试保留，不在主线。

### 2.1 任务 `tasks/ordering.py`（键 `ord_n{n}_h{h}_d{d}`，候选格顺序 (8,4,5) (8,4,6) (9,5,6) (9,5,7)）
划分数据 `data/ordering/<key>_seed0.json`（+ `.report.json`）已入 git（每格 ≤11 MB）；重建命令（确定性，seed 0）：`python scripts/ord_build_splits.py --key ord_n8_h4_d5 --seed 0 --workers 11`（其余三格同理；每格 1–8 min）。
- n 个事件 0..n−1；硬约束 i<j 共 h 条；析取 (i<j)|(k<l) 共 d 条，每条一边与真实顺序 σ 一致、另一边不一致；暴力枚举验证恰有一个线性序。
- **生成（引导式，用户定）**：先抽 σ；硬约束在 σ 中成立的对里均匀抽；析取的**一致边只从 σ 的覆盖边（相邻对）中、且未被硬约束闭包蕴含的那些里抽**（先无放回，不够 d 条再有放回）；**不一致边在所有与 σ 矛盾的对里均匀抽**（不按"最能缩减线性序数"挑，避免不一致边带可学信号）；最终仍暴力验证唯一性。
  记录并报告（`data/ordering/*.report.json` → `_generation`）：引导前（纯均匀）与引导后的接受率、不一致边在 σ 中跨越距离的分布、一致边集合是否恰等于"覆盖边 − 硬约束蕴含边"（cover_exact；实测四格均为 100%：唯一线性序要求每条未被蕴含的覆盖边都被某条析取强制，这是构造的必然）、重复使用的覆盖边数。
  实测（完整划分构建，见下表）：纯均匀接受率 0/20000（早期 10 万次探测 1 个）；引导后 1.11% / 3.29% / 1.33% / 3.29%；S≥1 保留率 30.6% / 41.9% / 32.3% / 41.6%。
- 决策深度 S：DPLL，传播器 = 传递闭包（硬边 ∪ 已选边）+ 单元规则（一边被闭包否定 → 另一边被迫；一边被蕴含 → 视为已定）；failed-disjunct probing = **单边试、仅闭包**（只加这一边算闭包，不做进一步单元传播），出环则另一边被迫；叶 = 全序或冲突；S = 最小决策树深度；只保留 S ≥ 1。
  **性质（实测记录）**：若 probe 内也做单元传播（一层 lookahead），引导生成的全部唯一解实例都变成 S=0——即这些题都能被一层 lookahead 解掉；S≥1 是相对"单边试、仅闭包"这一固定传播器定义的。
- 两个下限：决策下限 = d + n 个元素；Kahn 下限 = 沿 σ 的就绪集轨迹（就绪 = 硬边 ∪ 已定边闭包下无未放前驱，再经 failed-element probing 单独试、只算闭包剪枝）的就绪集元素数之和；各 × c ∈ {2, 2.5, 3}。
- 格统计（在引导生成的最终分布上算；命中拒格规则则拒，`tasks/ordering.cell_decision`）：#LE(hard) 中位 ≥20；**闭包传播下第一次非单元素就绪集的位置中位 ≤ n/2**（"单元素步占比 ≤0.8" 已作废：与 S=1 天然冲突；占比只报）；硬偏序最大反链中位 ≥3；无向被提及图哈密顿路径数中位 ≥3；probing 后存活析取数中位 ≥3；S≥1 保留率 ≥1%；全结构同构类 eval 与 train 不相交；S 分布、析取冗余率。
  **实测（stop 划分 500 题）**：

  | 格 | #LE(hard) 中位 | 首次非单元素就绪集位置中位（直方图） | 单元素步占比中位（只报） | 反链 | 哈密顿路径 | 存活析取 | 冗余率 | S 分布 | Kahn 下限中位（元素） | 决策下限 | 均匀接受率 | 引导接受率 | S≥1 保留率 | 判定 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | (8,4,5) | 1680 | 0（0:499, 2:1） | 0.875 | 5 | 24 | 5 | 0.031 | 1:473 2:27 | 12 | 13 | 0/20000 | 1.11% | 30.6% | 通过 |
  | (8,4,6) | 2240 | 0（0:499, 1:1） | 0.875 | 5 | 52 | 6 | 0.079 | 1:464 2:36 | 12 | 14 | 0/20000 | 3.29% | 41.9% | 通过 |
  | (9,5,6) | 7560 | 0（0:497, 1:3） | 0.889 | 5 | 60 | 6 | 0.048 | 1:474 2:26 | 13 | 15 | 0/20000 | 1.33% | 32.3% | 通过 |
  | (9,5,7) | 10080 | 0（0:496, 1:2, 2:2） | 0.889 | 5 | 116.5 | 7 | 0.097 | 1:451 2:49 | 13 | 16 | 0/20000 | 3.29% | 41.6% | 通过 |

  四格的同构类 eval ∩ train = 0；cover_exact 100%；不一致边跨越距离分布见各格 `.report.json`（距离 1 约 25%，随距离递减）。首次非单元素就绪集几乎总在位置 0（硬约束只有 h 条，起点就绪集通常 ≥2）。
- 划分：train 2000 / stopping-eval 500 / reporting-eval 500 / direct-check 2000，seed 流互不相交；每题事件随机重编号、约束与析取两边顺序随机；同构类去重与零重叠断言。
- IR 题面（所有臂相同）：`n=8` / `hard: 0<3 2<5 …` / `disj: (1<6)|(6<2) …` / `reason inside <think>, then give the order` / `answer: n digits, first to last`。答案 `<answer>n 个空格分隔事件号</answer>`，exact match，≤32 token；chance = 1/#LE(hard)；冻结准入同时报严格与宽松抽取。
- 每题存：σ、S、两个下限、规范路径（最低编号未决析取优先、先第一边）、每步真值就绪集 / 下一事件 / 已定析取（解码器标签）。

### 2.2 捷径与策略检测器（`cot_compress/shortcuts.py`，训练前冻结）
- 捷径（预注册时进准入门 <5%；**D13 起只是描述量**，r4 / D9 / D12 / D13 四种定义并报）：IR 抄写 + 尾随顺序；全排列枚举；析取分配枚举（位串 / 编号 case / Gray 码 / 嵌套 try-both）；guess-then-verify 循环。
- 策略类（**纯描述量**，不进任何门、不参与任何判定或读法）：英文分情况、符号分情况、传播为主、枚举（探索分支数 > 3× DPLL 最小值 2^S）、mixed_probe_enum（= 枚举 且（propagation 标记 ≥1 或 try-both 标记））、路径拼接（只用题面边连路）、其他；复制率 = ≥4 token IR 片段占比。
  定义以 `cot_compress/shortcuts.py` 为准：**case 标记** = `strategy.ROLES["case_splits"]`（assume / suppose / case / if）及其 B_warm 符号（» § ¿）的出现次数，分支数 = case 标记 + `try` 次数；**字母占比** = `lengths.letter_fraction`（think 段非空白字符中 Unicode L* 类字符的比例；英文 / 符号分情况以 0.3 为界，A 的"低字母占比"同为 <30%）。人工抽样审计 `scripts/14_strategy_audit.py`（每臂 50 条 → 精确率 / 召回率表）。冻结原生轨迹（01_calibrate）与每臂收敛后正确轨迹（06_diagnose）都跑。

### 2.2b 白名单（`cot_compress/vocab_mask.py` + `results/whitelist_extra_banned.json`，均入 git）
think 阶段 allowed set（Qwen3 分词器，embedding 行数 151936）：allowed = 解码后只含 Unicode N* / P* / S* / 空白的 token（无 U+FFFD 字节碎片、无其他 special token、无 `<answer>` 分词片段），**外加豁免 `</think>`**（该 token 含字母，但它是结束思考、解除屏蔽的唯一出口，始终允许；mask 在第一个 `</think>` 处解除，预算到顶时由控制器强制写入 `</think>\n\n<answer>`，强制 token 不进损失）。臂 B（letterfree）= **8351** 个 token（含 `</think>`）；臂 A″（letterfree ∪ 单字母 token，带 / 不带前导空格，共 104 个）= **8455**；臂 A = 全词表。
审计追加禁集 287 个 id（NFKC 后为字母、带圈 / 括号拉丁字母、区域指示符、可 leet 反解的词；`results/whitelist_audit.md`）已从白名单中扣除——不含该文件时会得到 8638 / 8742，属错误配置（r1 冻结 `eceff232` 漏了该文件，已作废）。

### 2.3 B_warm（`cot_compress/warm.py`，映射表 `docs/warm_map.md`）
控制词 → 固定符号（含 v8 的 before/after/cycle/first/last/next/ready/order），其余含字母 token 删除；100 条轨迹上策略类分布不变（英文分情况 → 符号分情况为变换定义，单测）；SFT（`scripts/11_sft_warm.py`）+ 无 RL 评估（`train.py --arm Bwarm --eval-only`）。

### 2.4 解码器（`cot_compress/decoder.py`，`scripts/08_decoder.py`）
仅前缀的 n-gram（n≤3）词袋，复制片段掩掉；标签 = 题目真值 steps[t]（下一事件 / 真值就绪集 / 已定析取），绝不从轨迹读取；解析不了的前缀排除并报解析率。
- **对齐规则（`parse_step`，发布并冻结）**：在前缀文本里找所有匹配 `\d+(?:\s*(?:<|→|->|,)\s*\d+)+` 的链（数字用 `<`、`→`、`->`、`,` 连接）；丢弃含有不在 0..n−1 内数字的链；步数 t = 各链"去重后事件个数"的最大值，封顶 n；一个链都没有 → 不可解析（该前缀排除）。标签取 `steps[min(t, n)]`。
- **前缀切法**：每条轨迹的 think 段按**字符长度**的 0.25 / 0.5 / 0.75 / 1.0 切四个前缀（不是按 token）；特征对切出的前缀文本再分词。仪器门：任一臂 next 准确率 ≥90% 且在该臂的移植轨迹**和** C_rand 轨迹上都掉 ≥10pp（配对 bootstrap）；B 条款只在 B 解析率 ≥50% 时生效。另报 soundness / completeness、B_warm 映射配对的密码 Spearman（描述量）。

### 2.5 准入 §4.2（`cot_compress/admission.py`；冻结 Qwen3-4B，n=500，全部通过）
先按 2.1 的格统计规则筛，再上 GPU 做八条：(1) 直接作答 exact（`<think></think>` 预填）≤ 2 × 2^−d，**在 direct-check 2000 题上测**；1/#LE(hard) 只报；逐对准确率 vs 硬偏序均匀随机扩展作诊断；严格与宽松抽取都报，准入用严格；(2) 原生 exact ∈ [60,80]%（S≥1 总体）；(3) 强制预算曲线 {0.1,0.25,0.5,0.75,1.0}×M 到 native−5pp 的最小预算 ≥0.5M；(4) 白名单 unigram 填充在 ≥0.5M 任一预算不进 native 10pp 内；(5) 原生轨迹移植 i→j（预填，只生成答案；配对 = 随机错位 derangement，`admission.derangement`）使准确率向 direct 掉一半以上；(6) 冻结 letter-ban ≤ direct+10pp；(7) 捷径检测器命中 <5%（**D13：改为人工审计"模板 / 抄答案"占比 <5%，四个检测器降为描述量，见 `docs/deviations_log.md` 与 `results/audit_criterion7_<key>.md`**）；(8) 中位原生长度 ≥ 5 × Kahn 下限（c=2.5）。
M（奖励里的长度归一化）= 准入这 500 条原生轨迹的中位长度（`results/admission_<key>.json["M"]`），不再用 M.json 的 32 题。
训练后残留检查（`cot_compress/stopping.py`；**C_rand 豁免**）：每 50 步在 direct-check 2000 题上测直接作答；**仅当带轨迹 acc ≥ 冻结 direct + 10pp 之后生效**（锁存）；停止条件 direct ≥ acc_trace − (acc_trace − frozen_direct)/2 连续两次 → 停 run，标不可解释；E1 比较要求臂间训练后直接作答差 ≤3pp，否则标 "not comparable"。

### 2.6 训练（`scripts/train.py`，`configs/cloud_4b.yaml`，`configs/matrix_core.yaml`）
臂 A×3、A″×2、B×4、C_rand×2、B_warm-SFT（每个 A seed 一个）、B_warm-RL×2（条件：B1 在 ≤200 步的最佳准确率 < A1 最佳 − 15pp）。LoRA attention + MLP + embed_tokens（lm_head tied 不挂；embed 按 `scripts/10` 的 q+embed 门控），r=32 α=64 dropout 0，lr 1e-5 warmup 10；奖励 v4（v 只在答案区）；训练 T=1.0，评估 T=0.6 top-p 0.95 top-k 20；停止：连续两个 50 步区间 |ΔL|/L_后一次 <5% 且 |Δacc|<4pp（stopping-eval 500 题上的 L_mean 与 acc；**分母用后一次 eval**）。400 步未收敛 → 以 step-400 为指定 checkpoint；每个 run 结束写 `designated_ckpt.json`（converged / max_steps / uninterpretable）。
**step-0 eval 计入判定历史**：训练开始前的 eval（step 0，即冻结策略）进入 `hist`，因此最早可在 step 100 触发收敛停止（需要 step 0 / 50 / 100 三个点构成两个连续区间）；残留检查的连续计数同样从 step 0 开始。
操纵门（A seed 1，`run_matrix.first_run_gate`）：相对 step-0 eval token（L_mean）减少 ≥30%、acc 损失 ≤5pp、收敛 L_median ≥ 2 × **Kahn 下限**中位（c=2.5，与 §4.2 第 8 条同一下限；c=2 / 3 只报）；失败按 λ=0.3 → λ=1.0 → G=32 重调。**kill 判定**：A1 收敛 L_median ≤ 1.5 × `decision_lb_tokens["2.5"]` 中位数 → 矩阵直接停下报告，不重调（kill = "压到底了没有空间"，重调 = "压不动"，两者互斥）。
残留检查见 §2.5（acc ≥ 冻结 direct + 10pp 后生效；direct ≥ acc − (acc − 冻结 direct)/2 连续两次；C_rand 豁免）；收敛时臂间直接作答差 ≤3pp 才可比，否则 "not comparable"。
C_rand seed i 绑 B seed i（think 段 unigram 与长度分布取自同 seed 的 B 收敛轨迹）；think 长度**每个 prompt 组抽一次**（同组 G 条生成共享同一段上下文）。B_warm-SFT 的评估挂 B 的 mask（`--arm Bwarm --eval-only`，letterfree）。run_matrix 顺序：A1（20 步测速 → 收敛，操纵门）→ Bwarm-SFT s1（只依赖 A1，立刻做）→ B1 → 200 步 warm 判定 → B2–4 → A2 → Bwarm-SFT s2 → A3 → Bwarm-SFT s3 → A″ → C_rand → 条件 Bwarm-RL（跑之前检查预算投影：已花 + 待跑 × 每 run 投影 vs `cost.budget_usd` = **300 美元（写死）**；超预算 → B_warm-RL 跳过，记 "search-budget result"，写 `results/search_budget_<key>.json`）。

### 2.7 端点 §4.5（`cot_compress/endpoints.py`，`scripts/analyze.py`）
E1：每个收敛策略的强制预算曲线（自身收敛长度 0.1…1.0 倍，reporting-eval，每 seed isotonic）；y = 总准确率（外化并列），x = 正确轨迹 token（无条件长度并列）；匹配 y = min_臂（该臂各 seed **自身收敛预算处** isotonic acc 的等权均值）− 5pp；税按题配对 bootstrap，seed 等权分块；seed 级 Mann-Whitney 敏感性表；token / code point / zstd-19 bits（并集字典 + 各臂字典）；档位区间：[67,∞) strong / [25,67) moderate / [10,25) small / (−10,10) none / (−∞,−10] negative；**区间化**：每个单位的 bootstrap 区间两端同档才给档（跨档只报 span），三单位同档才声称；点估计所在档另行并报（不用于声称）；附：冻结→A 压缩、A 主流策略类内的税（枚举 = 探索分支 > 3× DPLL 最小值，人工审计样本 `scripts/14_strategy_audit.py`）、对两个下限的比、按题匹配长度、A 字母占比（低 = <30%）；**不可计算 = 任一臂曲线不穿过匹配 y**（该臂没有任何 seed 的 isotonic 曲线达到 y*；个别未达到的 seed 记入 non_crossing、不进税）→ 输出无匹配论文表。策略类内的税等附项均为纯描述，不产生任何读法。
E2：B_warm-SFT 总准确率在源 A 5pp 内才有效，长度比 = 编码税，按 A seed 报；否则"词承载结构或映射有损"。
E3（对 B 必需）：(i) B > C_rand 匹配长度（按题配对，seed 分块）；(ii) 移植（随机错位配对，与准入第 5 条一致）和 unigram 重采样各使 B 准确率向训练后 direct 掉一半以上；(iii) 复制率 <30%。解码器：仅前缀 n-gram（n≤3）词袋，复制片段掩掉；标签 = 目标题真值（答案一致的析取选择、真实就绪集 / 下一事件），按发布的解析器对齐模型步数，绝不从轨迹读；解析不了的前缀排除，报解析率。仪器门：任一臂 ≥90% 且在该臂移植 / C_rand 轨迹上掉 ≥10pp；B 的条款只在 B 解析率 ≥50% 时生效。soundness / completeness 和密码统计为描述量。

## 3. Success Criteria（v8，写定，不事后改）

判定全部由 §2.5（准入）、§2.6（门 / 残留检查）、§2.7（E1 / E2 / E3）的预注册规则给出，不另设口头标准：
1. E1 可计算（三臂曲线都穿过匹配 y，臂间训练后直接作答差 ≤3pp，无不可解释的 run）；
2. 税的档位按区间化规则给出（每个单位的 bootstrap 区间两端同档，且 token / code point / zstd bits 三单位同档）；
3. 对 B 的任何正面表述都以 E3 (i)–(iii) 全部满足为前提；解码器结论以仪器门通过为前提。

## 4. 预期结果的读法（v8，写定）

- 税的 bootstrap 区间整体落在 [25,67) 或 [67,∞)（三单位同档）且 E3 满足 → 无字母表示在匹配准确率下更短，报告该档（moderate / strong）。
- 区间整体落在 (−10,10) → 无税：长度优化后的自然语言已与无字母表示等长。
- 区间整体落在 (−∞,−10] → 负税，单独报告。
- 区间跨档（或三单位不同档）→ 只报区间与 span，不声称档位。
- E1 不可计算、直接作答差 >3pp（"not comparable"）、或 B 冷启动失败且 B_warm-RL 未恢复 → 不作长度比较，输出无匹配表与 E2（B_warm-SFT 有效则报编码税，否则"词承载结构或映射有损"）。

## 5. 内容度量（eval.py 必做）

| 度量 | 做法 |
|---|---|
| 截断敏感性 | 思考段截到 25% / 50% 长度后接 `</think>` 强制作答，记录准确率下降 |
| 打乱敏感性 | 随机打乱思考段 token 顺序后作答，记录下降 |
| 线性探针 | 状态跟踪每步真实状态已知；思考段等间隔位置取 hidden state，线性探针预测当前杯子状态，报告准确率随位置曲线。臂 A、B 都做 |
| 参考模型困惑度 | Qwen3-0.6B-Base 下思考文本每 token PPL |
| 屏蔽集占比 | 思考段落在屏蔽集里的 token 比例（臂 A 也算） |
| 压缩率 / 熵 | gzip 字节比、bigram 熵 |

## 6. 模型选择

| 位置 | 模型 | 加载 | 说明 |
|---|---|---|---|
| 本地主实验 | Qwen3-1.7B | 4-bit QLoRA | Unsloth GRPO 最成熟路径 |
| 本地可选 | Qwen3.5-2B | bf16 LoRA | Unsloth 文档明确不推荐 Qwen3.5 做 4-bit；混合 GatedDeltaNet 架构需要 flash-linear-attention / causal-conv1d，对 sm_120 可能要本地编译。实测显存后决定是否用 |
| 云端主 | Qwen3-4B | bf16 LoRA + vLLM | 与 Little 2026 对齐，可用其压缩率 / 准确率校准臂 A |
| 云端第二 | Qwen3.5-9B | bf16 LoRA + vLLM | |
| 困惑度参考 | Qwen3-0.6B-Base | bf16 | 与 Qwen3-1.7B 共用 tokenizer |

**降级说明。** 原目标 Qwen3.5-2B。Unsloth 当前支持 Qwen3.5 GRPO，但要求 `fast_inference=False`，且文档明确"不推荐对 Qwen3.5 做 QLoRA 4-bit 训练"。8 GB 显存下 bf16 LoRA + GRPO 采样余量极小，故本地主实验降级到 Qwen3-1.7B 4-bit。

## 7. 环境

- 宿主机 Windows 10 22H2，RTX 5060 8 GB（Blackwell sm_120），驱动 591.74。一切在 WSL2 Ubuntu 24.04 内做，用户 `zechen`，项目 `~/cot-compress`，`HF_HOME=~/hf`。不碰宿主机 Python，不把数据放 `/mnt/c`。
- PyTorch 只从 `https://download.pytorch.org/whl/cu128` 装，bitsandbytes ≥ 0.45.3。cu121 / cu124 一律不用。本地不装 vllm。
- **导入顺序约定。** `cot_compress/__init__.py` 无条件先 `import unsloth`；所有 `scripts/*.py` 第一个 import 必须是 `import cot_compress`。原因：trl 0.24.0 的 `trainer/callbacks.py` 无条件 import mergekit（未安装），裸 `from trl import GRPOTrainer` 直接 `ModuleNotFoundError`，只有 Unsloth 先导入时才能绕过；Unsloth 本身也要求先于 transformers/trl 导入。
- 训练可用显存约 6.8 GB（实测：torch 可见 7.96 GiB，Windows 桌面常驻占用约 1.3 GiB）。**但 WSL 共享内存溢出阈值更低**：2026-09-16 实测 batch 8 × 3072 生成时 torch 峰值 4.23 GiB，上下文过约 2048 时解码从 21 steps/s 跌到 1.8（`results/calib/*.batches.json` 的 `steps_per_s_by_256`）。经验规则：torch 峰值分配 ≤ 3.5 GiB 时不溢出；生成用 batch ≤ 6、序列 ≤ 2.4k。训练配置按这个上限设计。**第二种溢出来源是缓存分配器碎片**：多次变长 generate 后 reserved 涨到占满显卡（allocated 2.9 GiB 时 nvidia-smi 7.85 GiB），同样触发换页（21→5 steps/s）。对策：每个 batch 后 `torch.cuda.empty_cache()`（`generation.generate_texts`）；训练里生成前后 `gc.collect()+empty_cache()`（`train.py` 包装 `trainer._generate`）。`expandable_segments` 在 WSL **不受支持**（unsloth_zoo 导入时会把它从环境变量里剥掉，2026-09-17 实测进程内为 None），改用运行时可设的 `garbage_collection_threshold:0.6,max_split_size_mb:256`（`cot_compress/memory.py`）。smoke A #1：step 1 alloc 峰值 2.93 GiB 但 reserved 6.55 GiB，step 2 训练前向 cudaErrorMemoryAllocation。本地配置：4-bit 加载，`num_generations=4`，`max_completion_length=512`，`per_device_train_batch_size=1` + 梯度累积，Unsloth 梯度检查点。
- 安装步骤见 `setup/`：`apt.sh`（需要 sudo，由用户执行）→ `wslconfig.txt`（放到 `%USERPROFILE%\.wslconfig`，`wsl --shutdown`）→ `install.sh`（uv 环境）→ `scripts/00_check_env.py`。

## 8. 执行阶段与预计耗时

| 阶段 | 内容 | 预计耗时 | 显存 | 状态 |
|---|---|---|---|---|
| 1 | 环境 | 30 min | — | 进行中 |
| 2 | 任务 + 校准 + 臂 0 | 实测：每难度 100 题原生 2048 预算约 27 min（batch 6，92 s/batch），direct + 3 个强制格约 3 min；4 个 mul 难度共约 2 h | 峰值 2.87 GiB（reserved 3.0） | 校准完成 2026-09-16 |
| 3 | 计算 M | 实测：32 题 × 1 次采样（T=0.6，batch 4，cap 3072）1000 s；M(mul_3x3)=1897.5，自然结束 28/32 | 峰值 ~2.9 GiB | 完成 2026-09-17 |
| 4 | GRPO 脚本 + smoke test | 实测臂 A λ=0：3 步 174/175/141 s（均 163 s/步，含 4×≤3072 生成 + 4 次前向/反向）；生成 batch 内并行（一次 generate 4 条）；12/12 自然结束 | alloc 峰值 2.98 GiB，reserved 5.79（生成后释放） | 完成 2026-09-17；首版在 step 2 OOM，见 §7 |
| 5 | mask + 臂 B smoke test + 样本人工检查 | 约 20 min | 待实测 | |
| 6 | 分析脚本 | — | — | |
| 7 | 正式本地 run：臂 A(λ=0.3) 200 步（臂 B 今晚只 smoke） | 估：200 × 165 s ≈ 9.2 h + 9 次 eval（50 题 native@3072 + @1024，估 25–30 min）≈ 4 h → ~13 h | 同上 | 2026-09-17 夜启动 |

数字会在实测后更新，更新记入第 10 节。

**实测吞吐（2026-09-16，Qwen3-1.7B 4-bit，HF generate）。** 单流 14.4 tok/s；batch 4 时 77 tok/s 总吞吐（每步约 19 次前向/s）。单流下一条 512 token 思考约 35 s，GRPO 每步 4 条生成若串行约 2 分多钟。**阶段 4 写训练脚本时必须确认 Unsloth 的 GRPO 生成是 batch 内并行的**（`num_generations` 条一次前向），并把实测每步耗时填回上表。

## 9. 云端切换（正式实验全部上云；本地只做开发和 smoke test，2026-09-17 定）

### 9.1 技术栈与路径选择

- 目标机：A100 80GB，Ubuntu 22.04，CUDA 12.x。安装 `setup/install_cloud.sh`（uv 环境；vllm 先装以固定 torch；`trl>=0.25`；unsloth 可选）。装完脚本内置断言 `rollout_func` 存在。
- **训练路径 = TRL 原生 `GRPOTrainer` + PEFT + `use_vllm=True, vllm_mode=colocate` + `rollout_func`**（`configs/cloud_4b.yaml: model.use_unsloth=false`）。
  原因：Unsloth 的 `fast_inference` 自带 vLLM 但不暴露 `logits_processors`，臂 B 的屏蔽与预算强制注入不进去；TRL≥0.25 的 `rollout_func` 让我们自己调 `llm.generate(SamplingParams(logits_processors=[VLLMSpanProcessor]))`（`cot_compress/rollout.py`）。
  Unsloth 路径（`use_unsloth=true`）保留为无 vLLM 的备选，语义与本地相同。
- eval 也走 vLLM（`generation.backend=vllm`）：300 题 × 3k token 用 HF generate 每次约 20 min，vLLM 约 5 min。
- **本地未测的部分（第一小时必须验证）**：`rollout.py` 全部；TRL 0.25 `rollout_func` 收到的 prompts 是否已按 `num_generations` 重复、返回的 logprobs 格式；vLLM 是否接受 per-request `logits_processors`（V1 引擎可能不接受 → `export VLLM_USE_V1=0`）；colocate 下 `trainer.llm` 可用于 eval。
- 屏蔽（臂 B）与预算强制在 vLLM 里的语义与本地一致：含 `</think>` → 不屏蔽；长度 ≥ `cap − reserve − len(close)` → 逐 token 强制 `</think>\n\n<answer>`；否则屏蔽全部含字母类字符的 token。

### 9.2 第一小时验收清单（v4，按顺序）

| # | 步骤 | 命令 | 预期 | 耗时 |
|---|---|---|---|---|
| 0 | 同步 | 本地 `CLOUD=user@host bash setup/sync_cloud.sh push` | 只传代码/配置 | 1 min |
| 1 | 安装 | `bash setup/install_cloud.sh` | 末尾 `rollout_func available: True` | 8–12 min |
| 2 | 环境检查 | `python scripts/00_check_env.py` | `trl_hooks.rollout_func == true` | 3 min |
| 3 | 白名单审计 | `python scripts/07_audit_whitelist.py --tokenizer Qwen/Qwen3-4B` | `results/whitelist_audit.md`；标记数与明细人工过目 | 2 min |
| 4 | vLLM 处理器验证 | 用 `rollout.VLLMSpanProcessor` 对 `vllm.LLM(Qwen3-4B)` 生成 4 条 cap=64 | 出现 `</think>\n\n<answer>`，臂 B 样本 0 字母；不通过 → `export VLLM_USE_V1=0`；再不通过 → `use_unsloth=true` 退 HF generate | 5 min |
| 5 | 准入校准（含 K/2 格） | `python scripts/01_calibrate.py --config configs/cloud_4b.yaml --admission --keys cups_n6_k12 cups_n6_k6 cups_n8_k16 cups_n8_k8 --n 500` | `results/admission_<cell>.json` 里 `admitted` 且 pending 只剩 7_D | 60–90 min（每格 n=500：native + 5 预算 + 5 填充 + 移植 + mask） |
| 6 | 算 M | `python scripts/02_compute_M.py --config configs/cloud_4b.yaml --key <准入格>` | `results/M.json` | 5 min |
| 7 | 臂 A smoke | `python scripts/train.py --config configs/cloud_4b.yaml --arm A --smoke` | 3 步各 ≤ 90 s，自然结束 ≥ 50%，显存 ≤ 70 GB，日志里 `logp_renorm=full-vocab` | 8 min |
| 8 | 臂 B smoke + 零命中 | `--arm B --smoke` 后 `python scripts/04_check_mask_samples.py --run runs/smoke_B --n 10` | `PASS`；日志 `logp_renorm=on`；reward_log 里 `hard` 违规（`</think>` 先行）的 r = −5 | 8 min |
| 9 | 启动 A seed 1 并等操纵检查 | `nohup python scripts/run_matrix.py --matrix configs/matrix_core.yaml > matrix.log &` | 第一个 run 收敛后 `results/matrix_log.md` 出现 `first-run gate ... ok: true`（token −30%、acc −≤5pp），否则矩阵停下 | ≤ 400 步 ≈ 5 h |

### 9.3 实验矩阵与估算

核心（`configs/matrix_core.yaml`）：2 任务（mul_3x3*、cups_n6_k12*）× 臂 A/B × λ ∈ {0.3, 0.7, 1.5} = **12 run**；关键格第二 seed（主 λ 那档的 4 格）= **4 run**；条件项：臂 B 冷启动失败 → 瓶颈 SFT 预热（`cot_compress/sft_warmup.py`，未实现）后再跑；成功 → 臂 C 填充对照（1–2 run）；规模点（`configs/matrix_scale_8b.yaml`）：Qwen3-8B 上 cups 的 A/B 各 1 = **2 run**。*难度键以云端校准为准。

每 run 200 步、G=16、cap 3072、eval n=300 每 25 步，A100 80GB 估算（±2×，vLLM 吞吐与自然长度未实测）：

| 项 | 4B | 8B |
|---|---|---|
| 每步：vLLM 采样 16 × ~2.5k token（~2.5k tok/s） | ~16 s | ~30 s |
| 每步：训练 16 × 3.3k token LoRA 前向/反向（梯度检查点） | ~25 s | ~50 s |
| 每步合计 | ~45 s → 200 步 ≈ 2.5 h | ~85 s → 200 步 ≈ 4.7 h |
| eval（300 题 vLLM + @1024 强制）× 8 | ~50 min | ~80 min |
| 诊断（06_diagnose，n=100） | ~15 min | ~25 min |
| **每 run** | **≈ 3.5–4 h** | **≈ 6.5–7 h** |

总量：12 + 4 = 16 个 4B run ≈ 60 h；2 个 8B ≈ 14 h；条件项 ≈ 8 h；校准/M/smoke ≈ 2 h → **≈ 85 GPU·h**。按 A100 80GB ≈ $1.9/h（RunPod/Lambda 量级）≈ **$160**；估算不确定度 ±2×，先跑完 1 个 4B run 校正后再定预算。

### 9.4 抢占与恢复

所有云端脚本每 25 步存 checkpoint（`train.save_every`）；`run_matrix.py` 对已有 checkpoint 的 run 自动加 `--resume --no-eval-at-start`，对已有 `final/` 的 run 跳过，`results/matrix_log.md` 记每个 run 的返回码与耗时。单个 run 手动恢复：`python scripts/train.py --config configs/cloud_4b.yaml --arm B --tag _<key>_s<seed> --resume --no-eval-at-start --set reward.lambda=<λ> task.key=<key> train.seed=<seed>`。

## 10. v7 协议摘要（历史：稀疏 Knights-and-Knaves；2026-09-18 被 v8 取代，代码与测试保留）

预注册 commit hash：见 §2（freeze r4 `d207c263df93`）（生成器 tasks/kk.py、检测器 templates.py / strategy.py、白名单 vocab_mask.py + results/whitelist_extra_banned.json、B_warm 映射 warm.py + docs/warm_map.md、configs/*.yaml、data/kk/*.json 的 seed 与划分报告一并进 git。）

| 模块 | 文件 | 要点 |
|---|---|---|
| 任务 | `tasks/kk.py`, `scripts/kk_build_splits.py`, `data/kk/` | 主格 kk_n10_s1（备用 kk_n12_s1）：N 人，宽度 2，9 种形式**均匀**，每人被引用 ≤3；均匀构造式采样 + 拒绝，只保留天然 S=1（单元传播 + failed-literal probing 解不掉、一次分情况即可；N=10 接受率约 0.2–0.4%，约 2.5 s/题/核）；暴力唯一解；规范形去重（颜色细化 + 平局枚举）；train 2000 / stopping-eval 500 / reporting-eval 500，seed 流互不相交，断言零重叠；IR 题面（可选英文 gloss，只作参考）；答案 N 个 0/1，exact match，chance 2^-N；oracle 下限 = 最小树文字数 × c，c∈{2,2.5,3}；每题记 gold_tree 与 decoder_labels[k]，k∈{0,1}。S=2 手工实例只留在单测里作算法测试 |
| 模板检测器 | `cot_compress/templates.py` | dpll_cdcl / bitmask_enum / english_assume / ir_copy_assign；准入第 7 条：≥5% 原生轨迹命中 → 拒格 |
| 策略类检测器 | `cot_compress/strategy.py` | 分情况标记数、传播步数、枚举、dump-and-verify、复制率（≥4 token IR 片段）；词表 ⊆ warm 控制词，B_warm 变换下计数不变 |
| B_warm | `cot_compress/warm.py`, `docs/warm_map.md`, `scripts/11_sft_warm.py` | 控制词 → 单 token 符号，其余含字母 token 删除；SFT（fresh adapter，同 LoRA 配置）+ `train.py --arm Bwarm --eval-only` 无 RL 直接评估 |
| 解码器 | `cot_compress/decoder.py`, `scripts/08_decoder.py` | L2 多项 LR，n-gram（n≤3）词袋，复制片段掩掉；标签 = 深度 k∈{0,1} 单元传播后的部分赋值（k=1 = 分情况之后）；按题划分；A 上 k=1 acc_assigned ≥ 0.90 为仪器检查；B vs 移植 / C_rand 基线，配对 bootstrap |
| 准入 | `cot_compress/admission.py`, `scripts/01_calibrate.py --admission --backend vllm --keys kk_n10_s1 kk_n12_s1` | 八条：(1) direct ≤ chance+10pp；(2) native ∈ [60,80]；(3) 到 native−5pp 的最小预算 ≥0.5M（只在 acc > chance+10pp 的点比较）；(4) 白名单填充在 ≥0.5M 任一预算不进 native 10pp 内；(5) 移植使准确率向 chance 掉一半以上；(6) 冻结 letter-ban ≤ direct+10pp；(7) 模板命中 <5%；(8) 中位原生长度 ≥ 5× oracle 下限（c=2.5；c=2/3 只报）。格顺序 kk_n10_s1 → kk_n12_s1；训练后逐臂 direct ≤ 冻结 direct + 10pp |
| 臂 | `scripts/train.py`, `configs/cloud_4b.yaml`, `configs/matrix_core.yaml` | A、A″、B、C_rand、B_warm-SFT、B_warm-RL（条件：冷启动失败）；LoRA attention+MLP+embed_tokens（r32 α64；**lm_head 不挂**：Qwen3-4B tied embeddings，lm_head 增量同步不进 vLLM，不做 untied 副本；embed 按 `scripts/10_check_lora_embed_sync.py` 的 q+embed 结果门控，`cot_compress/lora.py`）；lr 1e-5，10 步 warmup；训练 T=1.0，评估 T=0.6/top-p 0.95/top-k 20；奖励 v4（v 只在答案区）；停止：连续两个 50 步区间 ΔL<5% 且 Δacc<4pp（stopping-eval）；操纵门 A s1：≥30% 压缩、≤5pp 损失、L ≥ 2× oracle 下限（c=2.5，c=2/3 只报），失败 λ=0.3 → λ=1.0 → G=32；冷启动失败：B 最佳外化 < A 最佳 − 15pp；B 命中率（exact / Hamming≤2）每次 eval 记 |
| 端点 | `cot_compress/endpoints.py`, `scripts/analyze.py`, `scripts/06_diagnose.py` | E1：强制预算曲线（0.1…1.0 × 自身长度，y = 外化准确率，x = 正确轨迹 token）→ isotonic → 匹配准确率 = A/A″/B 全达到的最高 y − 5pp（5pp 内无点则不计算）→ 税 = (L_A−L_B)/L_A 按 seed + 范围 + 精确单边 Mann-Whitney，三单位（token / code point / zstd-19 共享字典 bits），阈值 25%/67%，<10% 无税，≤−10% 负税单独报；附加：冻结→A 压缩量、A 主流策略类内的税、按题匹配次要长度、A 字母占比曲线（<30% 为低）。E2：B_warm-SFT 外化在 A 5pp 内才有效，长度比 = 编码税。E3：B > C_rand（每 seed）；移植 / unigram 各掉一半以上；复制率 <30%；解码器 B ≥ 基线 + 10pp；替换密码检查（Spearman ≥ 0.85 判密码） |
| 成本 | `scripts/train.py --dry-run`, `cot_compress/cost.py` | 20 步：每步墙钟 / token / 显存峰值 → 400 步 × 矩阵的 GPU 小时与费用 |
| 仪器阳性对照 | `scripts/13_instrument_control.py` → `results/instrument_control.json` | 手写 DPLL 推导英文版 401 token / 符号版 222 token（比 0.55；code point 比 0.23）、warm(英文) 285 token；模板 / 策略检测器在三版上的输出（进论文附录） |

## 11. 调整日志

| 日期 | 调整 | 原因 |
|---|---|---|
| 2026-09-15 | 本地主模型 Qwen3.5-2B → Qwen3-1.7B 4-bit | 见第 6 节 |
| 2026-09-16 | 环境通过：torch 2.11.0+cu128 / bnb 0.50.2 / unsloth 2026.9.4 / trl 0.24.0 / transformers 5.5.0；Qwen3-1.7B 4-bit 加载 1.36 GiB，峰值 1.41 GiB，单流 14.4 tok/s | `results/env_check.json` |
| 2026-09-16 | 云端 vllm rollout 路径标记 requires trl>=0.25，本地不动 trl 版本；加云端切换清单（§9） | trl 0.24.0 无 `rollout_func` |
| 2026-09-16 | 约定 `import cot_compress`（即 unsloth）先于 trl；`00_check_env.py` 改为显式先导 unsloth 并读源码检测 `rollout_func` | trl 0.24 裸导入失败；Unsloth 补丁后签名检查失效 |
| 2026-09-16 | 状态跟踪任务砍掉 `move`，只留 swap/flip | move 语义易歧义、每步约 700 token；1536 预算内只推到第 2 步 |
| 2026-09-16 | 校准增加 (b) guided 基线；选择标准加中位思考 ≤ 1500、触顶 < 10%；训练加预算强制 90% | 原生思考基线 2k–8k，本地跑不动 |
| 2026-09-16 | 评估记录每 batch 耗时 / 峰值显存 / 每 256 步 steps-per-s | batch 5 × 6144 生成近 1 小时无输出，疑似 KV 溢出 WSL 共享内存 |
| 2026-09-16 | 小样本 probe（n=3/格，预算 3072）：(a) 原生思考 mul 9/9 但中位 2.0–2.8k；cups 9/9 触顶；(b) guided mul 2–3/9、cups 1/9、2/12 复读循环；direct 1/9 / 1/9。(a′)=(a)+预算强制（`</think>\n\n<answer>`）mul@1536 9/9、@1024 5/9、@768 5/9；cups@任意 ≈0 | 见 §2 候选基线；预算强制收尾必须连 `<answer>` 一起预填，只追加 `</think>` 时模型在答案区继续算 |
| 2026-09-16 | (b) 系统提示三版：v1 "think inside <think>" 有推理但不写标记；v2 "Reason step by step, then give only the final answer" 使模型直接作答（0 推理）；v3 "Show your step-by-step working … do not answer directly" 恢复推理 | 非思考模式对 "give only the final answer" 极敏感 |
| 2026-09-16 | 长跑 GPU 任务用 `setsid nohup` 脱离 Claude Code 任务跟踪，日志 tail 监控 | 其内存守卫在 WSL 上误判（`free` 显示 20 GB 可用时仍 kill），连 sleep 循环也被杀 |
| 2026-09-16 | 臂 0 = (a′)（原生思考 + 预算强制 B），本地主任务 = mul，cups 移云端 4B；加 §2.1 语言参与度 | 小样本 probe：(a′)@1536 mul 9/9，cups 0/9；(b) 两任务都不可用 |
| 2026-09-16 | 校准生成改 batch 6、原生预算 2048（(a′) 格只需前 1536 token） | batch 8 × 3072 在 ~2048 处溢出 WSL 共享内存，解码 21→1.8 steps/s，单 key 要 2 h |
| 2026-09-16 | n=100 校准完成（`results/calibration.csv`）：(a′) mul_4x3@1536 = 62%（direct 4%）、mul_3x3@1024 = 77%（direct 8%）进窗口；mul_4x4@1536 51%、mul_3x3@1536 87%、mul_4x2 direct 31% 出窗口；cups_n5_k8 / n6_k8 forced ≤ 33% 确认 cups 本地无窗口 | mul_5x4 因单调性省略，换成 mul_3x3 / mul_4x2 补 B=1024 备选 |
| 2026-09-16 | 缓存分配器碎片导致第二种显存溢出（reserved 7.85 GiB vs allocated 2.9 GiB）；每 batch `empty_cache` + `expandable_segments` 后稳定 92 s/batch | 见 §7 |
| 2026-09-17 | 训练起点 (a′)→原生思考自然结束（max_completion_length=3072，到顶强制收尾）；(a′)@1024 降为 eval 附加指标；L 边界改为生成开始→`<answer>`；M 由 32 题原生 L 中位数定；主难度 mul_3x3 | (a′) 下所有 L≡预算，长度惩罚在组内相对优势里是常数被抵消，长度梯度为零 |
| 2026-09-17 | 臂 0（n=50，native@3072）：acc 0.98，@1024 强制 0.78，L 中位 1820，触顶 18%；M=1897.5；臂 A λ=0.3 正式 run 启动（200 步，~14 h） | 预演 `<answer>` 操作化见 §2；smoke A/B 通过，臂 B 10 样本屏蔽命中 0 |
| 2026-09-17 | 臂 B 屏蔽改为按字符类型（含 L* 一律禁，无 N 课程）；格式加 `</think>`→`<answer>` 间只许空白 | N=500 频率屏蔽被去空格拼词 / 夹中文绕过 |
| 2026-09-17 | 正式实验全部上云；§9 重写：TRL 原生 + vLLM colocate + rollout_func 路径、第一小时验收清单、矩阵与估算、抢占恢复；新增 install_cloud.sh / sync_cloud.sh / cloud_8b.yaml / rollout.py / run_matrix.py / 06_diagnose.py | 本地 8 GB 只能 G=4，λ=0.3 长度信号过弱 |
| 2026-09-17 | 设计 v4 定稿（三轮外部评审）：奖励改为 10·[c(1−min(λL/M,0.9)) − 0.5v(1−c) − 0.05vc]、λ=0.5、无门控；mask 白名单去掉 answer 片段/special；策略 logp 在 mask 下重归一化；新臂 A″/B_seq/C_rand/D；准入 7 项；停止判据与操纵检查；端点/诊断/H1、H2 判定；矩阵 cups 19 + mul 2 | 见 §2 |
| 2026-09-17 | 违规分级：text_after_think / second_think_tag 为 hard（c:=0，r=−5），multi_answer / answer_too_long 为 soft；所有臂预填 `<think>\n`；eval 加 acc_clean | 本地 drill：被屏蔽模型第 0 步输出 `</think>` 逃逸，v4 原公式只扣 0.5 分 |
| 2026-09-18 | 云端准入定格：cups_n8_k20 / cap 5120 / M=3278（合并 n=1000）；删臂 D（准入第 7 条改为逐臂诊断 1）；mul 训练 run 砍掉（只报冻结数据）；矩阵 v5 顺序与裁剪；eval n=1000 每 50 步 | 见 §2、results/cloud_log.md |
| 2026-09-18 | cups 答案改为完整终态（exact match，答案区 ≤32 token，chance=1/(n!·2^n)）；加重写风格描述量；准入按新任务重跑（k12/k16/k20 + K/2，cap 5120） | 单点查询允许只回溯查询位置的捷径 |
| 2026-09-16 | `env.sh` 改用 `HF_XET_HIGH_PERFORMANCE=1`；`.gitignore` 加 `unsloth_compiled_cache/` 等运行产物，保留 `results/env_check.json` | huggingface_hub 1.31 弃用 hf_transfer |
