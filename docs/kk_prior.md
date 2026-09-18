# 先行工作：K&K 数据集（Xie et al. 2024）与 Logic-RL（Xie et al. 2025）的生成器与难度定义

目的：v7 主任务换成演绎类任务前，确认我们的生成器与他们**不撞车、不共享模板**。
来源：arXiv 2410.23123 全文附录 C（PDF 抽取）+ 公开仓库 `AlphaPav/mem-kk-logic` 的 `data_prep/lib_kk.py`（逐行核对）；arXiv 2502.14768 全文 §2。

## 1. K&K 数据集（"On Memorization of LLMs in Logical Reasoning"，Xie, Huang, Bhardwaj, Tan, Liu, Guo, Song, Zheng, Zhou, 2024）

### 1.1 抽象表示（appendix C.1）
- N 人，编号 `0..N-1`；每人一条陈述，为 Python tuple 树：
  - 叶：`('telling-truth', i)` / `('lying', i)`
  - 复合：`('not', s)`, `('and', s1, s2, …)`, `('or', s1, s2, …)`, `('->', s1, s2)`, `('<=>', s1, s2)`
- 语义：赋值 B₁..B_N ∈ {knight, knave}，合法当且仅当 ⋀ᵢ (Bᵢ ⇔ Sᵢ)。

### 1.2 生成器（C.2 + `lib_kk.py: KKProblemSampler._sample_statement`）
- 难度规格 = **(N, W, D)**：人数 N、每条陈述的最大宽度 W、最大深度 D。发布集固定 **W=2, D=2**，N∈{2..8}。
- 每人依次采样一棵陈述树，采样规则（实测源码，比论文更细）：
  - 掷 `dice ∈ {0..5}` 均匀：0 → 叶；1 → `not`；2 → `and`；3 → `or`；4 → `->`；5 → `<=>`。深度耗尽（depth_constraint==1）强制叶。
  - 叶：均匀选 telling-truth/lying，均匀选人 i∈[0,N)；**跳过 "自称说谎"**（`('lying', self)`，平凡不可满足）。允许 `('telling-truth', self)`。
  - `and`/`or` 子句数 ∈ [2, W]（W=2 时恒为 2）；`->`/`<=>` 恒 2 个子句；子句**去重**（同一父节点下不重复）。
  - 因 D=2、叶只能是"X 是骑士/骗子"，`not` 只作用在叶上（NL 格式化器只支持深度 ≤2，代码有 TODO 注明）。
- **唯一解筛选**：暴力枚举 2^N 赋值，`sample_valid_problems` 只保留恰有 1 个解的题；并用 `unique_statements` 集合去重整题；最多重试 `max_retry`（默认 1000；论文 perturber 用 2000）。N=8 时约 30% 随机题唯一解。
- 发布规模：每个 N 训练 1000/测试 100/验证 50（N=2 只 200 训练）。

### 1.3 难度定义
- 唯一显式难度轴 = **N（人数）**，W、D 固定。论文的"难度级别"= N∈{2..8}，训练/评测按 N 分桶（"2-ppl task … 8-ppl task"）。
- 没有按解题步数、回溯次数、约束传播深度等定义的难度；难度是构造参数，不是求解代价。
- 同一 N 下题目的求解难度实际差异很大（如例题 5 人题 Reasoner 回溯 2 次即解），论文未分层。

### 1.4 扰动（C.4, C.7）
- 数学级（改题、改答案，仍要求唯一解且解不同）：**statement**（换整棵树）、**leaf**（改一个叶：翻 truth/lying 或换人）。
- 语言级（题不变、答案不变）：uncommon name（50 个冷门名）、random role pair（saint/sinner, hero/villain, angel/devil, altruist/egoist, sage/fool, pioneer/laggard）、reorder statements、flip role（knave 说真话）。

### 1.5 自然语言模板（C.5 + `lib_kk.py`）——**撞车检查的关键**
- 前缀（固定）：`A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet N inhabitants: A, B, …, and Z.`
- 每条陈述随机取 **18 个说话模板之一**，如 `{name} said that {content}.` / `{name} stated, "{content}".` / `According to {name}, "{content}".` / `In {name}'s words: "{content}".` / `"{content}," {name} declared.` / `As {name} put it, "{content}".` …（全部是"某人说 X"的英语引述句式）。
- 内容格式化（`format_statement`）：叶 → `X is a knight` / `X is a knave`；not → `X is not a knight`；and/or → 用 ` and `/` or ` 连接；`->` → `If A then B`；`<=>` → `A if and only if B`。
- 后缀（固定）：`So who is a knight and who is a knave?`
- 名字：52 个常见英文名（Emma, Liam, Olivia, Noah, …）。
- 答案文本：`A is a knight, B is a knave, and C is a knight.`
- Reasoner/NL-Reasoner 生成的合成 CoT：顺序假设 + 回溯（"Assume X is a knight. No contradiction … X cannot be a knave, because this would contradict …"），用于他们的 SFT 实验，不是我们关心的部分。

## 2. Logic-RL（"Unleashing LLM Reasoning with Rule-Based Reinforcement Learning"，Xie, Gao, Lin, Xia, Chen, Tang, Chen, Xie, Kang, Ou, Wang, Yi, Qin, Zhang, Zhou, Chen, Huang, Chen, 2025）
- **数据完全复用 K&K 数据集**（引用 [17] 即上文），未自建生成器。训练集：K&K 的 N=3..7，<5000 题；OOD 测试 N=8（和 N=2）。
- 论文口头描述难度为"人数 2–8 × 布尔算子组合 1–4 种"，实际就是 K&K 的 (N, W=2, D=2) 发布集；"算子组合数"不是他们控制的独立轴。
- 课程学习：按 N=3→7 逐级各一 epoch vs 混合，结论差异不显著（RQ6）。
- 记忆度量：直接沿用 K&K 的 LiMem = Acc·(1−CR)，扰动只用 statement + reorder 两种。
- 格式/奖励（与我们相关的部分）：`<think></think><answer></answer>`，系统提示要求 `<answer> (1) Zoey is a knight, (2) … </answer>`；`<think>` 直接放在 prompt 末尾（与我们的 `<think>\n` 预填相同做法）；格式分 ±1（每个标签恰出现一次且顺序正确、think 内需有"真实推理"、answer 可抽取），答案分 +2/−1.5/−2；REINFORCE++，lr 4e-7，温度 0.7，3600 步，Qwen2.5-7B-Instruct-1M。
- 他们列出的奖励黑客行为（跳过 think、在 answer 里推理、反复猜答案、在 answer 后重开 think、复述题目/"thinking process here"）与我们 v4 的 hard/soft 违规集合有重叠——这是格式层面的通用现象，不构成任务/模板撞车。

## 3. 与我们的生成器不撞车 / 不共享模板的检查清单（供 v7 规格核对）
下列任一项与 K&K 不同即可保证模板层面独立；结构层面若也是"N 人 + 真/假陈述 + 唯一赋值"的 SAT 变体，则需要在论文中显式说明与 K&K 的关系。

| 维度 | K&K / Logic-RL | 我们的 v7 生成器应满足（待规格确认） |
|---|---|---|
| 逻辑骨架 | 每人一条关于他人身份的布尔陈述，(Bᵢ⇔Sᵢ) 的 SAT，2^N 枚举 | 若也是 knights/knaves 骨架 → 直接撞车，需换题型（如约束传播/表格推理/关系链演绎） |
| 难度轴 | 只有 N；W=D=2 固定 | 建议用**求解代价**（最小推导步数、必要的分支/回溯数）作难度，而非仅实例规模 |
| 陈述文法 | 6 种节点类型均匀采样，深度 ≤2，宽度 2，禁 "自称说谎" | 不要复用 `('lying', i)` 形式的 tuple 文法或均匀掷 6 面骰的采样 |
| 唯一解 | 暴力 2^N 计数 = 1，去重 + 重试 | 唯一解检查可以类似（无法避免），但答案空间 / 校验方式应不同 |
| NL 前缀 | "A very special island is inhabited only by knights and knaves…" | 不得出现 island/knights/knaves/inhabitants 措辞 |
| 说话模板 | 18 个 "{name} said/stated/remarked … {content}" | 若题目里有引述，不要用这 18 句中的任何一句 |
| 内容措辞 | "X is a knight", "If A then B", "A if and only if B", "X is not a knight" | 避免同一 4 种连接词措辞的组合；或者根本不使用引述式陈述 |
| 名字 | 52 常见名 + 50 冷门名 | 用非人名实体（编号对象、符号）可完全避开 |
| 答案格式 | "(1) Zoey is a knight, (2) …" / "A is a knight, B is a knave, and …" | 与我们 32-token 答案区、exact-match 的约定另行定义 |
| 扰动 | statement/leaf + 4 种语言级 | 若 v7 用扰动做诊断，命名和定义要区别开（如称"约束替换"） |

结论：K&K 的生成器本质是"人数 N 固定、深度 2 的随机布尔陈述树 + 2^N 唯一解筛选"，模板是 18 句英文引述 + 固定岛屿前缀。只要 v7 不用 knights/knaves 世界观、不用"人说话"的引述模板、难度不用单一 N 轴，就不撞车也不共享模板。
