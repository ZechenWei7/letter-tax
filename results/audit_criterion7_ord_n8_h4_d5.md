# 准入第 7 条人工审计：ord_n8_h4_d5，cap 10240（D13）

- 对象：冻结 Qwen3-4B 在 stop 划分 500 题上的原生轨迹（`results/calib/admission_ord_n8_h4_d5__native.jsonl`，准入 (8,4,5)@10240 那一批，native 0.608，M = 8599.5）。
- 审计范围：被 D12 定义下任一自动捷径检测器标中的全部 **66 条**（只 guess_verify 32、只 ir_copy_order 20、其余 14）。**未被任何检测器（D12）标中的 434 条没有读。**
- 读法：每条读命中点前后的上下文、开头与结尾；ir_copy_order 里 5 条答错 / 被截断的另读了后段约 2000 字。审计人：Claude（本仓库的编码代理），2026-09-19；用户审阅了前 52 条的分类表后定 D13。
- 分类：**模板 / 抄答案**（能指出复现的是什么算法或来源）｜**正常**（重述约束后正常推理）｜**瞎搭**（推理失败后摆候选）。

## 结论

| 分类 | 只 guess_verify 32 | 只 ir_copy_order 20 | 其余 14 | 合计 66 |
|---|---|---|---|---|
| 模板 / 抄答案 | 0 | 0 | 0 | **0** |
| 正常 | 28 | 13 | 9 | 50 |
| 瞎搭 | 4 | 7 | 5 | 16 |

**audit_template_rate = 0 / 500 = 0.000 < 5% → 第 7 条通过。**（分母 500；被标中的 66 条全读，0 条模板 / 抄答案；未标中的 434 条未读。）

检测器命中率（描述量，同一批 500 条）：r4 0.922｜D9 0.258｜D12 0.132｜D13 0.060。

两个系统性误触发（D13 已修）：
- guess_verify：32 条里 16 条是被恒等排列触发的——模型清点事件时写 `The events are 0,1,2,3,4,5,6,7. Yes`，这个清单本身是一个合法全排列、后面跟着验证词、又与紧邻的真候选不同。
- ir_copy_order：20 条里没有一条是"抄题面然后直接给顺序"；抄写出现在最终核对清单（10）、开头复述（6）、卡住后回读题面（4）。
- assign_enum 的 10 条（含重叠）全部由 `either way / both cases / try both` 口头语计数 ≥ d 触发，没有一条是对 2^d 个分配的系统枚举；perm_enum #347 由一句 `a permutation of all 8 events` 触发。（这两处 D13 未改，只记录。）

瞎搭的 16 条里 14 条答错，多数被截断；都是先认真推了 40–90%、卡住后才开始摆候选，没有一条从头就在猜。另：模型的逐条核对本身经常算错位置（#57、#365），"核对通过"不等于真通过。

N = 自然结束，F = 被强制收尾；✓ = 答对，✗ = 答错。# = 该题在 stop 划分 / jsonl 里的行号（0 起）。

## 只命中 guess_verify 的 32 条

| # | 结局 | 分类 | 触发点实际在干什么 |
|---|---|---|---|
| 16 | F✗ | 瞎搭 | 后段 `Alternatively… Hmm. How about…` 连试 4 个候选，每个都在第一条硬约束上失败 |
| 39 | N✓ | 正常 | 推出两个候选，并排比较 `Are both of these valid?` |
| 90 | F✓ | 正常 | 两个推出的序列并排复核 |
| 95 | N✗ | 正常（恒等排列） | 事件清点 |
| 97 | N✓ | 正常（恒等排列） | 事件清点；全程只有 2 个不同顺序 |
| 99 | F✗ | 正常 | 分情况时列出同一情形下的自由度（`could be A, or B, etc.`），没收敛 |
| 115 | F✓ | 正常 | 两个候选并排比较 |
| 143 | F✗ | 正常 | 推出 `4 can be placed anywhere before 6`，然后列出 3 种放法 |
| 163 | N✓ | 正常（恒等排列） | 事件清点 |
| 171 | N✗ | 瞎搭 | `I'm stuck. Let me try this sequence… Another try… Another sequence…`，共 11 个不同候选 |
| 172 | N✓ | 正常（恒等排列） | 事件清点，2 对都是 |
| 183 | N✗ | 正常 | 两个候选并排比较 |
| 190 | N✓ | 正常（恒等排列） | 事件清点 |
| 213 | F✗ | 正常 | 两个候选并排比较 |
| 215 | F✗ | 瞎搭（恒等排列触发） | 触发来自清点，但后段确实在 `Let me try this sequence` 连搭，共 9 个不同候选 |
| 220 | N✓ | 正常（恒等排列） | 事件清点 |
| 227 | F✗ | 正常 | 两个候选并排比较 |
| 246 | N✓ | 正常（恒等排列） | 事件清点 |
| 249 | F✗ | 瞎搭（恒等排列触发） | 同 215；11 个不同候选，结尾仍在 `Let me try: Order: …` |
| 251 | N✗ | 正常 | 两个候选并排比较 |
| 305 | N✓ | 正常 | 得到答案后查唯一性：`But wait, what about 6,4,1,2,0,7,5,3? Let's check` |
| 332 | F✗ | 正常 | 三个推出的候选并排比较 |
| 345 | N✓ | 正常（恒等排列） | 事件清点 |
| 348 | N✓ | 正常（恒等排列） | 事件清点 |
| 355 | N✓ | 正常（恒等排列） | 事件清点 |
| 374 | N✓ | 正常（恒等排列） | 事件清点 |
| 381 | N✓ | 正常（恒等排列） | 事件清点 |
| 441 | N✓ | 正常（恒等排列） | 事件清点 |
| 457 | N✓ | 正常 | 按析取项的两侧列出事件 4 的几种放法 |
| 472 | F✗ | 正常 | `2 and 3 can be placed anywhere else`，列出 3 种补全 |
| 480 | N✓ | 正常（恒等排列） | 事件清点 |
| 488 | F✗ | 瞎搭 | `Wait, maybe… Another try… this is impossible. What's the correct order?` |

## 只命中 ir_copy_order 的 20 条

| # | 结局 | 分类 | 抄写出现在哪 |
|---|---|---|---|
| 8 | N✓ | 正常 | 最终核对清单逐条引用 |
| 57 | N✗ | 瞎搭 | 卡住后回读题面；之后 `Let me try to build the sequence`，核对时位置读错（7 在位置 0、5 在位置 2，却判 `5 < 7 is true`），接受了错解 |
| 63 | N✓ | 正常 | 分情况推理中引用当前析取项 |
| 75 | N✓ | 正常 | 推理中引用，加最终清单 |
| 128 | N✗ | 瞎搭 | 卡住后回读题面；原话 `But this is just a guess. Let's check`，第 2 个猜的就被接受 |
| 146 | N✓ | 正常 | 开头核对题面 `that's five disjunctions. Let me count again` |
| 160 | N✓ | 正常 | 最终清单 |
| 185 | N✓ | 正常 | 开头复述，推理中引用 |
| 234 | N✓ | 正常 | 开头复述 |
| 269 | N✓ | 正常 | 分情况推理中引用 |
| 289 | N✓ | 瞎搭 | 卡住后回读题面；`I think I have to conclude there's a mistake in the problem… I'll go with…`，两个候选里挑了一个，碰对了 |
| 290 | N✓ | 正常 | 开头复述，加最终清单 |
| 292 | N✓ | 正常 | 最终清单，之后查唯一性 |
| 365 | F✗ | 瞎搭 | 最终清单核对的是错的候选，核对本身出错；后段一路在摆候选 |
| 366 | F✗ | 瞎搭 | 开头复述；后段 `Let me try: 5,2,7,6,…` 逐个修补候选，13 处 try / stuck |
| 369 | F✗ | 瞎搭 | 抄写全在最后 5%，对连试的候选逐个套清单；金标准顺序试过也核对通过，但提交的是 `Another possibility` 的另一个候选 |
| 378 | F✓ | 正常 | 开头复述，加 `approach this differently` 重列约束 |
| 425 | N✓ | 正常 | 开头核对题面 |
| 446 | N✓ | 正常 | 开头核对题面 |
| 485 | N✓ | 正常 | 开头复述 |

## 其余 14 条（assign_enum / perm_enum / 重叠）

| # | 命中 | 结局 | 分类 | 触发点实际在干什么 |
|---|---|---|---|---|
| 24 | assign_enum | F✓ | 正常 | 触发 = 5 处 `either way / both cases`（分情况推理的口头语），不是 2^d 分配枚举 |
| 166 | assign_enum | F✓ | 正常 | 同上（14 处）；两个候选并排比较后用硬约束链排除一个 |
| 293 | ir_copy_order + guess_verify | F✗ | 瞎搭 | 结尾在 `Option 5: 4,3,6,0,5,2,1,7` 逐个列候选套清单 |
| 318 | assign_enum | F✗ | 正常 | `either way` 口头语；分情况推理没收敛 |
| 328 | assign_enum | F✗ | 正常 | `try both possibilities` + 两个候选并排比较 |
| 343 | assign_enum | F✗ | 正常 | `both cases` 口头语；两种情形都推出矛盾后回查 |
| 347 | perm_enum | F✗ | 正常 | 触发 = 一句 `the order is a permutation of all 8 events`（关键词误触发） |
| 388 | perm_enum + guess_verify | F✗ | 瞎搭 | 后段连摆候选，9 个无推理切换的不同顺序 |
| 396 | assign_enum | N✓ | 正常 | `either way` 口头语 |
| 407 | assign_enum + guess_verify | F✗ | 瞎搭 | 两个候选并排比较后卡住，结尾 `Hmm. Let's try: 3, 7, 4, 0, 2,` |
| 427 | assign_enum | F✓ | 正常 | `in both cases` 口头语（11 处） |
| 454 | assign_enum | F✓ | 正常 | `either way / both cases`；中途 `I'm stuck` 但之后按约束链推出 |
| 466 | perm_enum | F✗ | 瞎搭 | 事件 4 的位置定不下来，连列 8 个放法 |
| 493 | assign_enum + guess_verify | F✓ | 正常 | `either way / both cases`（18 处）；Path 1 / Path 2 并排比较 |

## 典型摘录

- 恒等排列误触发（#381）：`So far, we have 4, 6, 0, 3, 2, 1, 7, 5. Wait, but that's all 8 events. Let me check: The events are 0,1,2,3,4,5,6,7. In the order above: 4, 6, 0, 3, 2, 1, 7, 5.`
- 两候选并排比较（#39）：`The first order was 2, 0, 4, 3, 5, 7, 6, 1. The second order is 2, 0, 3, 4, 5, 7, 6, 1. Are both of these valid? Let me check the second order again.`
- 列自由度（#143）：`So 4 must be before 6. So 4 can be placed anywhere before 6. So in this case, the order could be 0,4,7,5,6,1,2,3. Or 0,7,4,5,6,1,2,3. Or 0,7,5,4,6,1,2,3.`
- 瞎搭（#171）：`I'm stuck. Let me try to look for a possible sequence… Let me try this sequence: 5, 0, 4, 7, 3, 6, 2, 1. … Invalid. Another try: 5, 0, 4, 3, 7, 6, 2, 1. … No. Another sequence: 5, 0, 4, 2, 3, 7, 6, 1.`
- 瞎搭（#128）：`Maybe the sequence is 7, 1, 3, 4, 2, 0, 5, 6. … But this is just a guess.`
- ir_copy 的最终清单（#8）：`1. (3<5)|(1<2): 3 is sixth, 5 is seventh. So 3 <5 is true. So yes. 2. (5<6)|(2<0): 2 <0 is true. Yes. …`
- ir_copy 的回读题面（#57）：`Perhaps the fifth disjunction is not (3 < 2) or (0 < 2), but maybe I misread it. Let me check the problem again. The disjunctive constraints are: (3<4)|(2<5) (4<1)|(0<7) …`
- assign_enum 的口头语（#427）：`The fourth disjunction is (5 < 4) or (6 < 3). In both cases, 6 < 3 is true, so that's okay. So both orders are valid? But the problem states that there is exactly one.`
