# 第二阶段：推导渲染模板与连接词映射（冻结于计算长度阶梯之前）

**不在 OSF 注册的 stage-1 协议之内。** 代码 `cot_compress/derivation.py`，核对脚本 `scripts/phase2_render_check.py`（结果 `results/phase2_render_check.json`）。
stage-1 的代码（`tasks/ordering.py` 等）未改动。

## 求解过程（被渲染的对象）

stage-1 的同一求解器：传递闭包 + 单元规则传播；failed-disjunct 单边试探；规范路径"最低编号未决析取优先、先第一边"。
`derive()` 逐步记录这个过程，并对每题断言：根传播的终态与 `tasks.ordering.probe` 相同，决策序列与 `canonical_path` 相同，最终全序等于 σ。
规范路径里先试的一边若是假的：直接冲突 → 记录冲突；未直接冲突（S≥2 的 "refuted_deeper"）→ 记录一棵完整的反驳子树（在该假设下按最低编号再分，两边都导出冲突），使每条推导都是一个完整的证明。

| 步骤 | 含义 | N2 模板 |
|---|---|---|
| force | 链证明 b 在 a 之前 → 析取的 a<b 一边不成立 → 另一边 c<d 被迫 | ` since b<…<a not a<b forces c<d` |
| sat | 链证明某边已被蕴含 → 该析取已满足 | ` since a<…<b holds (a<b)|(c<d)` |
| assume | 决策 | ` suppose a<b` |
| cycle | 冲突：闭包出环 | ` contradiction a<b<…<a` |
| both | 冲突：析取两边都被否定 | ` contradiction (a<b)|(c<d) since b<…<a since d<…<c` |
| back | 冲突后弹出最近的假设，其析取另一边被迫 | ` so c<d` |
| order | 全序 | ` order 3 0 7 1 4 6 5 2` |

每条链的每一环都是当时可用的显式约束（硬约束或已定析取边）；`verify()` 独立重放、逐环检查。一步一行，每行以空格加连接词开头。
N1 用完整英文句子表达同一步骤序列（"Since 6 comes before 3 and 3 comes before 1, 7 cannot come before 6, so the other option must hold: 7 comes before 5."），模板见代码。

## 连接词映射（Qwen3-4B 分词器，均为带前导空格的单 token）

| 连接词 | N2 | N2s（乱词对照） | N3（纯符号） | N3 在 B 白名单 | N3 在 token_freq 表中的频次 |
|---|---|---|---|---|---|
| since | ` since` 2474 | ` table` 1965 | ` :` 549 | ✓ | 58,452 |
| not | ` not` 537 | ` green` 6176 | ` !` 753 | ✓ | 5,857 |
| forces | ` forces` 8437 | ` river` 14796 | ` =>` 589 | ✓ | 不在表内（表只含高频 token） |
| holds | ` holds` 9982 | ` paper` 5567 | ` ;` 2587 | ✓ | 62,556 |
| suppose | ` suppose` 22477 | ` window` 3241 | ` ?` 937 | ✓ | 3,865 |
| contradiction | ` contradiction` 49759 | ` music` 4627 | ` #` 671 | ✓ | 1,706 |
| so | ` so` 773 | ` garden` 13551 | ` =` 284 | ✓ | 473,873 |
| order | ` order` 1973 | ` yellow` 13753 | ` @` 569 | ✓ | 385,855 |

N3 的选择：从白名单里带前导空格、纯 ASCII 标点、按 `results/token_freq.json` 频次排序的候选中挑（排除模板本身用到的 `< | ( )` 与数字）；用户点名的 `=> ! # ;` 全部采用；` =>` 是单 token 且在白名单内，但不在频次表的高频列表里（如需严格"最常见"，可换成 ` ,`/` .`/` \"`/` '` 等，未换）。
序数：N2/N2s/N3 全部不用序数词——析取按其两边原样复述，步骤按行排列；故三者除连接词外逐 token 相同。

## 核对结果（全部 5,500 题：stage-1 的 train 2000 / stop 500 / report 500 / direct 2000 + 新评估集 500）

- 四个版本往返（render → parse → 与原步骤完全相等）：**5500 / 5500**，每个版本。
- 独立重放（verify）：**5500 / 5500**，每个版本。
- N2 / N2s / N3 逐 token 对齐（等长，只在连接词位置不同，且该位置正是对应连接词的 id）：**5500 / 5500**。每题连接词位置中位 18（8–60）。
- N3 无字母 token、全部 token 在 B 白名单（8351）内：**5500 / 5500**。
- 单边试探在传播不动点后触发的次数：**0**（与推断一致：加边 a<b 成环 ⇔ 闭包已蕴含 b<a ⇔ 单元规则已处理）。
- 步骤类型合计：force 22492 · assume 9484 · back 5149 · both 4393 · sat 2031 · cycle 756 · order 5500。
- 同时打开的假设数上限：1 → 3714 题，2 → 1452，3 → 304，4 → 30（规范路径按最低编号分支，不是最优分支，故嵌套可深于 S；代码上限 4，无一题超限）。

## 新评估集（`data/ordering/ord_n8_h4_d5_p2eval_seed1.json`，`scripts/phase2_build_eval.py`）

seed = 1、划分名 `p2eval`、500 题，同一生成器与分布。与 stage-1 全部 5000 题：同构类规范形重叠 0，题面重叠 0（两层断言）。S 分布 1:471 / 2:29（stage-1 stop 为 473 / 27）；#LE(hard) 中位 1680、反链 5、哈密顿路径 22.5、存活析取 5。

## 示例（新评估集第 13 题；按规则取第一个同时含 force / sat / assume / back / 冲突、且不超过 16 步的题）

```
ID ord_n8_h4_d5/p2eval/1/13
PROMPT
n=8
hard: 2<4 1<7 6<3 4<0
disj: (3<1)|(3<2) (7<6)|(7<5) (4<5)|(1<5) (7<6)|(6<1) (5<1)|(5<7)
reason inside <think>, then give the order
answer: 8 digits, first to last

--- N1 ---
Suppose that 3 comes before 1.
Since 6 comes before 3, 3 comes before 1 and 1 comes before 7, 7 cannot come before 6, so the other option must hold: 7 comes before 5.
Since 6 comes before 3 and 3 comes before 1, the constraint that 7 comes before 6 or 6 comes before 1 is already satisfied.
Since 1 comes before 7 and 7 comes before 5, the constraint that 4 comes before 5 or 1 comes before 5 is already satisfied.
But then the constraint that 5 comes before 1 or 5 comes before 7 cannot be satisfied, because 1 comes before 7 and 7 comes before 5, and also 7 comes before 5. This is a contradiction.
So that assumption was wrong, and the other option must hold: 3 comes before 2.
Suppose that 7 comes before 6.
Since 7 comes before 6, the constraint that 7 comes before 6 or 6 comes before 1 is already satisfied.
Suppose that 4 comes before 5.
But then the constraint that 5 comes before 1 or 5 comes before 7 cannot be satisfied, because 1 comes before 7, 7 comes before 6, 6 comes before 3, 3 comes before 2, 2 comes before 4 and 4 comes before 5, and also 7 comes before 6, 6 comes before 3, 3 comes before 2, 2 comes before 4 and 4 comes before 5. This is a contradiction.
So that assumption was wrong, and the other option must hold: 1 comes before 5.
Since 1 comes before 5, 5 cannot come before 1, so the other option must hold: 5 comes before 7.
Therefore the order, from first to last, is 1 5 7 6 3 2 4 0.

--- N2 ---
 suppose 3<1
 since 6<3<1<7 not 7<6 forces 7<5
 since 6<3<1 holds (7<6)|(6<1)
 since 1<7<5 holds (4<5)|(1<5)
 contradiction (5<1)|(5<7) since 1<7<5 since 7<5
 so 3<2
 suppose 7<6
 since 7<6 holds (7<6)|(6<1)
 suppose 4<5
 contradiction (5<1)|(5<7) since 1<7<6<3<2<4<5 since 7<6<3<2<4<5
 so 1<5
 since 1<5 not 5<1 forces 5<7
 order 1 5 7 6 3 2 4 0

--- N2s ---
 window 3<1
 table 6<3<1<7 green 7<6 river 7<5
 table 6<3<1 paper (7<6)|(6<1)
 table 1<7<5 paper (4<5)|(1<5)
 music (5<1)|(5<7) table 1<7<5 table 7<5
 garden 3<2
 window 7<6
 table 7<6 paper (7<6)|(6<1)
 window 4<5
 music (5<1)|(5<7) table 1<7<6<3<2<4<5 table 7<6<3<2<4<5
 garden 1<5
 table 1<5 green 5<1 river 5<7
 yellow 1 5 7 6 3 2 4 0

--- N3 ---
 ? 3<1
 : 6<3<1<7 ! 7<6 => 7<5
 : 6<3<1 ; (7<6)|(6<1)
 : 1<7<5 ; (4<5)|(1<5)
 # (5<1)|(5<7) : 1<7<5 : 7<5
 = 3<2
 ? 7<6
 : 7<6 ; (7<6)|(6<1)
 ? 4<5
 # (5<1)|(5<7) : 1<7<6<3<2<4<5 : 7<6<3<2<4<5
 = 1<5
 : 1<5 ! 5<1 => 5<7
 @ 1 5 7 6 3 2 4 0
```
