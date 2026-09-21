# 描述量脚本（只读归档，不进任何判定）

这些脚本只读 `runs/<run>/rollouts.jsonl.zst` 与 `runs/<run>/eval_step*.jsonl.zst`，不 import 训练路径、不写 pod、不改配置。
本来写在临时目录（本机 /tmp、pod /tmp），两处都会在重启 / pod stop 时清空，故入库保留，便于对臂 B 重算同样的量。

| 脚本 | 作用 | 产物 |
|---|---|---|
| `cot_scan.py` | 按训练步统计正确轨迹的长度、字母占比、叙述标记中位数 | 终端表 → `results/cloud_log.md` 的 CoT 抽样表 |
| `cot_scan2.py` | 同上但按每千字符归一（密度），另加"首个完整顺序之后的文本占比"（**该指标已判定不可用**，正则会命中开头抄写的约束串，见 cloud_log） | 终端表 |
| `group_deg.py` | GRPO 组内退化率：全对 / 全错 / **零奖励差**组占比、组内奖励差与 L 差 | 终端表 → `ops/notes_group_degeneracy.md` |
| `pair_dump.py` | 从两个 checkpoint 的 eval 归档里取"都答对"的同题对，按缩减百分位取 p25/p50/p75，导出并排文件 | `ops/cot_samples/` |
| `cot_dump.py` | 按步导出中位长度的正确轨迹首尾片段（人工读） | 终端 |

用法（在 pod 上，路径写死为 A1 的 run）：`cd /workspace/cot-compress && .venv/bin/python <脚本> [步号列表]`
例：`.venv/bin/python ops/analysis/cot_scan.py 1,25,50,75,100`
对臂 B 重算时需要改脚本里写死的 run 路径。
