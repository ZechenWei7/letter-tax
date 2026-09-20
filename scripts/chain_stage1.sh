#!/usr/bin/env bash
# D14：阶段 1 自动链（pod 上 setsid nohup 启动）：A1 → 操纵门 → B_warm-SFT(A1) 评估 → B1（≤200 步 warm 判定）→ 各 run 结束后内容诊断 → 写 CHAIN_DONE 标记 → pod_stop.sh --delay 300。
# 任何门失败 / OOM / 非零退出 / 残留检查触发：run_matrix --strict 立刻 exit 4，本脚本照样写标记并停机（不自动重调，不进入矩阵其余部分）。
# 本地 pull / 提交由本地 watcher 在停机延时内完成；没赶上也无妨，结果在 /workspace 网络卷上。
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
source setup/env_cloud.sh
mkdir -p /workspace/logs; rm -f results/CHAIN_DONE
# D20 预热：开机后先跑一个可丢弃的 1 步 dry-run（与正式 run 相同的引擎配置 → 相同的编译缓存键），退出码不计；正式 run 因而总是热缓存启动。不是实验数据。
echo start > /workspace/logs/warmup.start; rm -f /workspace/logs/warmup.rc
python scripts/train.py --config configs/cloud_4b.yaml --arm A --dry-run --tag _warmup --set task.key=ord_n8_h4_d5 train.dry_run_steps=1 > /workspace/logs/warmup.log 2>&1
echo "rc=$?" > /workspace/logs/warmup.rc
python scripts/run_matrix.py --matrix configs/matrix_stage1.yaml --strict >> /workspace/logs/chain_stage1.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) rc=$rc" > results/CHAIN_DONE
echo "- $(date -u '+%Y-%m-%d %H:%M') UTC chain_stage1 finished rc=$rc (4 = strict halt; see results/matrix_halt.json / matrix_log.md)" >> results/cloud_log.md
bash scripts/pod_stop.sh --delay 300 --force >> /workspace/logs/chain_stage1.log 2>&1
