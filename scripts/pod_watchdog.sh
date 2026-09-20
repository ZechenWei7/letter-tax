#!/usr/bin/env bash
# 守夜狗（无人值守时的兜底停机）。每 PERIOD 秒检查一次，命中任一条即停机并退出：
#   (a) 连续 IDLE_MIN 分钟没有 GPU 计算进程        → 空转兜底（链已结束 / 进程死了 / pod_stop 没生效）
#   (b) /workspace/logs/DECISION_PENDING 存在且     → 决策倒计时到点、用户未回复
#       mtime 超过 DEADLINE_MIN 分钟
# 停机 = scripts/pod_stop.sh --force（Stop 不是 Terminate；/workspace 网络卷保留；训练每 25 步有 checkpoint，
# run_matrix 下次自动 --resume）。用户回复后取消倒计时：rm /workspace/logs/DECISION_PENDING。
# 用法（pod 上）: setsid nohup bash scripts/pod_watchdog.sh > /workspace/logs/watchdog.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
IDLE_MIN="${IDLE_MIN:-90}"; DEADLINE_MIN="${DEADLINE_MIN:-90}"; PERIOD="${PERIOD:-300}"
MARK=/workspace/logs/DECISION_PENDING
idle_since=0
log() { echo "$(date -u '+%F %T') $*"; }
stop_now() {                                              # $1 = 原因（进 cloud_log）
  log "$1 -> pod_stop --force"
  { echo "- $(date -u '+%Y-%m-%d %H:%M UTC') pod_watchdog: $1 → pod_stop --force"
    [ -f "$MARK" ] && sed 's/^/    /' "$MARK"; } >> results/cloud_log.md 2>/dev/null || true
  sync; bash scripts/pod_stop.sh --force; exit 0
}
log "watchdog start: idle >= ${IDLE_MIN} min 或 DECISION_PENDING 超过 ${DEADLINE_MIN} min → 停机（每 ${PERIOD}s 检查）"
while true; do
  now=$(date +%s)
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "${n:-0}" -gt 0 ]; then idle_since=0; elif [ "$idle_since" -eq 0 ]; then idle_since=$now; fi
  if [ "$idle_since" -ne 0 ] && [ $(( (now - idle_since) / 60 )) -ge "$IDLE_MIN" ]; then
    stop_now "GPU 连续空转 $(( (now - idle_since) / 60 )) min（阈值 ${IDLE_MIN}）"
  fi
  if [ -f "$MARK" ]; then
    age=$(( (now - $(stat -c %Y "$MARK")) / 60 ))
    [ "$age" -ge "$DEADLINE_MIN" ] && stop_now "决策倒计时 ${age} min 到点（阈值 ${DEADLINE_MIN}），用户未回复"
  fi
  sleep "$PERIOD"
done
