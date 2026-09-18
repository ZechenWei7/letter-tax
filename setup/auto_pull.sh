#!/usr/bin/env bash
# 本地 watcher：云端每出现新的 eval 行（任一 run 的 eval.jsonl 变长）或 matrix_log 变化，就自动 pull results/ samples/ 与曲线回 cloud_pull/。
# 云机访问不到本地（NAT），所以由本地轮询发起。用法（本地）: setsid nohup bash setup/auto_pull.sh > cloud_pull/auto_pull.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source <(grep -E '^(CLOUD|SSH_KEY|REMOTE_DIR)=' setup/sync_cloud.sh)
HOST="${CLOUD%%:*}"; PORT="${CLOUD##*:}"
SSH="ssh -p $PORT -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
INTERVAL="${INTERVAL:-120}"; mkdir -p cloud_pull; last=""
while true; do
  sig=$($SSH "$HOST" "cd $REMOTE_DIR 2>/dev/null && (cat runs/*/eval.jsonl results/matrix_log.md results/admission_*.json 2>/dev/null | md5sum)" 2>/dev/null | cut -d' ' -f1)
  if [ -z "$sig" ]; then echo "$(date -u +%H:%M:%S) WARN: cannot reach pod or project dir (volume down?)"
  elif [ "$sig" != "$last" ]; then
    echo "$(date -u +%H:%M:%S) change detected -> pull"; bash setup/sync_cloud.sh pull >/dev/null 2>&1 && last="$sig" && echo "  pulled ok" || echo "  PULL FAILED"
  fi
  sleep "$INTERVAL"
done
