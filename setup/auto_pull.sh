#!/usr/bin/env bash
# 本地 watcher：云端每出现新的 eval 行（任一 run 的 eval.jsonl 变长）或 matrix_log 变化，就自动 pull results/ samples/ 与曲线回 cloud_pull/。
# 云机访问不到本地（NAT），所以由本地轮询发起。用法（本地）: setsid nohup bash setup/auto_pull.sh > cloud_pull/auto_pull.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source <(grep -E '^(CLOUD|SSH_KEY|REMOTE_DIR)=' setup/sync_cloud.sh)
HOST="${CLOUD%%:*}"; PORT="${CLOUD##*:}"
SSH="ssh -p $PORT -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
INTERVAL="${INTERVAL:-120}"; CURVES_EVERY="${CURVES_EVERY:-600}"; mkdir -p cloud_pull; last=""; last_curves=0
# D19：曲线文件（每步的 steps.jsonl 等，几 KB）每 CURVES_EVERY 秒固定拉一次，供 ops/dashboard.py 用；不含 *.zst / reward_log.jsonl / checkpoint。
# 大归档仍只在 eval.jsonl / matrix_log / 准入 json 变化时随下面的完整 pull 走。
pull_curves() {
  rsync -rltz --no-perms --no-owner --no-group -e "$SSH" --prune-empty-dirs \
    --include '*/' --include 'runs/*/steps.jsonl' --include 'runs/*/eval.jsonl' --include 'runs/*/designated_ckpt.json' --include 'runs/*/uninterpretable.json' --include 'runs/*/agg10.json' \
    --include 'results/matrix_log.md' --include 'results/matrix_halt.json' --include 'results/CHAIN_DONE' --include 'results/*.json' \
    --exclude '*' "$HOST:$REMOTE_DIR/" cloud_pull/
  mkdir -p cloud_pull/logs; rsync -rltz --no-perms --no-owner --no-group -e "$SSH" --include '*.rc' --include '*.start' --exclude '*' "$HOST:/workspace/logs/" cloud_pull/logs/   # 各 run 的退出码（看板据此标"崩溃"）
}
while true; do
  now=$(date +%s)
  if [ $((now - last_curves)) -ge "$CURVES_EVERY" ]; then pull_curves >/dev/null 2>&1 && last_curves=$now && echo "$(date -u +%H:%M:%S) curves pulled" || echo "$(date -u +%H:%M:%S) WARN: curves pull failed"; fi
  sig=$($SSH "$HOST" "cd $REMOTE_DIR 2>/dev/null && (cat runs/*/eval.jsonl results/matrix_log.md results/admission_*.json 2>/dev/null | md5sum)" 2>/dev/null | cut -d' ' -f1)
  if [ -z "$sig" ]; then echo "$(date -u +%H:%M:%S) WARN: cannot reach pod or project dir (volume down?)"
  elif [ "$sig" != "$last" ]; then
    echo "$(date -u +%H:%M:%S) change detected -> pull"; bash setup/sync_cloud.sh pull >/dev/null 2>&1 && last="$sig" && echo "  pulled ok" || echo "  PULL FAILED"
  fi
  sleep "$INTERVAL"
done
