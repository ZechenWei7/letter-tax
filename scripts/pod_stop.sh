#!/usr/bin/env bash
# 长任务收尾：Stop（不是 Terminate）当前 RunPod pod。/workspace（网络卷）保留，容器盘清空，GPU 计费停止。
# 用法（在 pod 上）:
#   bash scripts/pod_stop.sh                 # 停当前 pod（ID 取自 pod 自身环境 RUNPOD_POD_ID）
#   bash scripts/pod_stop.sh --dry-run       # 只核对：runpodctl 可用、API key 有效、pod 存在且 RUNNING；不停机
#   bash scripts/pod_stop.sh --force         # 即使还有 GPU 计算进程也停（默认拒绝，防止误杀在跑的任务）
#   bash scripts/pod_stop.sh --delay 120     # 先等 120 秒（给日志落盘 / 本地 pull 留时间）
#   bash scripts/pod_stop.sh --pod <id>      # 指定 pod id（从别的机器调用时）
# 典型收尾:  python scripts/run_matrix.py ... ; bash scripts/pod_stop.sh --delay 300
# 前提：runpodctl 装在 /workspace/bin（持久），API key 在 /workspace/.runpod/config.toml（~/.runpod 是指向它的软链；容器盘重建后本脚本会重建软链）。
# API key 不进仓库、不进日志。
set -uo pipefail
DRY=0; FORCE=0; DELAY=0; POD=""
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1 ;; --force) FORCE=1 ;; --delay) DELAY="$2"; shift ;; --pod) POD="$2"; shift ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;; esac; shift; done

[ -d /workspace/.runpod ] && [ ! -e "$HOME/.runpod" ] && ln -s /workspace/.runpod "$HOME/.runpod"      # 容器盘重建后恢复配置软链
CTL="$(command -v /workspace/bin/runpodctl || command -v runpodctl || true)"
[ -n "$CTL" ] || { echo "[pod_stop] runpodctl not found (expected /workspace/bin/runpodctl)" >&2; exit 3; }
if [ -z "$POD" ]; then POD="${RUNPOD_POD_ID:-}"; fi
if [ -z "$POD" ] && [ -r /proc/1/environ ]; then POD="$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p' | head -1)"; fi
[ -n "$POD" ] || { echo "[pod_stop] cannot determine pod id (pass --pod <id>)" >&2; exit 4; }

INFO="$(timeout 60 "$CTL" pod get "$POD" < /dev/null 2>&1)" || { echo "[pod_stop] 'runpodctl pod get $POD' failed (API key / network?):" >&2; echo "$INFO" | head -3 >&2; exit 5; }
STATUS="$(printf '%s' "$INFO" | sed -n 's/.*"desiredStatus": *"\([A-Z]*\)".*/\1/p' | head -1)"
echo "[pod_stop] pod=$POD status=${STATUS:-unknown} ctl=$CTL"

if [ "$FORCE" -ne 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  BUSY="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -c . || true)"
  if [ "${BUSY:-0}" -gt 0 ]; then echo "[pod_stop] refusing: $BUSY GPU compute process(es) still running (use --force to stop anyway)" >&2; exit 6; fi
fi
if [ "$DRY" -eq 1 ]; then echo "[pod_stop] dry-run OK: would run '$CTL pod stop $POD'"; exit 0; fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ "$DELAY" -gt 0 ] 2>/dev/null && { echo "[pod_stop] waiting ${DELAY}s before stop"; sleep "$DELAY"; }
echo "- $(date -u '+%Y-%m-%d %H:%M UTC') pod_stop.sh: stopping pod $POD (Stop, not Terminate)" >> "$ROOT/results/cloud_log.md" 2>/dev/null || true
sync
exec timeout 120 "$CTL" pod stop "$POD" < /dev/null
