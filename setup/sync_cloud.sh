#!/usr/bin/env bash
# 本地 ⇄ 云端同步。用法:
#   CLOUD=user@host[:port] bash setup/sync_cloud.sh push     # 本地 → 云端：只传代码和配置
#   CLOUD=user@host[:port] bash setup/sync_cloud.sh pull     # 云端 → 本地：results/ samples/ 与各 run 的曲线文件，不传 checkpoint
# 可选 REMOTE_DIR（默认 ~/cot-compress）、LOCAL_PULL_DIR（默认 ./cloud_pull，避免覆盖本地 results/）。
set -euo pipefail
# 默认目标：RunPod A100 SXM 80GB（2026-09-19 第二次迁移后的 pod 2dy8hd8l24yo24；/workspace 网络卷随迁移完整拷贝）。容器盘 stop 时清空 → 一切放 /workspace。
# push 会带上 .git（pod 上可核对 HEAD）与 results/ 里入库的白名单审计文件；results/ 其余内容不推（不覆盖云端结果）。
CLOUD="${CLOUD:-root@195.26.233.65:39125}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
HOST="${CLOUD%%:*}"; PORT="${CLOUD##*:}"; [ "$PORT" = "$CLOUD" ] && PORT=22
REMOTE_DIR="${REMOTE_DIR:-/workspace/cot-compress}"
LOCAL_PULL_DIR="${LOCAL_PULL_DIR:-cloud_pull}"
SSH="ssh -p $PORT -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
cd "$(dirname "$0")/.."
case "${1:-}" in
  push)
    rsync -rlvz --checksum --no-perms --no-owner --no-group --no-times -e "$SSH" --delete \
      --exclude .venv/ --exclude runs/ --exclude samples/ --exclude unsloth_compiled_cache/ --exclude __pycache__/ \
      --exclude '*.pyc' --exclude '*.log' --include 'results/.keep' --include 'results/whitelist_extra_banned.json' --include 'results/whitelist_audit.md' --include 'results/audit_criterion7_*.md' --include 'results/admission_ord_n8_h4_d5.json' \
      --exclude 'results/*' --exclude cloud_pull/ --exclude archive/ --exclude .pytest_cache/ \
      ./ "$HOST:$REMOTE_DIR/"
    echo "pushed code+configs to $HOST:$REMOTE_DIR" ;;
  pull)
    mkdir -p "$LOCAL_PULL_DIR"
    rsync -avz -e "$SSH" --prune-empty-dirs \
      --include '*/' --include 'results/***' --include 'samples/***' \
      --include 'runs/*/eval.jsonl' --include 'runs/*/steps.jsonl' --include 'runs/*/reward_log.jsonl' \
      --include 'runs/*/ppl.jsonl' --include 'runs/*/diagnostics.json' --include 'runs/*/diagnostics_rows.jsonl' --include 'runs/*/tb/***' --include 'runs/*/README.md' \
      --include 'runs/*/rollouts.jsonl.zst' --include 'runs/*/eval_step*.jsonl.zst' --include 'runs/*/diag_*.jsonl.zst' \
      --exclude 'runs/*/checkpoint-*' --exclude 'runs/*/final' --exclude '*' \
      "$HOST:$REMOTE_DIR/" "$LOCAL_PULL_DIR/"
    echo "pulled results/samples/curves into $LOCAL_PULL_DIR" ;;
  ssh) shift; exec $SSH "$HOST" "$@" ;;
  *) echo "usage: [CLOUD=user@host[:port]] $0 push|pull|ssh [cmd]"; exit 1 ;;
esac
