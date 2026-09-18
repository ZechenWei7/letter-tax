#!/usr/bin/env bash
# 把 Windows 暂存目录同步到 ~/cot-compress，并去掉 CR
set -euo pipefail
SRC="$1"; DST="$HOME/cot-compress"
mkdir -p "$DST"
cp -r "$SRC"/. "$DST"/
find "$DST" -path "$DST/.venv" -prune -o -type f -print | grep -E '\.(py|sh|md|toml|yaml|txt)$|\.gitignore$' | xargs -r sed -i 's/\r$//'
chmod +x "$DST"/setup/*.sh
echo "synced to $DST"
