#!/usr/bin/env bash
# 需要 sudo，由用户手动执行。
set -euo pipefail
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev build-essential git-lfs curl
git lfs install
