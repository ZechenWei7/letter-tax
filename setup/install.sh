#!/usr/bin/env bash
# 不需要 sudo。前提：setup/apt.sh 已执行，uv 已在 ~/.local/bin。
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$HOME/hf"
mkdir -p "$HF_HOME"
cd "$HOME/cot-compress"

uv venv .venv --python 3.12 --seed
source .venv/bin/activate

# 1) torch 只从 cu128 index 装（sm_120 需要）
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# 2) 其余依赖从 PyPI 装；unsloth 会带上 trl / transformers / peft / unsloth_zoo
uv pip install "unsloth" "unsloth_zoo" "trl" "transformers>=5" "peft" "accelerate" \
  "bitsandbytes>=0.45.3" "datasets" "tensorboard" "pyyaml" "matplotlib" "pandas" \
  "pytest" "sentencepiece" "protobuf" "huggingface_hub[hf_transfer]"

# 3) 如果第 2 步把 torch 换成了非 cu128 版本，重装
if ! python setup/check_torch_cuda.py; then
  echo ">> torch was replaced by a non-cu128 build, reinstalling from cu128 index"
  uv pip install --index-url https://download.pytorch.org/whl/cu128 --reinstall torch torchvision
fi

uv pip freeze > setup/requirements.lock.txt
echo "install done"
