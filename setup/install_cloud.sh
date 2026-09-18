#!/usr/bin/env bash
# 云端（A100 80GB，Ubuntu 22.04，CUDA 12.x 镜像）安装。不需要 sudo。前提：uv 在 PATH（没有就先 curl -LsSf https://astral.sh/uv/install.sh | sh）。
# 关键约束：trl>=0.25（GRPOTrainer rollout_func）、vllm、torch 与驱动匹配。装完必须跑 00_check_env.py 且 rollout_func == true。
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/hf}"; mkdir -p "$HF_HOME"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv-cache}"   # 持久盘：stop 后重装走缓存
cd "$(dirname "$0")/.."

CUDA_MAJOR_MINOR=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" || echo "12.8")
echo ">> driver CUDA ${CUDA_MAJOR_MINOR}"
case "$CUDA_MAJOR_MINOR" in
  12.[0-5]*) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
  12.[6-7]*) TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
  *)         TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
esac
echo ">> torch index ${TORCH_INDEX}"

# 1) 固定一组互相兼容、且与驱动（CUDA 12.8）匹配的版本（2026-09-17 在 RunPod A100 / driver 570 上定）：
#    torch 2.8.0（PyPI 默认 cu128）← vllm 0.10.2（仍有 V0 引擎：VLLM_USE_V1=0 可用 per-request logits_processors）← trl 0.25.1（其 vllm extra 就是 ==0.10.2，有 rollout_func）
#    transformers<5（vllm 0.10.2 早于 transformers 5）。不装 unsloth：它会把 trl 降回 0.24（云端走 TRL 原生路径，不需要）。
#    不要用最新 vllm：0.29 的默认 wheel 拉 torch 2.12（CUDA 13），driver 570 跑不了；TRL 1.13 也只支持 vllm<=0.28。
rm -rf .venv && uv venv .venv --python 3.12 --seed && source .venv/bin/activate
uv pip install "vllm==0.10.2" "trl==0.25.1" "transformers>=4.56.2,<5" "peft" "accelerate" "datasets" "tensorboard" "pyyaml" "matplotlib" "pandas" \
  "pytest" "sentencepiece" "protobuf" "huggingface_hub" "zstandard" "safetensors"
# 3) torch 是否与驱动匹配
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "ok", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
python -c "import trl, vllm; print('trl', trl.__version__, 'vllm', vllm.__version__)"
uv pip freeze > setup/requirements.cloud.lock.txt

# 4) 环境检查：必须 rollout_func == true
COT_NO_UNSLOTH=${COT_NO_UNSLOTH:-0} python scripts/00_check_env.py --skip-model
python - <<'PY'
import inspect, sys
try:
    import unsloth  # noqa
except Exception:
    pass
from trl import GRPOTrainer
import trl.trainer.grpo_trainer as g
ok = "rollout_func" in inspect.getsource(g)
print("rollout_func available:", ok)
sys.exit(0 if ok else 1)
PY
echo "install_cloud done"
