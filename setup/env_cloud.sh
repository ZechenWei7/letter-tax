# source 这个文件进入云端环境（RunPod，全部在 /workspace 持久盘）
export HF_HOME=/workspace/hf
export UV_CACHE_DIR=/workspace/.uv-cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/cot-compress:${PYTHONPATH:-}   # vLLM V1 引擎子进程要能 import cot_compress.vllm_v1_proc
export COT_NO_UNSLOTH=1          # 云端不装 unsloth（TRL 原生路径）
export TMPDIR=/tmp            # 临时文件放容器盘
# D20：编译缓存放持久盘（容器盘每次 pod stop 都会清空 → 开机后第一个进程全冷编译）。vLLM torch.compile / inductor / triton 三处。
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor
export TRITON_CACHE_DIR=/workspace/.cache/triton
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
source /workspace/cot-compress/.venv/bin/activate
cd /workspace/cot-compress
