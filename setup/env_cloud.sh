# source 这个文件进入云端环境（RunPod，全部在 /workspace 持久盘）
export HF_HOME=/workspace/hf
export UV_CACHE_DIR=/workspace/.uv-cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/cot-compress:${PYTHONPATH:-}   # vLLM V1 引擎子进程要能 import cot_compress.vllm_v1_proc
export COT_NO_UNSLOTH=1          # 云端不装 unsloth（TRL 原生路径）
export TMPDIR=/tmp            # 临时文件放容器盘
source /workspace/cot-compress/.venv/bin/activate
cd /workspace/cot-compress
