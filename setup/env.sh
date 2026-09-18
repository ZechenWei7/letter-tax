# source 这个文件进入环境
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$HOME/hf"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
# expandable_segments 在 WSL 不受支持（unsloth_zoo 导入时会剥掉）；分配器调优改在代码里做：cot_compress/memory.py
source "$HOME/cot-compress/.venv/bin/activate"
cd "$HOME/cot-compress"
