"""显存分配器调优。WSL 上 expandable_segments 不受支持（unsloth_zoo 导入时会把它从 PYTORCH_CUDA_ALLOC_CONF 里剥掉），
改用运行时可设的两项：garbage_collection_threshold（reserved 超过阈值就主动回收）+ max_split_size_mb（大块不再被切碎）。
配合每个 batch / 每步的 torch.cuda.empty_cache()。实测背景见 README §7。
"""
import os, torch

DEFAULT = "garbage_collection_threshold:0.6,max_split_size_mb:256"

def tune_allocator(settings: str = DEFAULT) -> str:
    try:
        torch.cuda.memory._set_allocator_settings(settings)
        return f"allocator settings applied: {settings} (backend={torch.cuda.get_allocator_backend()})"
    except Exception as e:  # 版本差异时不致命
        return f"allocator settings NOT applied ({e!r}); env={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}"
