"""cot-compress 包。

导入顺序约定：本模块无条件先 import unsloth。原因：
  1. Unsloth 要求在 transformers / trl 之前导入才能打补丁；
  2. trl 0.24.0 的 trainer/callbacks.py 无条件 import mergekit（未安装），裸 `from trl import GRPOTrainer`
     会 ModuleNotFoundError，只有 Unsloth 先导入时才能绕过。
所有 scripts/*.py 第一个 import 必须是 `import cot_compress`（或 `import unsloth`）。
"""
import os as _os
if _os.environ.get("COT_NO_UNSLOTH", "0") != "1":
    try:
        import unsloth  # noqa: F401  必须最先（本地 trl 0.24 路径的硬要求）
    except ImportError:              # 云端 TRL 原生路径可不装 unsloth（trl>=0.25 无 mergekit 裸导入问题）
        pass

__all__ = ["config", "parsing", "generation", "model", "evaluate"]
