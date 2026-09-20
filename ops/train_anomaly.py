#!/usr/bin/env python3
"""排查用包装器：打开 autograd anomaly 模式后原样执行 scripts/train.py（不改训练代码）。
anomaly 模式会在 backward 出错时打印出错算子对应的 forward 调用栈，用来定位 A1 的 `CUDA error: invalid argument`。
用法与 train.py 完全相同：python ops/train_anomaly.py --config ... --arm A --dry-run ...   （只用于 --dry-run 排查，不用于正式 run）"""
import runpy, sys, pathlib
import torch
torch.autograd.set_detect_anomaly(True, check_nan=False)
root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "scripts")); sys.argv[0] = str(root / "scripts" / "train.py")
runpy.run_path(sys.argv[0], run_name="__main__")
