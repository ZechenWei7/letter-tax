"""持久盘可写性保护：global volume 断过一次（geesefs "Transport endpoint is not connected"），
任何写 checkpoint / 启动 run 之前先探测，不可写就报错停下，而不是静默丢数据。"""
from __future__ import annotations
import os, pathlib, time, uuid

class WorkspaceNotWritable(RuntimeError):
    pass

def assert_writable(path, what: str = "") -> None:
    """在 path 下写-读-删一个探针文件（含 fsync）；任何一步失败 → WorkspaceNotWritable。"""
    d = pathlib.Path(path)
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / f".wprobe_{uuid.uuid4().hex[:8]}"
        payload = f"{time.time()}"
        with open(probe, "w") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        if open(probe).read() != payload:
            raise OSError("read-back mismatch")
        probe.unlink()
    except Exception as e:
        raise WorkspaceNotWritable(f"{d} not writable ({what}): {e!r}") from e

def assert_checkpoint_saved(ckpt_dir) -> None:
    """保存后核对：目录存在且 adapter 权重非空。"""
    d = pathlib.Path(ckpt_dir)
    files = [p for p in d.glob("adapter_model.*")] if d.exists() else []
    if not files or any(p.stat().st_size == 0 for p in files):
        raise WorkspaceNotWritable(f"checkpoint {d} missing or empty after save: {[p.name for p in files]}")
