"""全量留档（2026-09-18）：jsonl.zst 追加写。每次 append 写一个独立 zstd frame，多 frame 拼接的文件用 stream_reader 直接顺序解压。
用途：runs/<run>/rollouts.jsonl.zst（每步全部 rollout）、eval_step<N>.jsonl.zst（每次 eval 的全部输出）、diag_<name>.jsonl.zst（诊断中间产物）。"""
from __future__ import annotations
import io, json, pathlib

def append_jsonl_zst(path, rows) -> int:
    import zstandard
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    if not data:
        return 0
    with open(path, "ab") as f:
        f.write(zstandard.ZstdCompressor(level=9).compress(data))
    return len(rows)

def read_jsonl_zst(path):
    import zstandard
    with open(path, "rb") as f:
        reader = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)
