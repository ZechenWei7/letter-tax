import json
from cot_compress.archive import append_jsonl_zst, read_jsonl_zst

def test_append_frames_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl.zst"
    assert append_jsonl_zst(p, [dict(step=1, text="a" * 500, ids=[1, 2, 3])]) == 1
    assert append_jsonl_zst(p, [dict(step=2, text="中文 b", ids=[]), dict(step=2, text="c")]) == 2      # 第二个 frame 追加
    assert append_jsonl_zst(p, []) == 0
    rows = list(read_jsonl_zst(p))
    assert [r["step"] for r in rows] == [1, 2, 2] and rows[1]["text"] == "中文 b" and rows[0]["ids"] == [1, 2, 3]
    assert p.stat().st_size < 600                                                                      # 压缩生效
