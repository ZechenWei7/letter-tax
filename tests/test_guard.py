import os, pytest
from cot_compress.guard import assert_writable, assert_checkpoint_saved, WorkspaceNotWritable

def test_assert_writable(tmp_path):
    assert_writable(tmp_path / "runs" / "x", "ok case")
    assert not list((tmp_path / "runs" / "x").glob(".wprobe_*"))          # 探针已清理
    f = tmp_path / "afile"; f.write_text("x")
    with pytest.raises(WorkspaceNotWritable):
        assert_writable(f / "sub", "path under a regular file")           # 模拟挂载失效：mkdir / 写入失败

def test_assert_checkpoint_saved(tmp_path):
    with pytest.raises(WorkspaceNotWritable):
        assert_checkpoint_saved(tmp_path / "checkpoint-25")
    d = tmp_path / "checkpoint-25"; d.mkdir(); (d / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(WorkspaceNotWritable):
        assert_checkpoint_saved(d)                                         # 空文件 = 静默丢数据
    (d / "adapter_model.safetensors").write_bytes(b"abc"); assert_checkpoint_saved(d)
