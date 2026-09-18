"""YAML 配置加载 + `--set a.b=c` 覆盖 + `_base:` 继承。"""
from __future__ import annotations
import copy, pathlib, yaml
from typing import Any

def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst

def _parse_value(s: str) -> Any:
    # 用 yaml 解析标量：true/false/null/数字/字符串
    return yaml.safe_load(s)

def load_config(path: str | pathlib.Path, sets: list[str] | None = None) -> dict:
    path = pathlib.Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base", None)
    if base:
        base_cfg = load_config(path.parent / base)
        cfg = _deep_update(base_cfg, cfg)
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return cfg

def get(cfg: dict, dotted: str, default=None):
    node = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node
