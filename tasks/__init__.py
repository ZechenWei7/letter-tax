"""任务注册。难度键决定任务：cups_* -> cups；mul_* / add_* -> arith。
数据无限生成，seed + 难度键 + split 决定随机流；eval 集固定 n 题。
"""
from __future__ import annotations
import random
from types import ModuleType
from . import cups, arith, kk, ordering

def task_for_key(key: str) -> ModuleType:
    if key.startswith("cups_"):
        return cups
    if key.startswith(("mul_", "add_")):
        return arith
    if key.startswith("kk_"):
        return kk
    if key.startswith("ord_"):
        return ordering
    raise ValueError(f"unknown difficulty key: {key}")

def _rng(seed: int, key: str, split: str) -> random.Random:
    return random.Random(f"cot-compress|{seed}|{key}|{split}")

def make_eval_set(key: str, n: int, seed: int = 0, split: str = "stop") -> list[dict]:
    """kk：从预生成划分取（split ∈ {stop, report}；stopping-eval 只用于停止判据，reporting-eval 只用于报告；n ≤ 500）。"""
    task = task_for_key(key)
    if task in (kk, ordering):
        assert split in ("stop", "report", "direct"), split
        items = task.load_splits(key, seed)[split]
        assert n <= len(items), f"eval n={n} > split size {len(items)}"
        return items[:n]
    rng = _rng(seed, key, "eval")
    items = []
    for i in range(n):
        it = task.generate(rng, key)
        it["id"] = f"{key}/eval/{seed}/{i}"
        items.append(it)
    return items

def iter_train(key: str, seed: int = 0):
    """kk：在 train 划分上循环（每轮按 seed 洗牌）；其余任务无限生成。"""
    task = task_for_key(key)
    if task in (kk, ordering):
        items = list(task.load_splits(key, 0)["train"]); rng = random.Random(f"{task.__name__}-train-order|{seed}")
        while True:
            order = list(range(len(items))); rng.shuffle(order)
            for i in order:
                yield items[i]
    rng = _rng(seed, key, "train")
    i = 0
    while True:
        it = task.generate(rng, key)
        it["id"] = f"{key}/train/{seed}/{i}"
        yield it
        i += 1
