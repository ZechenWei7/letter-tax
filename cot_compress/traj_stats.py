"""从已收敛 run 的样本 / 诊断行里取轨迹统计（C_rand、填充 / 移植诊断用）。
think 段 = 完成文本里第一个 </think> 之前（不含）。"""
from __future__ import annotations
import json, pathlib, random
from collections import Counter, defaultdict
from .parsing import THINK_CLOSE, ANSWER_OPEN

def load_rows(run_dir) -> list[dict]:
    run_dir = pathlib.Path(run_dir)
    cands = [run_dir / "diagnostics_rows.jsonl"]
    samples = sorted((run_dir.parents[0].parent / "samples" / run_dir.name).glob("step*.jsonl")) if (run_dir.parents[0].parent / "samples" / run_dir.name).exists() else []
    if samples:
        cands.append(samples[-1])
    for p in cands:
        if p.exists():
            return [json.loads(l) for l in open(p)]
    raise FileNotFoundError(f"no diagnostics_rows.jsonl or samples for {run_dir}")

def think_text(completion: str) -> str:
    t = completion.split(THINK_CLOSE, 1)[0] if THINK_CLOSE in completion else completion.rsplit(ANSWER_OPEN, 1)[0]
    return t.split("<think>", 1)[1] if "<think>" in t else t

def think_ids(tok, completion: str, lift_id: int) -> list[int]:
    return [i for i in tok(think_text(completion), add_special_tokens=False)["input_ids"] if i != lift_id]

class TrajStats:
    def __init__(self, tok, rows: list[dict], lift_id: int):
        self.tok = tok
        self.by_item: dict[str, list[list[int]]] = defaultdict(list)
        self.unigram = Counter(); self.lengths = []
        for r in rows:
            ids = think_ids(tok, r["completion"], lift_id)
            self.by_item[r["id"]].append(ids); self.unigram.update(ids); self.lengths.append(len(ids))
        self.support = sorted(self.unigram)
        tot = sum(self.unigram.values())
        self.probs = [self.unigram[i] / tot for i in self.support] if tot else []

    def sample_length(self, item_id: str, rng: random.Random) -> int:
        """同题优先，否则随机抽一条轨迹的长度。"""
        pool = self.by_item.get(item_id) or None
        return len(rng.choice(pool)) if pool else rng.choice(self.lengths)

    def sample_unigram(self, L: int, rng: random.Random) -> list[int]:
        return rng.choices(self.support, weights=self.probs, k=L) if L > 0 else []

    def sample_transplant(self, item_id: str, target_len: int | None, rng: random.Random) -> list[int]:
        """他题轨迹（i→j）：排除同题，取长度最接近 target_len 的。"""
        others = [(iid, ids) for iid, lst in self.by_item.items() if iid != item_id for ids in lst]
        if not others:
            return []
        if target_len is None:
            return rng.choice(others)[1]
        return min(others, key=lambda x: (abs(len(x[1]) - target_len), rng.random()))[1]
