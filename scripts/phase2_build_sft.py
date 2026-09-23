"""第二阶段闸门 1 的 SFT 数据（纯 CPU）：把 stage-1 train 划分（2000 题）的求解器推导按四种写法渲染成 SFT 行。
completion = 推导 + "\n</think>\n\n<answer>σ</answer>"；prompt 与 stage-1 相同（IR 题面，chat 模板 + "<think>\n" 预填由训练端加）。
与 B_warm-SFT 的行格式一致（prompt / completion 两列）。
用法：COT_NO_UNSLOTH=1 python scripts/phase2_build_sft.py → data/phase2/gate1_sft_{N1,N2,N2s,N3}.jsonl + manifest.json（含 sha256）"""
import os; os.environ.setdefault("COT_NO_UNSLOTH", "1")
import json, hashlib, pathlib, sys, statistics as st
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tasks import ordering as O
from cot_compress import derivation as D

out = ROOT / "data" / "phase2"; out.mkdir(parents=True, exist_ok=True)
items = O.load_splits("ord_n8_h4_d5", 0)["train"]
man = dict(source="ord_n8_h4_d5 seed0 train", n=len(items), renderer_commit="f887662", files={})
steps = {it["id"]: D.derive(it) for it in items}
for v in D.VERSIONS:
    p = out / f"gate1_sft_{v}.jsonl"
    with open(p, "w") as f:
        for it in items:
            txt = D.render(steps[it["id"]], it, v)
            assert D.parse(txt, it, v) == steps[it["id"]]
            f.write(json.dumps(dict(id=it["id"], prompt=it["prompt"], completion=f"{txt}\n</think>\n\n<answer>{it['answer']}</answer>", notation=v), ensure_ascii=False) + "\n")
    man["files"][v] = dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size,
                           completion_chars_median=st.median(len(json.loads(l)["completion"]) for l in open(p)))
json.dump(man, open(out / "manifest.json", "w"), indent=1); print(json.dumps(man, indent=1))
