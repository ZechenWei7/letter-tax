"""第二阶段渲染器核对（纯 CPU）：
  1. 每个连接词 / 符号在 Qwen3 分词器下（带前导空格）是单 token，报 id；N3 符号在 B 的白名单内。
  2. 对每道题：derive（记录版求解器，与 tasks.ordering.probe / canonical_path 断言一致）→ 四个版本渲染 → parse → 与原 steps 完全相等（往返）→ verify（独立重放，每条链的每一环都是当时可用的显式约束）。
  3. N2 / N2s / N3 逐 token 对齐：token 序列等长，只在连接词位置不同，且那些位置正是各自连接词的 id。
  4. N3 全文无字母 token，且全部 token 在 B 的白名单（letterfree，8351）内。
用法：HF_HOME=~/hf python scripts/phase2_render_check.py → results/phase2_render_check.json"""
import os; os.environ.setdefault("COT_NO_UNSLOTH", "1")
import json, pathlib, sys, collections, unicodedata, statistics as st
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from tasks import ordering as O
from cot_compress import derivation as D
from cot_compress.vocab_mask import think_allowed_ids, load_extra_banned

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
allowed = think_allowed_ids(tok, "letterfree", extra_banned=load_extra_banned())
ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]
has_letter = lambda s: any(unicodedata.category(ch)[0] == "L" for ch in s)

conn = {}
for v in ("N2", "N2s", "N3"):
    for c, w in D.MAPS[v].items():
        t = ids(" " + w); assert len(t) == 1, (v, c, w, t)
        conn.setdefault(c, {})[v] = dict(surface=" " + w, id=t[0], whitelist=(t[0] in allowed), has_letter=has_letter(w))
for c in conn: assert not conn[c]["N3"]["has_letter"] and conn[c]["N3"]["whitelist"], c
assert len({conn[c][v]["id"] for c in conn for v in ("N2", "N2s", "N3")}) == 3 * len(conn), "connective ids must be distinct"
cid = {v: {conn[c][v]["id"] for c in conn} for v in ("N2", "N2s", "N3")}

sets = {f"stage1/{k}": v for k, v in O.load_splits("ord_n8_h4_d5", 0).items() if isinstance(v, list)}
sets["p2eval"] = json.load(open(O.DATA_DIR / "ord_n8_h4_d5_p2eval_seed1.json"))["p2eval"]
res = dict(connectives=conn, per_set={}, failures=[])
ev_types = collections.Counter(); depth_hist = collections.Counter(); probe_fired = 0; nconn = []
for name, items in sets.items():
    ok = collections.Counter()
    for it in items:
        try: steps = D.derive(it)
        except D.ProbeFired as e: probe_fired += 1; res["failures"].append([it["id"], "probe_fired", str(e)]); continue
        except AssertionError as e: res["failures"].append([it["id"], "derive", str(e)]); continue
        ev_types.update(s["t"] for s in steps)
        depth = max((sum(1 for s in steps[:i + 1] if s["t"] == "assume") - sum(1 for s in steps[:i + 1] if s["t"] in ("cycle", "both")) for i in range(len(steps))), default=0)
        depth_hist[depth] += 1
        good = True
        texts = {v: D.render(steps, it, v) for v in D.VERSIONS}
        for v, txt in texts.items():
            try: back = D.parse(txt, it, v)
            except Exception as e: back = None; res["failures"].append([it["id"], f"parse_{v}", repr(e)[:120]])
            if back != steps: good = False; res["failures"].append([it["id"], f"roundtrip_{v}", "mismatch"]) if back is not None else None
            else: ok[f"roundtrip_{v}"] += 1
            vr = D.verify(back, it) if back is not None else (False, "no parse")
            if vr[0]: ok[f"verify_{v}"] += 1
            else: good = False; res["failures"].append([it["id"], f"verify_{v}", vr[1]])
        T = {v: ids(texts[v]) for v in ("N2", "N2s", "N3")}
        if not (len(T["N2"]) == len(T["N2s"]) == len(T["N3"])): good = False; res["failures"].append([it["id"], "align_len", [len(x) for x in T.values()]])
        else:
            pos = [i for i in range(len(T["N2"])) if not (T["N2"][i] == T["N2s"][i] == T["N3"][i])]
            if all(T["N2"][i] in cid["N2"] and T["N2s"][i] in cid["N2s"] and T["N3"][i] in cid["N3"] for i in pos) and \
               all(T["N2"][i] not in cid["N2"] for i in range(len(T["N2"])) if i not in pos): ok["aligned"] += 1; nconn.append(len(pos))
            else: good = False; res["failures"].append([it["id"], "align_pos", "diff outside connective slots"])
        if not any(has_letter(tok.decode([i])) for i in T["N3"]) and all(i in allowed for i in T["N3"]): ok["N3_letterfree_whitelist"] += 1
        else: good = False; res["failures"].append([it["id"], "N3_letter_or_offlist", ""])
        ok["all_pass"] += good; ok["n"] += 1
    res["per_set"][name] = dict(ok)
res.update(event_types=dict(ev_types), max_assumption_depth_hist=dict(sorted(depth_hist.items())), probe_fired=probe_fired,
           connective_slots_per_item=dict(median=st.median(nconn), min=min(nconn), max=max(nconn)), n_failures=len(res["failures"]))
res["failures"] = res["failures"][:50]
json.dump(res, open(ROOT / "results/phase2_render_check.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "connectives"}, indent=1))
print("\n连接词：")
for c, d in conn.items(): print(f"  {c:14}" + "  ".join(f"{v}:{d[v]['surface']!r}={d[v]['id']}" for v in ("N2", "N2s", "N3")))
