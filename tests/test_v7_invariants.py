"""v7 §9 测试清单（CPU 部分）：
  mask 覆盖 10k 采样零命中 + masked softmax 在白名单上和为 1；采样端 / 训练端 logprob 1e-3 内一致；
  禁用 token 的 lm_head 行在 think 位置零梯度；embed / lm_head 同步（TRL 名字映射，tiny 模型 + 假 vLLM）；
  C_rand 玩具（策略 think token 不进上下文、上下文 unigram 卡方 p>0.05、trace 位置 loss 恰为 0）；
  生成器数据文件：全体规范形唯一、唯一解（向量化 2^N）、S 暴力核对（子集）、置换不变。
"""
import json, math, os, pathlib, random
import numpy as np
import torch, pytest
import cot_compress  # noqa
from transformers import AutoTokenizer
from cot_compress.vocab_mask import banned_ids, think_allowed_ids, lift_id
from cot_compress.masked_logps import masked_selective_log_softmax, think_position_masks
from cot_compress.rollout import VLLMSpanProcessor, build_prefilled_prompt
from cot_compress.span import CLOSE_TEXT
from cot_compress.traj_stats import TrajStats
from tasks import kk

ROOT = pathlib.Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("unsloth/Qwen3-1.7B")

@pytest.fixture(scope="module")
def mask(tok):
    V = len(tok) + 267
    allowed = think_allowed_ids(tok, "letterfree", vocab_size=V)
    am = torch.zeros(V, dtype=torch.bool); am[torch.tensor(sorted(allowed))] = True
    return V, allowed, am, banned_ids(tok, "letterfree", vocab_size=V)

def test_mask_10k_samples_zero_hits_and_softmax_sums_to_one(tok, mask):
    V, allowed, am, banned = mask
    proc = VLLMSpanProcessor(tok, cap=4000, reserve=4, banned_ids=banned)
    g = torch.Generator().manual_seed(0); hits = 0; n = 0
    for _ in range(20):
        z = torch.randn(500, V, generator=g) * 4
        zs = torch.stack([proc([1, 2, 3], row.clone()) for row in z[:2]])         # 逐行处理器等价于批量屏蔽：用 -inf 掩码批量算
        z = z.masked_fill(~am[None, :], float("-inf"))
        p = torch.softmax(z, -1)
        assert torch.allclose(p[:, am].sum(-1), torch.ones(500), atol=1e-5)          # 白名单上和为 1
        assert (p[:, ~am] == 0).all()
        s = torch.multinomial(p, 1, generator=g).flatten()
        hits += sum(int(x) not in allowed for x in s.tolist()); n += len(s)
        assert torch.equal(torch.isinf(zs), ~am[None, :].expand(2, V))               # 处理器屏蔽集合 = allowed 补集
    assert n == 10000 and hits == 0

def test_sampler_trainer_logp_within_1e3(tok, mask):
    V, allowed, am, banned = mask
    proc = VLLMSpanProcessor(tok, cap=4000, reserve=4, banned_ids=banned); g = torch.Generator().manual_seed(2)
    for T in (1.0, 0.6):
        z = torch.randn(V, generator=g) * 4
        zs = proc([1], z.clone()); p = torch.softmax(zs / T, -1)
        for t in torch.multinomial(p, 5, generator=g).tolist():
            lp_s = torch.log(p[t]).item()
            lp_t = masked_selective_log_softmax((z / T)[None, None, :], torch.tensor([[t]]), am, torch.tensor([[True]]))[0, 0].item()
            assert abs(lp_s - lp_t) < 1e-3

def test_banned_lm_head_rows_zero_grad_at_think_positions(mask):
    """tiny lm_head：logits = h @ W；think 位置的 masked logp 对禁用行的梯度为 0；answer 位置（全词表）非 0。"""
    V, allowed, am, banned = mask
    torch.manual_seed(0); d = 8; W = torch.randn(d, V, requires_grad=True); h = torch.randn(1, 2, d)
    ids = torch.tensor([[sorted(allowed)[10], banned[3]]]); think = torch.tensor([[True, False]])
    # 位置 0（think）：禁用列的梯度为 0
    lp0 = masked_selective_log_softmax(h @ W, ids, am, think)[0, 0]; lp0.backward()
    assert torch.all(W.grad[:, torch.tensor(banned)] == 0) and torch.any(W.grad[:, torch.tensor(sorted(allowed))] != 0)
    W.grad = None
    lp1 = masked_selective_log_softmax(h @ W, ids, am, think)[0, 1]; lp1.backward()
    assert torch.any(W.grad[:, torch.tensor(banned)] != 0)       # answer 位置：全词表参与

def test_embed_lm_head_sync_name_mapping_tiny_model():
    """镜像 TRL 0.25.1 _move_model_to_vllm 的名字映射：LoRA 挂 embed_tokens / lm_head / q_proj 的 tiny Qwen2 + 假 vLLM 模型（记录 load_weights）。
    检查：合并后的 embed / lm_head 权重以 vLLM 名字（model.embed_tokens.weight / lm_head.weight）同步，且数值 = 基座 + LoRA delta；lora_ 参数不外泄。"""
    import importlib.util
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model
    spec = importlib.util.spec_from_file_location("chk", ROOT / "scripts/10_check_lora_embed_sync.py"); chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)
    cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=False)
    torch.manual_seed(0); base = Qwen2ForCausalLM(cfg)
    emb0 = base.get_input_embeddings().weight.detach().clone(); lm0 = base.lm_head.weight.detach().clone()
    pm = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "embed_tokens", "lm_head"]))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_" in n: p.normal_(0, 0.1)          # PEFT：Linear 的 B 与 Embedding 的 A 初始为 0，两者都随机化才有 delta
    class FakeModel:
        def __init__(self): self.loaded = {}
        def load_weights(self, pairs):
            for n, p in pairs: self.loaded[n] = p.detach().clone()
    class Obj: pass
    fm = FakeModel(); llm = Obj(); llm.llm_engine = Obj(); llm.llm_engine.model_executor = Obj(); llm.llm_engine.model_executor.driver_worker = Obj()
    llm.llm_engine.model_executor.driver_worker.model_runner = Obj(); llm.llm_engine.model_executor.driver_worker.model_runner.model = fm; llm.reset_prefix_cache = lambda: None
    names = chk.trl_sync(pm, llm)
    assert "model.embed_tokens.weight" in fm.loaded and "lm_head.weight" in fm.loaded and "model.layers.0.self_attn.q_proj.weight" in fm.loaded
    assert not any("lora" in n or "base_layer" in n or "modules_to_save" in n for n in names)
    assert not torch.allclose(fm.loaded["model.embed_tokens.weight"], emb0) and not torch.allclose(fm.loaded["lm_head.weight"], lm0)   # delta 已合并
    pm.merge_adapter(); assert torch.allclose(fm.loaded["lm_head.weight"], base.lm_head.weight.detach()); pm.unmerge_adapter()
    assert torch.allclose(base.lm_head.weight.detach(), lm0, atol=1e-6)                                    # unmerge 后恢复

def _chisq_p(obs, exp):
    """卡方拟合优度 p（Wilson–Hilferty 正态近似，无 scipy）。"""
    chi = sum((o - e) ** 2 / e for o, e in zip(obs, exp) if e > 0); k = sum(1 for e in exp if e > 0) - 1
    if k <= 0: return 1.0
    z = ((chi / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))

def test_crand_toy(tok):
    """C_rand：上下文 think 段 = 从源 unigram 抽的 token（不含策略 token）；其 unigram 与源分布卡方 p>0.05；trace 位置 loss 恰为 0（completion 只含答案）。"""
    lid = lift_id(tok); rng = random.Random(0)
    vocab = [tok(w, add_special_tokens=False)["input_ids"][0] for w in ["1", "2", "3", " =", " &", " |", " !", " →", " ⇒", " »"]]
    rows = [dict(id=f"i{j}", completion="<think>\n" + tok.decode(rng.choices(vocab, weights=range(1, 11), k=60)) + "\n</think>\n\n<answer>1 0</answer>") for j in range(30)]
    st = TrajStats(tok, rows, lid)
    sampled = [st.sample_unigram(st.sample_length(f"i{j}", rng), rng) for j in range(200)]
    flat = [t for s in sampled for t in s]
    assert set(flat) <= set(st.support)                                                  # 不进策略 token
    tot = sum(st.unigram.values()); exp = [st.unigram[t] / tot * len(flat) for t in st.support]; obs = [flat.count(t) for t in st.support]
    assert _chisq_p(obs, exp) > 0.05
    prompt = "PROMPT<think>\n"; ctx = build_prefilled_prompt(tok, prompt, sampled[0])
    assert ctx.startswith(prompt) and ctx.endswith(CLOSE_TEXT)
    # loss：TRL 只对 completion 计 loss；C_rand 的 completion = 答案续写，trace（在 prompt 里）的 completion_mask 全 0
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]; comp_ids = tok("1 0</answer>", add_special_tokens=False)["input_ids"]
    full = ctx_ids + comp_ids; loss_mask = [0] * len(ctx_ids) + [1] * len(comp_ids)
    assert sum(loss_mask[:len(ctx_ids)]) == 0 and sum(loss_mask) == len(comp_ids)
    th, fo = think_position_masks(comp_ids, lid, force_start=10 ** 9, close_ids=tok(CLOSE_TEXT, add_special_tokens=False)["input_ids"])
    assert not any(fo)

def _count_solutions_np(n, stmts):
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)     # [2^n, n]
    ok = np.ones(2 ** n, dtype=bool)
    for s, f, x, y in stmts:
        bx = bits[:, x - 1]; by = bits[:, y - 1] if y is not None else None; nx = 1 - bx
        val = {"x": bx, "!x": nx, "x&y": bx & by, "x|y": bx | by, "x=y": (bx == by).astype(np.int8), "x!=y": (bx != by).astype(np.int8),
               "!x&y": nx & by, "x&!y": bx & (1 - by), "!x|y": nx | by}[f] if f not in ("x", "!x") else {"x": bx, "!x": nx}[f]
        ok &= (val == bits[:, s - 1])
    return int(ok.sum())

def test_generator_data_files_unique_and_S_verified():
    """data/kk/*.json（两格 6000 题）：规范形全体唯一、唯一解（向量化 2^N）、S==1（随机 150 题独立暴力核对）。文件缺失则跳过。"""
    from tests.test_kk import brute_min_depth
    files = sorted(f for f in (ROOT / "data/kk").glob("kk_n*_seed*.json") if not f.name.endswith(".report.json"))
    if not files: pytest.skip("no data files (run scripts/kk_build_splits.py)")
    rng = random.Random(0); forms = set(); total = 0
    for fp in files:
        sp = json.load(open(fp))
        items = [it for v in sp.values() for it in v]; total += len(items)
        for it in items:
            forms.add(it["meta"]["canonical"])
            assert _count_solutions_np(it["meta"]["n"], [tuple(s) for s in it["meta"]["stmts"]]) == 1
        for it in rng.sample(items, min(75, len(items))):
            assert brute_min_depth(it["meta"]["n"], [tuple(s) for s in it["meta"]["stmts"]]) == 1 == it["meta"]["S"]
        kk.assert_disjoint(sp)
    assert len(forms) == total                                                                # 跨格也不撞（不同 N 自然不撞）

@pytest.mark.skipif(not os.environ.get("KK_FULL"), reason="set KK_FULL=1: 10k 实例生成（多进程，~10 min）")
def test_generator_10k_instances_unique_and_S():
    """10k 实例（两格各 5000，独立 seed 流 'full10k'）：规范形唯一、唯一解、S==1（全部独立暴力核对）。"""
    import multiprocessing as mp
    from tests.test_kk import brute_min_depth
    forms = set()
    for key in ("kk_n10_s1", "kk_n12_s1"):
        with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
            items = pool.map(kk.generate_indexed, [(key, "full10k", 0, j) for j in range(5000)], chunksize=8)
        for it in items:
            st = [tuple(s) for s in it["meta"]["stmts"]]
            forms.add(it["meta"]["canonical"])
            assert _count_solutions_np(it["meta"]["n"], st) == 1
            assert brute_min_depth(it["meta"]["n"], st) == 1 == it["meta"]["S"]
    # 原始候选流不去重（build_splits 去重）；允许 ≤0.5% 撞形
    assert len(forms) >= 0.995 * 10000
