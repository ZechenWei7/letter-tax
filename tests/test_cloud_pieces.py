"""云端相关但可本地测的纯逻辑：VLLMSpanProcessor 阶段、C_rand/D 预填（think 不来自策略）、TrajStats、压缩统计、矩阵命名。"""
import torch, pytest, importlib.util, pathlib, random
import cot_compress  # noqa
from transformers import AutoTokenizer
from cot_compress.rollout import VLLMSpanProcessor, build_prefilled_prompt, make_prefilled_rollout_func
from cot_compress.vocab_mask import banned_ids, lift_id
from cot_compress.traj_stats import TrajStats, think_ids
from cot_compress.evaluate import compression_stats_rows
from cot_compress import rewards
from cot_compress.span import CLOSE_TEXT

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("unsloth/Qwen3-1.7B")

def test_vllm_processor_stateless_phases(tok):
    V = len(tok) + 267
    proc = VLLMSpanProcessor(tok, cap=40, reserve=4, banned_ids=banned_ids(tok, "letterfree", vocab_size=V, extra_banned=set()))
    the = tok(" the", add_special_tokens=False)["input_ids"][0]; digit = tok("7", add_special_tokens=False)["input_ids"][0]
    s = proc([digit] * 3, torch.zeros(V)); assert torch.isinf(s[the]) and not torch.isinf(s[digit]) and torch.isinf(s[V - 1])
    s = proc([1, 2, 3], [digit] * 3, torch.zeros(V)); assert torch.isinf(s[the])
    gen = [digit] * proc.force_start
    for k, cid in enumerate(proc.close_ids):
        s = proc(gen, torch.zeros(V)); assert s.argmax().item() == cid and torch.isinf(s[digit]); gen = gen + [cid]
    s = proc(gen, torch.zeros(V)); assert not torch.isinf(s[the]) and not torch.isinf(s[digit])
    lift = tok("</think>", add_special_tokens=False)["input_ids"]
    s = proc([digit, digit] + lift, torch.zeros(V)); assert not torch.isinf(s[the])
    reh = tok("<answer>42</answer>", add_special_tokens=False)["input_ids"]
    s = proc([digit] + reh, torch.zeros(V)); assert torch.isinf(s[the])

class _FakeLLM:
    """记录 prompts，返回固定答案 token；用于证明策略只采样答案。"""
    def __init__(self, tok): self.tok = tok; self.calls = []
    def generate(self, ctxs, sp, use_tqdm=False):
        self.calls.append((list(ctxs), sp.max_tokens))
        ans = self.tok("42</answer>", add_special_tokens=False)["input_ids"]
        class O:  # 模拟 vllm RequestOutput
            def __init__(s, ctx):
                s.prompt_token_ids = self.tok(ctx, add_special_tokens=False)["input_ids"]
                class C: token_ids = ans; logprobs = None; finish_reason = "stop"
                s.outputs = [C()]
        return [O(c) for c in ctxs]

def test_crand_and_d_prefilled_rollout(tok):
    lid = lift_id(tok)
    rows = [dict(id="i1", completion="<think>\n1 + 2 = 3 ; 3 * 4 = 12\n</think>\n\n<answer>12</answer>"),
            dict(id="i2", completion="<think>\n7 * 8 = 56 , 56 + 1\n</think>\n\n<answer>57</answer>")]
    stats = TrajStats(tok, rows, lid)
    assert stats.lengths and all(lid not in ids for ids in stats.by_item["i1"])
    fake = _FakeLLM(tok)
    class A: temperature = 1.0; top_p = 1.0; top_k = None
    rf = make_prefilled_rollout_func(lambda: fake, tok, mode="crand", stats=stats, item_id_of=lambda p: "i1", seed=0)
    try:
        _crand_body(rf, fake, tok, stats, A)
    finally:
        rewards.SIDE["L"] = None                                # 无论成败都清侧通道，避免污染其他测试

def _crand_body(rf, fake, tok, stats, A):
    out = rf(["PROMPT-A<think>\n", "PROMPT-B<think>\n"], A(), tok)
    ctxs, max_tokens = fake.calls[-1]
    assert max_tokens == 40                                     # 策略只生成答案区（32 token 上限 + 余量）
    support = set(stats.support)
    for ctx, L, ids in zip(ctxs, rewards.SIDE["L"], rf.last_think_ids):
        assert ctx.endswith(CLOSE_TEXT) and ctx.startswith("PROMPT-")
        body = ctx.split("<think>\n", 1)[1].rsplit("\n" + CLOSE_TEXT, 1)[0]
        assert body == tok.decode(ids)                          # 上下文里的 think 段就是采样器抽出的那串
        assert all(i in support for i in ids)                   # think token 全来自 unigram 支持集，不来自策略
        assert L == len(ids) and L in stats.lengths             # 长度来自 B 的长度分布（同题优先）
    assert out["completion_ids"][0] == tok("42</answer>", add_special_tokens=False)["input_ids"]
    rewards.SIDE["L"] = None
    rfd = make_prefilled_rollout_func(lambda: fake, tok, mode="d", seed=0)      # 空 think 的预填路径仍保留（臂 D 已删，仅作机制测试）
    rfd(["PROMPT-A<think>\n"], A(), tok)
    assert fake.calls[-1][0][0] == "PROMPT-A<think>\n" + CLOSE_TEXT and rewards.SIDE["L"] == [0]

def test_transplant_and_unigram(tok):
    lid = lift_id(tok)
    rows = [dict(id="a", completion="<think>\n1 2 3 4 5\n</think>\n\n<answer>1</answer>"), dict(id="b", completion="<think>\n9 8 7\n</think>\n\n<answer>2</answer>")]
    st = TrajStats(tok, rows, lid); rng = random.Random(0)
    tr = st.sample_transplant("a", target_len=3, rng=rng)
    assert tr == st.by_item["b"][0]
    assert len(st.sample_unigram(10, rng)) == 10

def test_compression_stats(tok):
    rows = [dict(completion="123 456 789 " * 40 + "<answer>1</answer>"), dict(completion="a quick brown fox jumps over the lazy dog " * 20 + "<answer>2</answer>")]
    d = compression_stats_rows(tok, rows); assert 0 < d["gzip_ratio"] < 1 and d["bigram_entropy_bits"] > 0
    rep = compression_stats_rows(tok, [dict(completion="7 " * 400 + "<answer>1</answer>")])
    assert rep["gzip_ratio"] < d["gzip_ratio"] and rep["bigram_entropy_bits"] < d["bigram_entropy_bits"]

def test_run_matrix_name():
    spec = importlib.util.spec_from_file_location("rm", pathlib.Path("scripts/run_matrix.py")); rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)
    assert rm.run_name(dict(key="cups_n6_k12", arm="B", lam=0.5, seed=1)) == "B_0.5_cups_n6_k12_s1"

def test_span_batch_state_engine_level(tok):
    """vLLM V1 引擎级处理器的核心状态机：批内多请求、活引用的 out 列表、屏蔽 / 强制 / 解除、无配置的请求不受影响。"""
    from cot_compress.rollout import SpanBatchState, EXTRA_KEY
    V = len(tok) + 267
    st = SpanBatchState(tok, V)
    the = tok(" the", add_special_tokens=False)["input_ids"][0]; digit = tok("7", add_special_tokens=False)["input_ids"][0]
    outB, outA, outPlain = [], [], []
    st.reqs[0] = st.make_entry({EXTRA_KEY: dict(cap=40, reserve=4, mode="letterfree")}, outB)
    st.reqs[1] = st.make_entry({EXTRA_KEY: dict(cap=40, reserve=4, mode="none")}, outA)
    assert st.make_entry(None, outPlain) is None and st.make_entry({}, outPlain) is None
    lg = st.apply(torch.zeros(3, V))
    assert torch.isinf(lg[0, the]) and not torch.isinf(lg[0, digit]) and torch.isinf(lg[0, V - 1])    # B：字母禁、数字留、越界禁
    assert not torch.isinf(lg[1]).any() and not torch.isinf(lg[2]).any()                               # A 与无配置请求不动
    fs = st.reqs[0]["force_start"]; outB.extend([digit] * fs); outA.extend([digit] * fs)               # 活引用：直接往列表里加
    for k, cid in enumerate(st.close_ids):
        lg = st.apply(torch.zeros(3, V))
        assert lg[0].argmax().item() == cid and lg[1].argmax().item() == cid and torch.isinf(lg[0, digit])
        outB.append(cid); outA.append(cid)
    lg = st.apply(torch.zeros(3, V)); assert not torch.isinf(lg[0]).any()                               # close 完成 → answering
    out2 = [digit, st.lift_id]; st.reqs[2] = st.make_entry({EXTRA_KEY: dict(cap=400, reserve=4, mode="letterfree")}, out2)
    lg = st.apply(torch.zeros(3, V)); assert not torch.isinf(lg[2]).any() and st.reqs[2]["lifted"]      # 自然 </think> → 解除
