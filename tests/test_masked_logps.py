"""(b) 采样器与 trainer 在同一 id 级 mask 下 log-prob 一致（1e-3）；think/forced 位置掩码；A 臂全词表。"""
import torch, pytest
import cot_compress  # noqa
from transformers import AutoTokenizer
from cot_compress.masked_logps import think_position_masks, masked_selective_log_softmax
from cot_compress.rollout import VLLMSpanProcessor
from cot_compress.vocab_mask import banned_ids, think_allowed_ids, lift_id
from cot_compress.span import CLOSE_TEXT

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("unsloth/Qwen3-1.7B")

def test_think_and_forced_masks(tok):
    lid = lift_id(tok); close = tok(CLOSE_TEXT, add_special_tokens=False)["input_ids"]
    # 自然结束：x x </think> a b
    ids = [5, 6, lid, 7, 8]
    th, fo = think_position_masks(ids, lid, force_start=100, close_ids=close)
    assert th == [True, True, True, False, False] and fo == [False] * 5
    # 强制：force_start=3，close 从位置 3 开始
    ids = [5, 6, 7] + close + [9, 9]
    th, fo = think_position_masks(ids, lid, force_start=3, close_ids=close)
    assert th[:4] == [True, True, True, True] and not any(th[4:])
    assert fo == [False] * 3 + [True] * len(close) + [False, False]
    # 无 </think> 且未到 force_start（截断）：全 think、无 forced
    th, fo = think_position_masks([5, 6], lid, 100, close); assert th == [True, True] and fo == [False, False]

def test_sampler_trainer_logp_consistency(tok):
    """vLLM 侧：logits_processor 屏蔽 → /T → softmax 取样本 logp；trainer 侧：masked_selective_log_softmax(logits/T)。"""
    V = len(tok) + 267; T = 1.0
    g = torch.Generator().manual_seed(1)
    banned = banned_ids(tok, "letterfree", vocab_size=V, extra_banned=set())
    allowed = think_allowed_ids(tok, "letterfree", extra_banned=set(), vocab_size=V)
    am = torch.zeros(V, dtype=torch.bool); am[torch.tensor(sorted(allowed))] = True
    proc = VLLMSpanProcessor(tok, cap=4000, reserve=4, banned_ids=banned)
    z = torch.randn(V, generator=g) * 4
    zs = proc([1, 2, 3], z.clone())                          # 采样器：think 阶段屏蔽后的 logits
    p = torch.softmax(zs / T, -1)
    t = torch.multinomial(p, 1, generator=g).item()
    logp_sampler = torch.log(p[t]).item()
    logits = (z / T)[None, None, :]; ids = torch.tensor([[t]]); think = torch.tensor([[True]])
    logp_trainer = masked_selective_log_softmax(logits, ids, am, think)[0, 0].item()
    assert abs(logp_sampler - logp_trainer) < 1e-3
    # 未屏蔽（answering 位置）：两边都是全词表
    p_full = torch.softmax(z / T, -1); t2 = torch.multinomial(p_full, 1, generator=g).item()
    lp_full = masked_selective_log_softmax(logits, torch.tensor([[t2]]), am, torch.tensor([[False]]))[0, 0].item()
    assert abs(lp_full - torch.log(p_full[t2]).item()) < 1e-3
    # 屏蔽 token 在 think 位置的 logp = -inf；A 臂（allowed=None）不屏蔽
    b0 = banned[0]
    assert masked_selective_log_softmax(logits, torch.tensor([[b0]]), am, think)[0, 0].item() == float("-inf")
    assert torch.isfinite(masked_selective_log_softmax(logits, torch.tensor([[b0]]), None, think)[0, 0])
