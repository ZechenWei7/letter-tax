"""v4 词表屏蔽：字符类白名单、special/answer 片段不在白名单、A″ 单字母、A 全词表、审计 leet 检测、采样 100% ∈ allowed。"""
import torch, pytest, importlib.util, pathlib
import cot_compress  # noqa
from transformers import AutoTokenizer
from cot_compress.vocab_mask import (think_allowed_ids, banned_ids, is_whitelisted_text, lift_id, single_letter_ids)
from cot_compress.span import SpanController

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("unsloth/Qwen3-1.7B")

def test_char_class_rules():
    for t in [" 123", "\n\n", " =", "×", "≥", "-", "(", "…", "3-", " ", ""]:
        assert is_whitelisted_text(t), t
    for t in [" a", "I", " the", "的", "年", "α", "д", "x", " to", "�", " 12a"]:
        assert not is_whitelisted_text(t), t

def test_letterfree_excludes_answer_pieces_and_specials(tok):
    allowed = think_allowed_ids(tok, "letterfree", extra_banned=set())
    lid = lift_id(tok)
    assert lid in allowed
    for tid in tok.all_special_ids:
        if tid != lid:
            assert tid not in allowed
    ans_ids = set(tok("<answer>42</answer>", add_special_tokens=False)["input_ids"]) - set(tok("42", add_special_tokens=False)["input_ids"])
    assert any(tid not in allowed for tid in ans_ids)                       # "answer" 片段含字母 → 不在白名单
    assert tok("<think>", add_special_tokens=False)["input_ids"][0] not in allowed
    for tid in tok(" 623 * 794 = 494662\n\n(1) + [2] ≠ 3", add_special_tokens=False)["input_ids"]:
        assert tid in allowed
    for tid in tok(" the and needto 的", add_special_tokens=False)["input_ids"]:
        assert tid not in allowed

def test_plus_single_and_none(tok):
    lf = think_allowed_ids(tok, "letterfree", extra_banned=set()); ps = think_allowed_ids(tok, "letterfree_plus_single", extra_banned=set())
    singles = single_letter_ids(tok)
    assert singles and singles <= ps and not (singles & lf)
    assert tok(" a", add_special_tokens=False)["input_ids"][0] in ps and tok(" ab", add_special_tokens=False)["input_ids"][0] not in ps
    V = len(tok) + 267
    assert len(think_allowed_ids(tok, "none", vocab_size=V)) == V                    # 臂 A/D：allowed = 词表大小
    assert banned_ids(tok, "none", vocab_size=V) == []
    assert len(banned_ids(tok, "letterfree", vocab_size=V, extra_banned=set())) > 0.9 * V

def test_extra_banned_applied(tok):
    seven = tok("7", add_special_tokens=False)["input_ids"][0]
    assert seven in think_allowed_ids(tok, "letterfree", extra_banned=set())
    assert seven not in think_allowed_ids(tok, "letterfree", extra_banned={seven})

def test_sampled_think_tokens_all_allowed(tok):
    """(a) 采样出的 think token 100% ∈ allowed set。"""
    V = len(tok) + 267
    allowed = think_allowed_ids(tok, "letterfree", extra_banned=set(), vocab_size=V)
    ctrl = SpanController(tok, prompt_len=0, cap=4000, reserve=4, banned_ids=banned_ids(tok, "letterfree", vocab_size=V, extra_banned=set()))
    g = torch.Generator().manual_seed(0)
    ids = torch.zeros((2, 5), dtype=torch.long)
    for _ in range(50):
        s = ctrl(ids, torch.randn(2, V, generator=g) * 3)
        samp = torch.multinomial(torch.softmax(s, -1), 1, generator=g)
        assert all(int(x) in allowed for x in samp.flatten())
        ids = torch.cat([ids, samp], 1)

def test_audit_leet_detection():
    spec = importlib.util.spec_from_file_location("audit", pathlib.Path("scripts/07_audit_whitelist.py")); a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
    assert "the" in a.leet_decodings("7h3") and "answer" in a.leet_decodings("4n5w3r")
    assert a.leet_decodings("123") and not any(d in a.COMMON_WORDS for d in a.leet_decodings("123"))
    assert a.leet_decodings("+-") == []
