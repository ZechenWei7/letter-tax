"""解析测试。注意：import cot_compress 会 import unsloth（慢，需要 GPU 环境）。"""
import cot_compress  # noqa: F401
from cot_compress.parsing import parse_completion, split_think, ANSWER_OPEN
from tasks import cups

def test_split_think():
    assert split_think("<think>abc</think>xyz") == ("abc", "xyz")
    assert split_think("abc</think>xyz") == ("abc", "xyz")   # 模板可能已注入 <think>
    assert split_think("no close") == (None, "no close")

def test_parse_ok():
    p = parse_completion("<think>\nswap...\n</think>\n\n<answer>3U 1D</answer>", validate=cups.is_valid)
    assert p.format_ok and p.answer == "3U 1D" and p.think.strip() == "swap..."

def test_parse_failures():
    assert parse_completion("<think>abc <answer>3U 1D</answer>").reason == "no_think_close"
    assert parse_completion("<think>a</think> 3U 1D").reason == "no_answer"
    assert parse_completion("<think>a</think> <answer>3U 1D").reason == "unfinished"
    assert parse_completion("<think>a</think> <answer>3U 1D</answer> or <answer>4U 1D</answer>").reason == "multi_answer"
    assert parse_completion("<think>a</think> <answer>cup 3</answer>", validate=cups.is_valid).reason == "bad_answer"
    # </think> 之前的 <answer> 不算
    assert parse_completion("<think><answer>1-up</answer></think> <answer>3U 1D</answer>").answer == "3U 1D"
    # 答案后允许有尾随文本（不算第二候选）
    assert parse_completion("<think>a</think> <answer>3U 1D</answer>\nThat is the answer.").format_ok

def test_parse_direct_mode():
    p = parse_completion(ANSWER_OPEN + "3U 1D</answer>", span="none", validate=cups.is_valid)
    assert p.format_ok and p.answer == "3U 1D"
    assert parse_completion(ANSWER_OPEN + "3U 1D", span="none").reason == "unfinished"

def test_parse_pre_answer_span():
    p = parse_completion("swap 1 2 -> ... so 3 down\n<answer>3D 1D</answer>", span="pre_answer", validate=cups.is_valid)
    assert p.format_ok and p.answer == "3D 1D" and p.think == "swap 1 2 -> ... so 3 down\n"
    assert parse_completion("no tags at all 3D 1D", span="pre_answer").reason == "no_answer"
    # 同值预演 = 一个候选（Qwen3 在思考里预演答案）；不同值 = 多候选
    p = parse_completion("think <answer>3U 1D</answer>. done\n</think>\n\n<answer>3U 1D</answer>", span="pre_answer",
                         validate=cups.is_valid, validate_norm=cups.normalize)
    assert p.format_ok and p.answer == "3U 1D" and p.think.endswith("</think>\n\n")   # L 段到最后一个 <answer>
    assert parse_completion("x</think> <answer>3U 1D</answer> <answer>4U 1D</answer>", span="pre_answer",
                            validate_norm=cups.normalize).reason == "multi_answer"
    assert parse_completion("<answer>9U 9D</answer> draft\n</think>\n\n<answer>3U 1D</answer>", span="pre_answer",
                            validate=cups.is_valid, validate_norm=cups.normalize).answer == "3U 1D"          # 思考内不同值草稿忽略
    # </think> 与最终 <answer> 之间只允许空白
    assert parse_completion("think\n</think>\n\n<answer>3U 1D</answer>", span="pre_answer", validate=cups.is_valid).format_ok
    assert parse_completion("think\n</think>\nSo the cup is 3.\n<answer>3U 1D</answer>", span="pre_answer",
                            validate=cups.is_valid).reason == "text_after_think"
    # 预演 + 合法收尾仍 OK；无 </think>（guided）不适用该规则
    assert parse_completion("<answer>3U 1D</answer> hmm\n</think>\n\n<answer>3U 1D</answer>", span="pre_answer",
                            validate=cups.is_valid, validate_norm=cups.normalize).format_ok
    assert parse_completion("reasoning then <answer>3U 1D</answer>", span="pre_answer", validate=cups.is_valid).format_ok
