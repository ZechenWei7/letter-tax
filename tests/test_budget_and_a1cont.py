"""D26 费用守卫投影 + 闸门 1 投影 + A1 延续报告的预先写定读法（每个分支一个构造样例）。"""
import importlib.util, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from cot_compress.cost import budget_projection, gate1_projection

spec = importlib.util.spec_from_file_location("a1r", ROOT / "ops/analysis/a1_cont_report.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)


def test_budget_projection_a1cont_plan():
    # 计划：150 步 × 430 s + 3 次 eval × 33 min + 收尾 0.25 h，开跑时已花 0.3 h（拷贝 / 载入）
    p = budget_projection(0.3, 1.6, 150, 430, 3, 33 * 60, 0.25)
    assert abs(p["projected_usd"] - round((0.3 + (150 * 430 + 3 * 1980) / 3600 + 0.25) * 1.6, 2)) < 1e-9
    assert 31 < p["projected_usd"] < 35


def test_budget_projection_over_cap():
    assert budget_projection(5.0, 1.6, 150, 600, 3, 2400, 0.25)["projected_usd"] > 35


def test_gate1_projection():
    p = gate1_projection(0.5, 1.6, 5, 900, 6, 1800, 0.1)
    assert p["projected_usd"] == round((0.5 + (5 * 900 + 6 * 1800) / 3600 + 0.1) * 1.6, 2)
    assert gate1_projection(0.5, 1.6, 5, 900, 6, 1800, 0.1)["projected_usd"] < 15
    assert gate1_projection(0.5, 1.6, 5, 3600, 6, 3600, 0.1)["projected_usd"] > 15


def test_letter_reading_keep():
    assert A.letter_reading([0.78, 0.785, 0.79])[0] == "KEEP"
    assert A.letter_reading([0.77, 0.80, 0.775])[0] == "KEEP"       # 边界含


def test_letter_reading_switch():
    assert A.letter_reading([0.76, 0.72, 0.69])[0] == "SWITCH"


def test_letter_reading_neither():
    assert A.letter_reading([0.78, 0.76, 0.74])[0] == "NONE"        # 出了 77–80% 但最后一次不低于 70%
    assert A.letter_reading([0.78, 0.79, 0.805])[0] == "NONE"


def test_length_reading_branches():
    assert A.length_reading(5289.6, 4500, 0.912, 0.86)[0] == "TRADE"      # 掉 5.2pp
    assert A.length_reading(5289.6, 4500, 0.912, 0.862)[0] == "TRADE"     # 恰好 5pp
    assert A.length_reading(5289.6, 4500, 0.912, 0.87)[0] == "CHEAPER"    # 掉 4.2pp
    assert A.length_reading(5289.6, 4500, 0.912, 0.93)[0] == "CHEAPER"    # 准确率反升
    assert A.length_reading(5289.6, 5300, 0.912, 0.80)[0] == "NONE"       # 长度没降
