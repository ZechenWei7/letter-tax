from cot_compress.style import rewrite_style, positions_per_step, style_summary

FULL = """Let me track. swap 3 5: 1U 2U 5U 4U 3U 6U 7U 8U
flip 2: 1U 2D 5U 4U 3U 6U 7U 8U
swap 1 8: 8U 2D 5U 4U 3U 6U 7U 1U
flip 4: 8U 2D 5U 4D 3U 6U 7U 1U"""
CHANGED = """swap 3 5: now position 3 = 5, position 5 = 3.
flip 2: position 2 is down.
swap 1 8: position 1 = 8, position 8 = 1.
flip 4: position 4 down."""

def test_positions_per_step():
    assert positions_per_step(FULL, 8) == [8, 8, 8, 8]
    assert positions_per_step(CHANGED, 8) == [2, 1, 2, 1]
    assert positions_per_step("no ops here", 8) == []

def test_rewrite_style():
    assert rewrite_style(FULL, 8)["style"] == "full_rewrite"
    assert rewrite_style(CHANGED, 8)["style"] == "changed_only"
    assert rewrite_style("swap 1 2: hmm", 8)["style"] == "unparsed"
    mixed = "swap 1 2: 2U 1U 3U 4U 5U 6U 7U 8U\nflip 3: position 3 down\nswap 4 5: position 4 = 5, 5 = 4\nflip 1: pos 1 down"
    assert rewrite_style(mixed, 8)["style"] == "mixed"
    s = style_summary([FULL, CHANGED], 8); assert s["style_frac"]["full_rewrite"] == 0.5 and s["n"] == 2
