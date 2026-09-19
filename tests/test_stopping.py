"""r3：停止判据（分母用后一次 eval）、残留检查（锁存生效、连续两次、C_rand 豁免）、designated_ckpt。"""
from cot_compress.stopping import converged, interval_small, ResidualCheck, designated_ckpt

def test_converged_uses_later_eval_as_denominator():
    assert interval_small((1000, 0.70), (1052, 0.70))          # 52/1052 = 4.94% < 5%（用前一次作分母则 5.2%）
    assert not interval_small((1052, 0.70), (1000, 0.70))      # 52/1000 = 5.2%
    assert not interval_small((1000, 0.70), (1000, 0.74)) and interval_small((1000, 0.70), (1000, 0.739))
    assert not converged([(1000, 0.7), (990, 0.7)])
    assert converged([(3000, 0.7), (1000, 0.70), (990, 0.71), (985, 0.70)]) and not converged([(3000, 0.7), (1000, 0.7), (990, 0.7)])
    assert converged([(1000, 0.7), (1000, 0.7), (1000, 0.7)])  # step-0 eval 计入：最早 step 100 可停

def test_residual_check_latched_and_two_in_a_row():
    rc = ResidualCheck(frozen_direct=0.02, arm="A")
    assert rc.update(0.10, 0.09)["active"] is False            # acc_trace < 0.02 + 0.10 → 未生效，不计
    r = rc.update(0.70, 0.30); assert r["active"] and not r["low"] and abs(r["threshold"] - 0.36) < 1e-9
    r = rc.update(0.70, 0.36); assert r["low"] and r["streak"] == 1 and not r["stop"]     # direct ≥ acc − (acc − fd)/2
    r = rc.update(0.70, 0.20); assert r["streak"] == 0
    rc.update(0.60, 0.40); r = rc.update(0.11, 0.08)           # 锁存：acc 回落后仍生效；阈值 0.11 − 0.045 = 0.065
    assert r["active"] and r["low"] and r["stop"]

def test_residual_exemptions():
    assert ResidualCheck(0.02, arm="Crand").update(0.7, 0.7)["stop"] is False and ResidualCheck(0.02, arm="Crand").frozen_direct is None
    assert ResidualCheck(None, arm="B").update(0.7, 0.7)["active"] is False
    assert ResidualCheck(0.02, arm="B").update(0.7, None)["stop"] is False

def test_designated_ckpt():
    assert designated_ckpt(400, 400, False, False) == dict(step=400, reason="max_steps", max_steps=400)
    assert designated_ckpt(250, 400, True, False)["reason"] == "converged" and designated_ckpt(150, 400, False, True)["reason"] == "uninterpretable"
