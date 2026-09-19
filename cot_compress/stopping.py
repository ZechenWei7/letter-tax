"""停止判据与训练后残留检查（§4.2 / r3），纯逻辑，可单测；train.py 的 EvalCallback 调用。
收敛：连续两个 eval 区间 |ΔL_mean| / L_后一次 < 5% 且 |Δacc| < 4pp（分母用**后一次** eval；step-0 eval 计入历史）。
残留检查（C_rand 豁免；需要冻结 direct）：
  生效条件：带轨迹准确率 acc_trace ≥ frozen_direct + 10pp 首次满足之后（锁存）；
  触发条件：direct_now ≥ acc_trace − (acc_trace − frozen_direct)/2，连续两次 eval → 停 run，标不可解释。
指定 checkpoint（designated_ckpt）：收敛停止 → 停止时的 step；残留停止 → 该 step（标不可解释）；跑满 max_steps 未收敛 → step = max_steps（reason="max_steps"）。
"""
from __future__ import annotations

def interval_small(prev: tuple, cur: tuple, dl_rel: float = 0.05, dacc: float = 0.04) -> bool:
    (lp, ap), (l, a) = prev, cur
    return abs(l - lp) / max(l, 1) < dl_rel and abs(a - ap) < dacc

def converged(hist: list[tuple], dl_rel: float = 0.05, dacc: float = 0.04) -> bool:
    if len(hist) < 3: return False
    h0, h1, h2 = hist[-3:]
    return interval_small(h0, h1, dl_rel, dacc) and interval_small(h1, h2, dl_rel, dacc)

class ResidualCheck:
    def __init__(self, frozen_direct: float | None, arm: str = "", activate_margin: float = 0.10):
        self.frozen_direct = None if arm == "Crand" else frozen_direct          # C_rand 豁免
        self.margin, self.active, self.streak = activate_margin, False, 0
    def update(self, acc_trace: float, direct_now: float | None) -> dict:
        """返回 dict(active, low, streak, stop, threshold)。"""
        fd = self.frozen_direct
        if fd is None or direct_now is None:
            return dict(active=False, low=False, streak=0, stop=False, threshold=None)
        if acc_trace >= fd + self.margin: self.active = True
        thr = acc_trace - (acc_trace - fd) / 2
        low = self.active and direct_now >= thr
        self.streak = self.streak + 1 if low else 0
        return dict(active=self.active, low=low, streak=self.streak, stop=self.streak >= 2, threshold=thr)

def designated_ckpt(step: int, max_steps: int, stopped_converged: bool, stopped_residual: bool) -> dict:
    reason = "uninterpretable" if stopped_residual else ("converged" if stopped_converged else ("max_steps" if step >= max_steps else "interrupted"))
    return dict(step=step, reason=reason, max_steps=max_steps)
