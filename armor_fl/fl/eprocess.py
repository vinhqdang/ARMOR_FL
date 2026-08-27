"""Generic anytime-valid sequential test (e-process / testing-by-betting).

Implements the bounded-mean betting e-value of Waudby-Smith & Ramdas (2020),
"Estimating means of bounded random variables by betting", specialized to a
fixed (non-adaptive) betting fraction rather than their online-Newton-step
tuning -- a simplification chosen for implementation robustness, and one that
preserves the exact martingale property as long as `null_mean` (m0) is a
FIXED, pre-registered constant rather than something re-estimated from the
same stream being tested.

Given a bounded observation X_t in [0, bound] with hypothesized null mean m0
(E_{H0}[X_t] = m0), the per-step e-value is

    e_t = 1 + lam * (X_t - m0),   lam = bet_fraction / m0   (m0 > 0)

which satisfies E_{H0}[e_t] = 1 exactly when E_{H0}[X_t] = m0, so the running
product E_t = prod_{s<=t} e_s is a nonnegative martingale under H0 with
E_{H0}[E_t] = 1 for every t. By Ville's inequality:

    P_{H0}( sup_{t<=T} E_t >= 1/alpha ) <= alpha   for every T,

i.e. thresholding at 1/alpha gives an anytime-valid test at level alpha no
matter how many rounds are monitored or when you stop looking -- unlike a
fixed-checkpoint z-test, whose type-I error inflates the more rounds you peek
at.

Callers must pass a `null_mean` that is fixed and NOT estimated from the same
client's own stream (see armor_fl.fl.armor for why: a client that is already
misbehaving during any self-calibration window would calibrate itself as
"normal"). armor_fl.fl.armor uses a theoretically grounded constant (1.0) for
exactly this reason, rather than any form of online or burn-in calibration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EProcess:
    null_mean: float | None = None       # m0; None until calibrated
    bound: float = 10.0                  # a.s. upper bound on X_t (post-clip)
    bet_fraction: float = 0.5            # c in lam = c / m0, c in (0, 1)
    log_value: float = 0.0               # log E_t, running (can go negative)
    n_obs: int = 0

    def calibrate(self, null_mean: float, bound: float | None = None) -> None:
        self.null_mean = max(null_mean, 1e-6)
        if bound is not None:
            self.bound = bound

    def is_calibrated(self) -> bool:
        return self.null_mean is not None

    def update(self, x: float) -> float:
        """Feed one observation, return this step's e-value (not cumulative)."""
        assert self.is_calibrated(), "EProcess.calibrate() must be called first"
        x_clipped = min(max(x, 0.0), self.bound)
        lam = self.bet_fraction / self.null_mean
        # Keep e_t >= 0 for x in [0, bound]: worst case x=0 -> 1 - lam*m0 = 1 - bet_fraction >= 0.
        e_t = 1.0 + lam * (x_clipped - self.null_mean)
        e_t = max(e_t, 1e-12)
        self.log_value += math.log(e_t)
        self.n_obs += 1
        return e_t

    @property
    def value(self) -> float:
        """E_t on the natural (non-log) scale -- can overflow for long/strong
        runs, prefer `log_value` for thresholding."""
        return math.exp(self.log_value)

    def exceeds(self, alpha: float) -> bool:
        """True once E_t >= 1/alpha, i.e. the anytime-valid rejection rule."""
        return self.log_value >= -math.log(alpha)

    def reset(self) -> None:
        self.log_value = 0.0
        self.n_obs = 0
