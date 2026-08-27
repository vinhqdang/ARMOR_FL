"""Validates the anytime-valid guarantee (Ville's inequality) empirically:
under a pure-null stream, P(ever exceed 1/alpha within T rounds) <= alpha,
and this should NOT inflate as T grows -- which is exactly the property a
naive per-round z-test threshold does not have."""
import math
import random

import numpy as np
import pytest

from armor_fl.fl.eprocess import EProcess


def run_one_null_stream(rng: np.random.Generator, T: int, alpha: float,
                         null_mean: float = 1.0, bound: float = 8.0) -> bool:
    """Simulate T rounds of exponential(mean=null_mean) observations (a
    plausible shape for a nonnegative robust z-score under H0), clipped to
    `bound`. Returns True iff the e-process ever exceeded 1/alpha."""
    ep = EProcess(bet_fraction=0.5)
    ep.calibrate(null_mean=null_mean, bound=bound)
    for _ in range(T):
        x = float(rng.exponential(scale=null_mean))
        ep.update(x)
        if ep.exceeds(alpha):
            return True
    return False


@pytest.mark.parametrize("T", [10, 50, 200, 1000])
def test_false_exclusion_rate_does_not_inflate_with_more_rounds(T):
    rng = np.random.default_rng(42)
    alpha = 0.05
    n_trials = 2000
    false_exclusions = sum(
        run_one_null_stream(rng, T, alpha) for _ in range(n_trials)
    )
    rate = false_exclusions / n_trials
    # Ville's inequality gives P(ever exceed) <= alpha for ALL T; allow slack
    # for Monte Carlo noise (Wilson-ish bound at n=2000, alpha=0.05).
    assert rate <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / n_trials), (
        f"T={T}: empirical false-exclusion rate {rate:.4f} exceeds alpha={alpha} "
        f"by more than Monte Carlo noise -- anytime-valid guarantee violated"
    )


def test_false_exclusion_rate_flat_across_horizons():
    """The key differentiator vs fixed-checkpoint peeking: rate at T=1000
    should not be meaningfully larger than at T=10."""
    rng = np.random.default_rng(7)
    alpha = 0.05
    n_trials = 3000
    rate_short = sum(run_one_null_stream(rng, 10, alpha) for _ in range(n_trials)) / n_trials
    rate_long = sum(run_one_null_stream(rng, 1000, alpha) for _ in range(n_trials)) / n_trials
    assert rate_long <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / n_trials)
    assert rate_short <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / n_trials)


def test_detects_shifted_alternative_reasonably_fast():
    """Sanity check for Theorem 2 (detection speed): under a mean-shifted
    alternative, the e-process should cross the threshold well within a
    modest number of rounds, most of the time."""
    rng = np.random.default_rng(0)
    alpha = 0.05
    null_mean = 1.0
    shifted_mean = 3.0  # attacker: 3x the null deviation on average
    bound = 15.0
    T_max = 100

    detected = 0
    detect_rounds = []
    n_trials = 500
    for _ in range(n_trials):
        ep = EProcess(bet_fraction=0.5)
        ep.calibrate(null_mean=null_mean, bound=bound)
        for t in range(1, T_max + 1):
            x = float(rng.exponential(scale=shifted_mean))
            ep.update(x)
            if ep.exceeds(alpha):
                detected += 1
                detect_rounds.append(t)
                break

    detection_rate = detected / n_trials
    assert detection_rate > 0.9, f"only {detection_rate:.2%} detected within {T_max} rounds"
    assert np.median(detect_rounds) < 30
