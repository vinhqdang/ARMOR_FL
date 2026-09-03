"""ARMOR-FL: Anytime-valid Robust Martingale-based Online Reweighting for
Federated Learning.

Per-round pipeline:
  1. Robust reference: coordinate-wise trimmed mean (or geometric median) of
     currently-trusted clients' updates -> m^t; robust scale MAD^t.
  2. Per-client population deviation z_k^t = ||theta_k^t - m^t|| / MAD^t, fed
     into a per-client population e-process (armor_fl.fl.eprocess.EProcess).
  3. Per-client self-referential drift e-process on the client's own local
     validation-loss trajectory (compares to its OWN history, not the
     population) -- decouples "drifting" (adapt) from "attacking" (downweight).
  3b. Per-client self-SHIFT e-process on the client's own z-score history:
     has this client moved relative to its OWN earlier position, as opposed
     to merely sitting off-center all along? This is what separates an attack
     that starts mid-run from static non-IID heterogeneity (see ArmorConfig).
  4. Trust-weighted aggregation: w_k^t = (n_k/N) * sigmoid(-beta * log E_pop_k^t).
     This smooth downweighting is NOT gated -- population evidence alone
     always reduces a diverging client's weight.
  5. Hard exclusion once log E_pop_k^t >= log(1/alpha_pop) AND either the
     shift e-process also fires or the client is a gross outlier
     (z >= gross_outlier_z). Excluded clients enter a probation window,
     after which their population and shift e-processes reset and they
     re-enter at reduced trust.

Both e-processes use a fixed, theoretically grounded null_mean of 1.0 rather
than any online/burn-in calibration -- see ArmorConfig's docstring note.
`burn_in_rounds` only delays ENFORCEMENT (flagging/exclusion), giving MAD and
own-loss-history estimates a few rounds to stabilize; evidence still
accumulates in the e-processes from round 1, so the anytime-valid guarantee
covers the full run.

Backbone-agnostic: operates purely on flattened parameter vectors, so it
composes with any client model (SE-1DSqueezeNet included).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from armor_fl.fl.eprocess import EProcess
from armor_fl.fl.robust_stats import (
    coordinate_trimmed_mean, geometric_median, weighted_average,
)


@dataclass
class ClientRecord:
    pop_eprocess: EProcess = field(default_factory=EProcess)
    drift_eprocess: EProcess = field(default_factory=EProcess)
    shift_eprocess: EProcess = field(default_factory=EProcess)
    excluded: bool = False
    probation_rounds_left: int = 0
    reinstated_count: int = 0
    loss_history: list[float] = field(default_factory=list)
    z_history: list[float] = field(default_factory=list)
    status: str = "trusted"  # trusted | drifting | suspect | excluded | probation


@dataclass
class ArmorConfig:
    alpha_pop: float = 0.05          # population e-process exclusion level
    alpha_drift: float = 0.05        # drift e-process flag level
    alpha_shift: float = 0.05        # self-shift e-process level (see note below)
    beta: float = 4.0                # trust-weight downweighting sharpness
    bet_fraction: float = 0.5        # fixed betting fraction (0, 1)
    burn_in_rounds: int = 3          # grace period before flags/exclusion act
    z_bound: float = 15.0            # a.s. bound on the population z-statistic
    drift_bound: float = 15.0        # a.s. bound on the drift statistic
    shift_bound: float = 15.0        # a.s. bound on the self-shift statistic
    probation_rounds: int = 3        # rounds before an excluded client can return
    reinstatement_trust_scale: float = 0.3  # initial weight scale on return
    center_method: str = "trimmed_mean"     # trimmed_mean | geometric_median
    trim_fraction: float = 0.2
    require_shift_for_exclusion: bool = True  # gate hard exclusion on the shift signal
    shift_baseline_rounds: int = 2   # own-z rounds used as the pre-attack baseline
    gross_outlier_z: float = 4.0     # z at/above which shift gating is bypassed

    # NOTE on null_mean=1.0: both statistics below are of the form
    # dist_i / median(dist_1..dist_K), and the median of a set divided by
    # itself has median exactly 1 by construction. So E_{H0}[X_t] = 1 is a
    # theoretically grounded, dataset-independent constant -- no per-client
    # or per-run calibration is needed (and, critically, none is learned
    # from the client's own behavior, which would let a client that is
    # already Byzantine during any "burn-in" calibrate itself as normal).
    #
    # NOTE on the self-shift e-process (`require_shift_for_exclusion`):
    # the population e-process alone cannot separate "persistently somewhat
    # off-center because this client's data is non-IID" from "attacking".
    # Any anytime-valid test will EVENTUALLY reject a client whose z-score
    # sits persistently above the population median, no matter how small
    # alpha_pop is -- that is the guarantee working as designed, not a
    # tuning failure. The loss-based drift e-process does not rescue this
    # case either: a persistently-heterogeneous honest client has a stable
    # OWN loss, so nothing looks like drift.
    #
    # The self-shift statistic therefore asks a different question:
    # has THIS client's own z-score changed relative to ITS OWN earlier
    # z-scores? Static heterogeneity (off-center from round 1, and staying
    # equally off-center) produces no shift; an attack that begins after
    # the baseline window does. Gating hard exclusion on this makes the
    # test one of CHANGE rather than of LEVEL.
    #
    # Honest limitation, stated here because it bounds the claim: a client
    # that is Byzantine from round 1 and perfectly constant thereafter
    # produces no shift signal either, and so is protected from hard
    # exclusion by this same gate (it is still smoothly downweighted via
    # the population e-process trust weight, which is not gated). Behavioural
    # data alone cannot distinguish that attacker from a merely-heterogeneous
    # honest client; doing so needs an assumption the data does not carry.
    # This design deliberately trades detection of always-on constant
    # attackers for not excluding honest non-IID clients.


@dataclass
class ClientUpdate:
    client_id: int
    vector: torch.Tensor
    n_k: int
    local_val_loss: float  # final local validation loss this round


@dataclass
class RoundResult:
    weights: dict[int, float]
    aggregated_vector: torch.Tensor
    z_scores: dict[int, float]
    log_e_pop: dict[int, float]
    log_e_drift: dict[int, float]
    log_e_shift: dict[int, float]
    status: dict[int, str]
    excluded_this_round: list[int]
    reinstated_this_round: list[int]
    mad: float
    burn_in_active: bool


class ArmorAggregator:
    def __init__(self, num_clients: int, config: ArmorConfig | None = None):
        self.cfg = config or ArmorConfig()
        self.records: dict[int, ClientRecord] = {
            cid: ClientRecord() for cid in range(num_clients)
        }
        self.round_idx = 0

    def _robust_center(self, vectors: list[torch.Tensor], ns: list[int]) -> torch.Tensor:
        if self.cfg.center_method == "geometric_median":
            return geometric_median(vectors, weights=[float(n) for n in ns])
        return coordinate_trimmed_mean(vectors, self.cfg.trim_fraction)

    def step(self, client_updates: list[ClientUpdate]) -> RoundResult:
        cfg = self.cfg
        self.round_idx += 1
        burn_in_active = self.round_idx <= cfg.burn_in_rounds

        weights: dict[int, float] = {}
        z_scores: dict[int, float] = {}
        log_e_pop: dict[int, float] = {}
        log_e_drift: dict[int, float] = {}
        log_e_shift: dict[int, float] = {}
        status: dict[int, str] = {}
        excluded_this_round: list[int] = []
        reinstated_this_round: list[int] = []

        # ---- pass 1: probation countdown / reinstatement, decide this round's
        # active set. An excluded client is skipped entirely below, so this
        # bookkeeping must happen BEFORE filtering, not inside the main loop. ----
        still_excluded_ids: set[int] = set()
        for u in client_updates:
            rec = self.records[u.client_id]
            if rec.excluded:
                rec.probation_rounds_left -= 1
                if rec.probation_rounds_left <= 0:
                    rec.excluded = False
                    rec.pop_eprocess.reset()
                    # The shift e-process must reset alongside the population one:
                    # otherwise its pre-exclusion evidence stays banked and the
                    # client is re-excluded on its first post-probation round,
                    # making probation a no-op.
                    rec.shift_eprocess.reset()
                    rec.reinstated_count += 1
                    reinstated_this_round.append(u.client_id)
                    rec.status = "probation"
                else:
                    still_excluded_ids.add(u.client_id)

        active = [u for u in client_updates if u.client_id not in still_excluded_ids]
        if len(active) < 2:
            active, still_excluded_ids = client_updates, set()  # never fully stall

        for u in client_updates:
            if u.client_id in still_excluded_ids:
                rec = self.records[u.client_id]
                status[u.client_id] = "excluded"
                log_e_pop[u.client_id] = rec.pop_eprocess.log_value
                log_e_drift[u.client_id] = rec.drift_eprocess.log_value
                log_e_shift[u.client_id] = rec.shift_eprocess.log_value
                weights[u.client_id] = 0.0

        # ---- pass 2: robust reference center / MAD over the active population ----
        vectors = [u.vector for u in active]
        ns = [u.n_k for u in active]
        center = self._robust_center(vectors, ns)
        dists = [torch.norm(u.vector - center).item() for u in active]
        mad = max(float(np.median(dists)) if dists else 1e-8, 1e-8)

        # ---- pass 3: e-processes + trust weighting for the active population ----
        for u, dist in zip(active, dists):
            rec = self.records[u.client_id]
            z = dist / mad
            rec.loss_history.append(u.local_val_loss)

            # ---- population e-process: fixed null_mean=1.0 (see ArmorConfig note) ----
            if not rec.pop_eprocess.is_calibrated():
                rec.pop_eprocess.calibrate(null_mean=1.0, bound=cfg.z_bound)
                rec.pop_eprocess.bet_fraction = cfg.bet_fraction
            rec.pop_eprocess.update(z)

            # ---- self-referential drift statistic: |own deviation| / own MAD,
            # against the client's OWN loss history (same median-normalization
            # trick, so null_mean=1.0 is grounded the same way as above). ----
            own_hist = rec.loss_history[:-1]
            if len(own_hist) >= 2:
                own_baseline = float(np.median(own_hist))
                own_mad = float(np.median([abs(v - own_baseline) for v in own_hist])) or 1e-3
                drift_stat = abs(u.local_val_loss - own_baseline) / own_mad
            else:
                drift_stat = 1.0  # not enough own history yet: neutral observation

            if not rec.drift_eprocess.is_calibrated():
                rec.drift_eprocess.calibrate(null_mean=1.0, bound=cfg.drift_bound)
                rec.drift_eprocess.bet_fraction = cfg.bet_fraction
            rec.drift_eprocess.update(drift_stat)

            # ---- self-shift statistic: has this client's OWN z-score changed
            # relative to its OWN pre-attack baseline? Static heterogeneity
            # (off-center from the start, equally off-center since) yields ~1
            # and never accumulates evidence; an attack starting after the
            # baseline window pushes this well above 1. See ArmorConfig's note. ----
            own_z_baseline = rec.z_history[:cfg.shift_baseline_rounds]
            if len(own_z_baseline) >= cfg.shift_baseline_rounds:
                base_level = float(np.median(own_z_baseline)) or 1e-3
                shift_stat = z / max(base_level, 1e-3)
            else:
                shift_stat = 1.0  # still inside the baseline window: neutral
            rec.z_history.append(z)

            if not rec.shift_eprocess.is_calibrated():
                rec.shift_eprocess.calibrate(null_mean=1.0, bound=cfg.shift_bound)
                rec.shift_eprocess.bet_fraction = cfg.bet_fraction
            rec.shift_eprocess.update(shift_stat)

            # ---- decision logic: decouple drift (adapt) from attack (downweight) ----
            pop_flag = (not burn_in_active) and rec.pop_eprocess.exceeds(cfg.alpha_pop)
            drift_flag = (not burn_in_active) and rec.drift_eprocess.exceeds(cfg.alpha_drift)

            if pop_flag and not drift_flag:
                rec.status = "suspect"  # Byzantine signature: diverges from population,
                                         # but locally self-consistent
            elif drift_flag and not pop_flag:
                rec.status = "drifting"  # moving with an evolving population consensus
            elif pop_flag and drift_flag:
                rec.status = "ambiguous"
            else:
                rec.status = "trusted"

            # w_k^t = n_k * sigmoid(-beta * log E_pop_k^t): smooth downweighting
            # (not a hard cutoff) as evidence of population divergence accumulates.
            trust_scale = 1.0 / (1.0 + math.exp(cfg.beta * rec.pop_eprocess.log_value))
            if rec.reinstated_count > 0 and rec.status != "trusted":
                trust_scale = min(trust_scale, cfg.reinstatement_trust_scale)
            w = u.n_k * trust_scale

            # Hard exclusion needs population-level evidence PLUS one of:
            #  (a) the client shifted from its own baseline (attack started
            #      mid-run -- the change-point case the shift e-process is for), or
            #  (b) the client is a gross outlier in absolute terms: sitting
            #      `gross_outlier_z` x further from the robust center than the
            #      population median does. Real non-IID heterogeneity moves a
            #      client's z modestly (empirically ~1-2x); an order-of-magnitude
            #      offset is not plausibly explained by data skew, so it is
            #      excluded even with no shift. This keeps always-on constant
            #      attackers detectable while still protecting merely-
            #      heterogeneous honest clients.
            shift_flag = (not cfg.require_shift_for_exclusion
                          or rec.shift_eprocess.exceeds(cfg.alpha_shift)
                          or z >= cfg.gross_outlier_z)
            if (rec.pop_eprocess.exceeds(cfg.alpha_pop) and shift_flag
                    and not burn_in_active):
                rec.excluded = True
                rec.probation_rounds_left = cfg.probation_rounds
                excluded_this_round.append(u.client_id)
                w = 0.0
                rec.status = "excluded"

            weights[u.client_id] = w
            z_scores[u.client_id] = z
            log_e_pop[u.client_id] = rec.pop_eprocess.log_value
            log_e_drift[u.client_id] = rec.drift_eprocess.log_value
            log_e_shift[u.client_id] = rec.shift_eprocess.log_value
            status[u.client_id] = rec.status

        contributing = [u for u in active if weights.get(u.client_id, 0.0) > 0]
        if not contributing:
            contributing = active  # never fully stall the federation
        agg_vec = weighted_average(
            [u.vector for u in contributing],
            [weights.get(u.client_id, u.n_k) for u in contributing],
        )

        return RoundResult(
            weights=weights, aggregated_vector=agg_vec, z_scores=z_scores,
            log_e_pop=log_e_pop, log_e_drift=log_e_drift,
            log_e_shift=log_e_shift, status=status,
            excluded_this_round=excluded_this_round,
            reinstated_this_round=reinstated_this_round, mad=mad,
            burn_in_active=burn_in_active,
        )
