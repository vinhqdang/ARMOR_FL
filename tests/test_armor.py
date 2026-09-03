"""Integration tests for ArmorAggregator on synthetic client updates:
verifies the two headline claims -- (1) a persistently Byzantine client gets
downweighted and eventually excluded while benign clients keep near-full
trust, and (2) a client that is merely drifting locally (unstable own loss,
but parameter updates that stay close to the honest population) is NOT
excluded, i.e. drift is decoupled from attack."""
import math

import torch

from armor_fl.fl.armor import ArmorAggregator, ArmorConfig, ClientUpdate

D = 40
NUM_CLIENTS = 10
ROUNDS = 25


def make_config(**overrides) -> ArmorConfig:
    base = dict(alpha_pop=0.05, alpha_drift=0.05, beta=4.0, burn_in_rounds=3,
                probation_rounds=3)
    base.update(overrides)
    return ArmorConfig(**base)


def test_byzantine_client_gets_excluded_benign_clients_mostly_survive():
    """Statistical, multi-seed check (mirrors test_eprocess.py's style): the
    byzantine client should be excluded almost every trial, and the benign
    false-exclusion rate across trials should stay close to alpha_pop -- a
    single seed can legitimately misfire since alpha_pop=0.05 does not mean
    zero false positives, it means a controlled rate of them."""
    cfg = make_config()
    n_trials = 40
    byzantine_id = 3
    byzantine_excluded_count = 0
    benign_false_exclusions = 0
    benign_client_rounds = 0

    for trial in range(n_trials):
        torch.manual_seed(trial)
        agg = ArmorAggregator(NUM_CLIENTS, cfg)
        excluded_ever = set()
        for r in range(ROUNDS):
            updates = []
            for cid in range(NUM_CLIENTS):
                if cid == byzantine_id:
                    vec = torch.randn(D) * 1.0 + 8.0
                    loss = 0.3 + 0.01 * torch.randn(1).item()
                else:
                    vec = torch.randn(D) * 1.0
                    loss = 0.5 + 0.05 * torch.randn(1).item()
                updates.append(ClientUpdate(client_id=cid, vector=vec, n_k=100,
                                             local_val_loss=loss))
            result = agg.step(updates)
            excluded_ever.update(result.excluded_this_round)
        if byzantine_id in excluded_ever:
            byzantine_excluded_count += 1
        for cid in range(NUM_CLIENTS):
            if cid == byzantine_id:
                continue
            benign_client_rounds += 1
            if cid in excluded_ever:
                benign_false_exclusions += 1

    detection_rate = byzantine_excluded_count / n_trials
    false_exclusion_rate = benign_false_exclusions / benign_client_rounds
    assert detection_rate > 0.9, f"byzantine detected in only {detection_rate:.0%} of trials"
    # Generous slack: alpha_pop bounds the false-exclusion probability PER
    # client-run, but with correlated z-scores across clients within a round
    # the realized rate can deviate somewhat from a naive iid bound.
    assert false_exclusion_rate < 3 * cfg.alpha_pop, (
        f"benign false-exclusion rate {false_exclusion_rate:.3f} far exceeds "
        f"alpha_pop={cfg.alpha_pop}"
    )


def test_drifting_but_honest_client_is_not_excluded():
    torch.manual_seed(1)
    agg = ArmorAggregator(NUM_CLIENTS, make_config())
    drifting_id = 5

    excluded_ever = set()
    statuses_seen = set()

    for r in range(ROUNDS):
        updates = []
        for cid in range(NUM_CLIENTS):
            vec = torch.randn(D) * 1.0  # ALL clients stay in the honest population
            if cid == drifting_id:
                # own local loss trends upward over rounds (simulates a new
                # local attack pattern the client is adapting to) -- but its
                # parameter update stays consistent with everyone else.
                loss = 0.3 + 0.05 * r + 0.02 * torch.randn(1).item()
            else:
                loss = 0.5 + 0.05 * torch.randn(1).item()
            updates.append(ClientUpdate(client_id=cid, vector=vec, n_k=100,
                                         local_val_loss=loss))
        result = agg.step(updates)
        excluded_ever.update(result.excluded_this_round)
        statuses_seen.add(result.status[drifting_id])

    assert drifting_id not in excluded_ever, (
        "a merely-drifting (but honest-update) client was wrongly excluded"
    )
    assert "drifting" in statuses_seen, (
        f"drift e-process never flagged the drifting client; saw statuses {statuses_seen}"
    )


def test_persistently_heterogeneous_honest_clients_are_not_excluded():
    """Regression test for the false-positive mechanism found on real data:
    clients whose updates sit persistently farther from the robust center
    (ordinary non-IID heterogeneity -- NOT attacks, and NOT changing over
    time) must not be hard-excluded.

    Before the self-shift gate, the population e-process would eventually
    exclude every one of these, because an anytime-valid test is guaranteed
    to reject any client whose z-score sits persistently above the median --
    that is the guarantee working as designed, which is exactly why level
    alone cannot be the exclusion criterion. The loss-based drift e-process
    does not catch this either: these clients' own losses are stable, so
    nothing looks like drift.
    """
    torch.manual_seed(7)
    agg = ArmorAggregator(NUM_CLIENTS, make_config())
    # Clients 5-9 are consistently ~2x more scattered than 0-4, from round 1
    # onward, and never change. All are honest.
    scatter = {cid: (1.0 if cid < 5 else 2.2) for cid in range(NUM_CLIENTS)}

    excluded_ever = set()
    for r in range(ROUNDS):
        updates = []
        shared_direction = torch.randn(D) * 0.1
        for cid in range(NUM_CLIENTS):
            vec = shared_direction + torch.randn(D) * 0.05 * scatter[cid]
            # own loss is stable w.r.t. the client's OWN history
            loss = 1.0 + 0.1 * scatter[cid] + 0.02 * torch.randn(1).item()
            updates.append(ClientUpdate(client_id=cid, vector=vec, n_k=100,
                                         local_val_loss=loss))
        result = agg.step(updates)
        excluded_ever.update(result.excluded_this_round)

    assert not excluded_ever, (
        f"honest but persistently-heterogeneous clients were excluded: "
        f"{sorted(excluded_ever)} (none of them are attacking)"
    )


def test_reinstatement_resets_after_probation():
    torch.manual_seed(2)
    cfg = make_config(probation_rounds=2)
    agg = ArmorAggregator(NUM_CLIENTS, cfg)
    byzantine_id = 0

    reinstated_seen = False
    for r in range(30):
        updates = []
        for cid in range(NUM_CLIENTS):
            if cid == byzantine_id and r < 10:
                vec = torch.randn(D) * 1.0 + 10.0
            else:
                vec = torch.randn(D) * 1.0
            loss = 0.5 + 0.05 * torch.randn(1).item()
            updates.append(ClientUpdate(client_id=cid, vector=vec, n_k=100,
                                         local_val_loss=loss))
        result = agg.step(updates)
        if result.reinstated_this_round:
            reinstated_seen = True

    assert reinstated_seen, "excluded client was never reinstated after probation"
