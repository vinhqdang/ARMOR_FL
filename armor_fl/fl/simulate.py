"""Full FL simulation loop: data partitioning -> per-round local training with
optional attack injection -> aggregation (FedAvg / robust baseline / ARMOR-FL)
-> global evaluation. Single-machine, custom loop (no Flower) for full control
over the aggregation-layer statistics.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from armor_fl.data.partition import dirichlet_partition, iid_partition
from armor_fl.eval.metrics import detection_metrics
from armor_fl.fl import attacks as atk
from armor_fl.fl.aggregators import SimpleUpdate, build_aggregator
from armor_fl.fl.armor import ArmorAggregator, ArmorConfig, ClientUpdate
from armor_fl.fl.client import LocalTrainConfig, evaluate, local_train
from armor_fl.fl.robust_stats import flatten_state_dict, unflatten_to_state_dict


@dataclass
class SimulationConfig:
    num_clients: int = 10
    client_fraction: float = 0.6
    num_rounds: int = 30
    local_epochs: int = 20
    patience: int = 5
    lr: float = 0.01
    batch_size: int = 32
    dropout: float = 0.5
    non_iid_alpha: float | None = None  # None = IID
    aggregator: str = "fedavg"          # fedavg | trimmed_mean | coordinate_median
                                          # | krum | multi_krum | foolsgold | armor
    aggregator_kwargs: dict = field(default_factory=dict)
    armor_config: ArmorConfig | None = None
    attack_type: str | None = None      # None | label_flip | sign_flip | gaussian_noise
                                          # | free_rider | alie
    malicious_fraction: float = 0.0
    attack_start_round: int = 1
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1


@dataclass
class SimulationResult:
    config: SimulationConfig
    per_round_metrics: list[dict] = field(default_factory=list)
    detection: dict = field(default_factory=dict)
    wall_clock_seconds: float = 0.0
    final_model_state: dict | None = None


def _make_client_partitions(y: np.ndarray, cfg: SimulationConfig) -> list[np.ndarray]:
    if cfg.non_iid_alpha is None:
        return iid_partition(y, cfg.num_clients, seed=cfg.seed)
    return dirichlet_partition(y, cfg.num_clients, alpha=cfg.non_iid_alpha, seed=cfg.seed)


def run_simulation(model_factory, X: np.ndarray, y: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    num_classes: int, cfg: SimulationConfig) -> SimulationResult:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    client_partitions = _make_client_partitions(y, cfg)
    malicious_ids = (atk.assign_malicious_clients(cfg.num_clients, cfg.malicious_fraction,
                                                    seed=cfg.seed)
                     if cfg.malicious_fraction > 0 else set())

    global_model = model_factory()
    template_state = copy.deepcopy(global_model.state_dict())
    global_vector = flatten_state_dict(template_state)

    if cfg.aggregator == "armor":
        armor_cfg = cfg.armor_config or ArmorConfig()
        aggregator = ArmorAggregator(cfg.num_clients, armor_cfg)
    else:
        aggregator = build_aggregator(cfg.aggregator, **cfg.aggregator_kwargs)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    result = SimulationResult(config=cfg)
    first_excluded_round: dict[int, int] = {}
    start_time = time.time()

    for round_idx in range(1, cfg.num_rounds + 1):
        n_select = max(1, int(round(cfg.client_fraction * cfg.num_clients)))
        selected = rng.choice(cfg.num_clients, size=n_select, replace=False).tolist()
        attack_active = (cfg.attack_type is not None
                          and round_idx >= cfg.attack_start_round)

        client_deltas: dict[int, torch.Tensor] = {}
        client_n: dict[int, int] = {}
        client_losses: dict[int, float] = {}
        honest_delta_pool: list[torch.Tensor] = []

        for cid in selected:
            idx = client_partitions[cid]
            if len(idx) < 4:
                continue
            n_val = max(1, int(0.2 * len(idx)))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            if len(train_idx) == 0:
                continue

            Xc, yc = X_t[train_idx], y_t[train_idx]
            Xv, yv = X_t[val_idx], y_t[val_idx]

            is_malicious = attack_active and cid in malicious_ids
            if is_malicious and cfg.attack_type == "label_flip":
                yc = torch.tensor(
                    atk.label_flip(yc.numpy(), num_classes, seed=cfg.seed + round_idx),
                    dtype=torch.long,
                )

            local_model = model_factory()
            local_model.load_state_dict(
                unflatten_to_state_dict(global_vector, template_state))
            train_cfg = LocalTrainConfig(max_epochs=cfg.local_epochs, patience=cfg.patience,
                                          lr=cfg.lr, batch_size=cfg.batch_size,
                                          device=cfg.device)
            local_model, loss_traj, _ = local_train(local_model, Xc, yc, Xv, yv, train_cfg)
            local_vector = flatten_state_dict(local_model.state_dict())
            honest_delta = local_vector - global_vector
            honest_delta_pool.append(honest_delta)

            client_deltas[cid] = honest_delta
            client_n[cid] = len(train_idx)
            client_losses[cid] = loss_traj[-1] if loss_traj else float("nan")

        # ---- parameter-space attacks (applied after honest deltas collected,
        # so ALIE can see the honest pool this round -- an omniscient-attacker
        # assumption standard in the ALIE paper) ----
        if attack_active and cfg.attack_type in ("sign_flip", "gaussian_noise",
                                                   "free_rider", "alie"):
            for cid in selected:
                if cid not in malicious_ids or cid not in client_deltas:
                    continue
                honest_delta = client_deltas[cid]
                if cfg.attack_type == "sign_flip":
                    client_deltas[cid] = atk.sign_flip(honest_delta)
                elif cfg.attack_type == "gaussian_noise":
                    client_deltas[cid] = atk.gaussian_noise(honest_delta)
                elif cfg.attack_type == "free_rider":
                    client_deltas[cid] = atk.free_rider(honest_delta)
                elif cfg.attack_type == "alie" and len(honest_delta_pool) > 1:
                    client_deltas[cid] = atk.alie_attack(honest_delta_pool)

        if not client_deltas:
            continue  # degenerate round (all partitions too small); skip

        uploaded_vectors = {cid: global_vector + delta for cid, delta in client_deltas.items()}

        if cfg.aggregator == "armor":
            armor_updates = [
                ClientUpdate(client_id=cid, vector=uploaded_vectors[cid], n_k=client_n[cid],
                              local_val_loss=client_losses[cid])
                for cid in uploaded_vectors
            ]
            round_result = aggregator.step(armor_updates)
            global_vector = round_result.aggregated_vector
            for cid in round_result.excluded_this_round:
                first_excluded_round.setdefault(cid, round_idx)
            armor_diag = {
                "excluded_this_round": list(round_result.excluded_this_round),
                "reinstated_this_round": list(round_result.reinstated_this_round),
                "mad": round_result.mad,
                "n_active": len(round_result.weights) if round_result.weights else 0,
            }
        else:
            simple_updates = [
                SimpleUpdate(client_id=cid, vector=uploaded_vectors[cid], n_k=client_n[cid])
                for cid in uploaded_vectors
            ]
            global_vector = aggregator.aggregate(simple_updates)
            armor_diag = {}

        if round_idx % cfg.log_every == 0 or round_idx == cfg.num_rounds:
            global_model.load_state_dict(
                unflatten_to_state_dict(global_vector, template_state))
            metrics = evaluate(global_model, X_test_t, y_test_t, device=cfg.device)
            metrics.update({"round": round_idx, "n_selected": len(selected),
                             "n_contributed": len(client_deltas)})
            metrics.update(armor_diag)
            result.per_round_metrics.append(metrics)

    result.wall_clock_seconds = time.time() - start_time
    global_model.load_state_dict(unflatten_to_state_dict(global_vector, template_state))
    result.final_model_state = global_model.state_dict()

    all_ids = set(range(cfg.num_clients))
    ever_excluded = set(first_excluded_round.keys())
    result.detection = detection_metrics(malicious_ids, ever_excluded, all_ids)
    result.detection["first_excluded_round"] = first_excluded_round
    result.detection["malicious_ids"] = sorted(malicious_ids)
    return result
