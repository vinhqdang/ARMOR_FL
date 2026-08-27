"""Baseline aggregation strategies, sharing one interface with ArmorAggregator
so the experiment runner can swap strategies without branching logic.

All operate on flattened parameter vectors (see robust_stats.flatten_state_dict).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from armor_fl.fl.robust_stats import (
    coordinate_median, coordinate_trimmed_mean, krum_select, weighted_average,
)


@dataclass
class SimpleUpdate:
    client_id: int
    vector: torch.Tensor
    n_k: int


class FedAvgAggregator:
    """Paper's Algorithm 3: size-weighted average, no robustness."""
    name = "FedAvg"

    def aggregate(self, updates: list[SimpleUpdate]) -> torch.Tensor:
        return weighted_average([u.vector for u in updates], [u.n_k for u in updates])


class TrimmedMeanAggregator:
    name = "TrimmedMean"

    def __init__(self, trim_fraction: float = 0.2):
        self.trim_fraction = trim_fraction

    def aggregate(self, updates: list[SimpleUpdate]) -> torch.Tensor:
        return coordinate_trimmed_mean([u.vector for u in updates], self.trim_fraction)


class CoordinateMedianAggregator:
    name = "CoordinateMedian"

    def aggregate(self, updates: list[SimpleUpdate]) -> torch.Tensor:
        return coordinate_median([u.vector for u in updates])


class KrumAggregator:
    name = "Krum"

    def __init__(self, num_byzantine_assumed: int = 2, multi: int = 1):
        self.f = num_byzantine_assumed
        self.multi = multi

    def aggregate(self, updates: list[SimpleUpdate]) -> torch.Tensor:
        vectors = [u.vector for u in updates]
        selected = krum_select(vectors, self.f, multi=self.multi)
        chosen = [updates[i] for i in selected]
        return weighted_average([u.vector for u in chosen], [u.n_k for u in chosen])


class FoolsGoldAggregator:
    """Cosine-similarity clustering defense (Fung et al. 2018): clients whose
    update DIRECTIONS are unusually similar to others (sign of coordinated/
    sybil poisoning) are downweighted via a history-pairwise-cosine penalty.
    Simplified single-round variant (no cross-round history) since baseline
    updates here are single flattened deltas per round, not gradient
    histories -- documented as a simplification relative to the original
    multi-round FoolsGold."""
    name = "FoolsGold"

    def __init__(self, epsilon: float = 1e-5, kappa: float = 1.0):
        self.epsilon = epsilon
        self.kappa = kappa

    def aggregate(self, updates: list[SimpleUpdate]) -> torch.Tensor:
        vectors = torch.stack([u.vector for u in updates], dim=0)
        n = vectors.shape[0]
        normed = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cos_sim = normed @ normed.T
        cos_sim.fill_diagonal_(0.0)
        max_cs, _ = cos_sim.max(dim=1)

        # Pardoning: rescale by the ratio of one's own max similarity to
        # others', so clients that are simply similar to a rightly-trusted
        # majority aren't punished for it.
        for i in range(n):
            for j in range(n):
                if i != j and max_cs[j] > max_cs[i]:
                    cos_sim[i, j] *= max_cs[i] / max_cs[j].clamp_min(1e-12)
        v = 1.0 - cos_sim.max(dim=1).values.clamp(0, 1 - self.epsilon)
        v = v / v.max().clamp_min(1e-12)
        v = torch.clamp(self.kappa * (torch.log(v / (1 - v).clamp_min(1e-12) + 1e-12) + 0.5),
                         0.0, 1.0)
        weights = [float(v[i]) * updates[i].n_k for i in range(n)]
        if sum(weights) <= 0:
            weights = [u.n_k for u in updates]
        return weighted_average([u.vector for u in updates], weights)


def build_aggregator(name: str, **kwargs):
    registry = {
        "fedavg": FedAvgAggregator,
        "trimmed_mean": TrimmedMeanAggregator,
        "coordinate_median": CoordinateMedianAggregator,
        "krum": KrumAggregator,
        "multi_krum": lambda: KrumAggregator(multi=kwargs.get("multi", 3)),
        "foolsgold": FoolsGoldAggregator,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unknown aggregator '{name}', choices: {list(registry)}")
    ctor = registry[key]
    try:
        return ctor(**kwargs)
    except TypeError:
        return ctor()
