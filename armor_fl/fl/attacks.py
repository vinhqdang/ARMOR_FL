"""Client-side attack simulation for the FL robustness experiments.

Attacks operate on a client's local dataset (label-flipping) or on its
trained update vector before upload (the parameter-space attacks). Each
attack function is applied only to clients selected as malicious for the run.
"""
from __future__ import annotations

import numpy as np
import torch


def label_flip(y: np.ndarray, num_classes: int, seed: int = 0,
                target_shift: int = 1) -> np.ndarray:
    """Deterministic label flip: y -> (y + target_shift) mod num_classes.
    Simulates a poisoned client training on systematically mislabeled data."""
    return (y + target_shift) % num_classes


def sign_flip(honest_delta: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Malicious DELTA (local - previous_global) = -scale * honest delta,
    i.e. push the global model in the opposite direction of genuine learning.
    Caller uploads `previous_global_vector + malicious_delta`."""
    return -scale * honest_delta


def gaussian_noise(honest_delta: torch.Tensor, sigma: float = 1.0,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Malicious delta = pure Gaussian noise, norm-matched to the honest
    delta so it isn't trivially caught by a magnitude-only check."""
    noise = torch.randn(honest_delta.shape, generator=generator, device=honest_delta.device)
    return noise * sigma * honest_delta.norm() / max(noise.norm().item(), 1e-8)


def free_rider(template_delta: torch.Tensor, jitter_sigma: float = 1e-3,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """Lazy client: near-zero delta (tiny noise only), contributing no real
    training signal while still collecting reward / avoiding
    detection-by-total-inactivity."""
    return torch.randn(template_delta.shape, generator=generator,
                        device=template_delta.device) * jitter_sigma


def alie_attack(honest_deltas: list[torch.Tensor], z_max: float = 1.5) -> torch.Tensor:
    """A Little Is Enough (Baruch et al. 2019): estimate the coordinate-wise
    mean/std across the CURRENT round's honest deltas and shift just inside
    the natural spread (mean - z_max * std), a bounded perturbation designed
    to evade simple range/distance-based outlier checks while still biasing
    the aggregate. Requires visibility into other honest clients' updates
    this round (an omniscient-attacker assumption standard in the ALIE paper).
    """
    stacked = torch.stack(honest_deltas, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    return mean - z_max * std


ATTACK_TYPES = ["label_flip", "sign_flip", "gaussian_noise", "free_rider", "alie"]


def assign_malicious_clients(num_clients: int, malicious_fraction: float,
                              seed: int = 0) -> set[int]:
    rng = np.random.default_rng(seed)
    n_malicious = round(num_clients * malicious_fraction)
    return set(rng.choice(num_clients, size=n_malicious, replace=False).tolist())
