"""Vector flatten/unflatten helpers and robust-aggregation primitives shared by
the FedAvg baseline, the classic robust baselines (Krum, trimmed mean, ...),
and ARMOR-FL's robust reference center."""
from __future__ import annotations

import torch


def flatten_state_dict(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.detach().reshape(-1).float() for v in state_dict.values()])


def unflatten_to_state_dict(vector: torch.Tensor,
                             template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    offset = 0
    for k, v in template.items():
        n = v.numel()
        out[k] = vector[offset:offset + n].reshape(v.shape).to(v.dtype)
        offset += n
    return out


def weighted_average(vectors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum().clamp_min(1e-12)
    stacked = torch.stack(vectors, dim=0)
    return (stacked * w.unsqueeze(-1)).sum(dim=0)


def coordinate_trimmed_mean(vectors: list[torch.Tensor], trim_fraction: float) -> torch.Tensor:
    """Coordinate-wise trimmed mean: drop the top/bottom `trim_fraction` of
    values per coordinate before averaging."""
    stacked = torch.stack(vectors, dim=0)  # (K, D)
    k = stacked.shape[0]
    n_trim = int(k * trim_fraction)
    if n_trim == 0 or k - 2 * n_trim <= 0:
        return stacked.mean(dim=0)
    sorted_vals, _ = torch.sort(stacked, dim=0)
    trimmed = sorted_vals[n_trim: k - n_trim]
    return trimmed.mean(dim=0)


def coordinate_median(vectors: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(vectors, dim=0)
    return stacked.median(dim=0).values


def geometric_median(vectors: list[torch.Tensor], weights: list[float] | None = None,
                      n_iter: int = 50, eps: float = 1e-8) -> torch.Tensor:
    """Weiszfeld's algorithm for the (weighted) geometric median."""
    stacked = torch.stack(vectors, dim=0)  # (K, D)
    if weights is None:
        weights = [1.0] * len(vectors)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum().clamp_min(1e-12)

    y = (stacked * w.unsqueeze(-1)).sum(dim=0)
    for _ in range(n_iter):
        dists = torch.norm(stacked - y.unsqueeze(0), dim=1).clamp_min(eps)
        inv = w / dists
        y_new = (stacked * inv.unsqueeze(-1)).sum(dim=0) / inv.sum().clamp_min(eps)
        if torch.norm(y_new - y) < eps:
            y = y_new
            break
        y = y_new
    return y


def krum_select(vectors: list[torch.Tensor], num_byzantine_assumed: int,
                 multi: int = 1) -> list[int]:
    """(Multi-)Krum: score each vector by the sum of squared distances to its
    (k - f - 2) nearest neighbors; return indices of the `multi` lowest-score
    vectors. `num_byzantine_assumed` is the attacker-count budget `f`."""
    k = len(vectors)
    f = min(num_byzantine_assumed, max(0, (k - 3) // 2))
    n_neighbors = max(1, k - f - 2)

    dist_matrix = torch.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                dist_matrix[i, j] = torch.norm(vectors[i] - vectors[j]) ** 2

    scores = []
    for i in range(k):
        d = torch.sort(dist_matrix[i])[0]
        # exclude self-distance (0 at position 0)
        scores.append(d[1:1 + n_neighbors].sum().item())
    order = sorted(range(k), key=lambda i: scores[i])
    return order[:max(1, multi)]
