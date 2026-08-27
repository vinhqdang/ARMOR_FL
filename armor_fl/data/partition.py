"""Client data partitioning: IID and Dirichlet non-IID (paper Sec. 5.2.2)."""
from __future__ import annotations

import numpy as np


def iid_partition(y: np.ndarray, num_clients: int, seed: int = 0) -> list[np.ndarray]:
    """Randomly shuffle and evenly split indices across clients."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return [chunk for chunk in np.array_split(idx, num_clients)]


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float,
                         seed: int = 0, min_overlap_frac: float = 0.02,
                         min_samples_per_client: int = 20) -> list[np.ndarray]:
    """Dirichlet-based non-IID partition (Hsu et al. 2019), matching paper Sec 5.2.2:
    - concentration alpha controls skew (smaller alpha = more skewed)
    - proportional sampling keeps per-client sample counts roughly balanced
    - a small amount of uniform overlap is mixed in so no client is fully
      missing a class (paper: "slight data overlap ... to prevent certain
      labels from being completely absent on specific clients")
    """
    rng = np.random.default_rng(seed)
    num_classes = int(y.max()) + 1
    client_idx: list[list[int]] = [[] for _ in range(num_clients)]

    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    for c, idxs in enumerate(class_indices):
        if len(idxs) == 0:
            continue
        idxs = rng.permutation(idxs)
        n_overlap = int(len(idxs) * min_overlap_frac)
        overlap_idxs, skewed_idxs = idxs[:n_overlap], idxs[n_overlap:]

        # Uniform overlap slice: every client gets an equal share.
        if n_overlap > 0:
            for i, chunk in enumerate(np.array_split(overlap_idxs, num_clients)):
                client_idx[i].extend(chunk.tolist())

        # Dirichlet-skewed slice.
        if len(skewed_idxs) > 0:
            proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
            counts = (proportions * len(skewed_idxs)).astype(int)
            counts[-1] = len(skewed_idxs) - counts[:-1].sum()  # fix rounding
            start = 0
            for i, cnt in enumerate(counts):
                client_idx[i].extend(skewed_idxs[start:start + cnt].tolist())
                start += cnt

    result = [np.array(sorted(idx), dtype=np.int64) for idx in client_idx]

    # Guarantee a floor sample count per client (rare with real datasets, but
    # protects against a degenerate empty-client edge case at very low alpha).
    for i, idx in enumerate(result):
        if len(idx) < min_samples_per_client:
            donor = int(np.argmax([len(r) for r in result]))
            n_needed = min_samples_per_client - len(idx)
            if donor != i and len(result[donor]) > n_needed:
                moved = rng.choice(result[donor], size=n_needed, replace=False)
                result[donor] = np.setdiff1d(result[donor], moved)
                result[i] = np.concatenate([idx, moved])
    return result


def client_class_histogram(y: np.ndarray, client_indices: list[np.ndarray],
                            num_classes: int) -> np.ndarray:
    """(num_clients, num_classes) count matrix, useful for logging/plots."""
    hist = np.zeros((len(client_indices), num_classes), dtype=np.int64)
    for i, idx in enumerate(client_indices):
        for c in range(num_classes):
            hist[i, c] = int((y[idx] == c).sum())
    return hist
