"""Robustness-evaluation metrics: malicious-detection precision/recall,
benign false-exclusion rate, and rounds-to-detect -- the metrics a
fixed-checkpoint baseline can't report an anytime-valid guarantee for."""
from __future__ import annotations


def detection_metrics(malicious_ids: set[int], ever_excluded_ids: set[int],
                       all_client_ids: set[int]) -> dict[str, float]:
    benign_ids = all_client_ids - malicious_ids
    tp = len(malicious_ids & ever_excluded_ids)
    fp = len(benign_ids & ever_excluded_ids)
    fn = len(malicious_ids - ever_excluded_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    benign_false_exclusion_rate = fp / len(benign_ids) if benign_ids else float("nan")

    return {
        "detection_precision": precision,
        "detection_recall": recall,
        "benign_false_exclusion_rate": benign_false_exclusion_rate,
        "n_malicious": len(malicious_ids),
        "n_excluded_malicious": tp,
        "n_excluded_benign": fp,
    }


def rounds_to_detect(malicious_ids: set[int],
                      first_excluded_round: dict[int, int],
                      attack_start_round: int = 1) -> dict[int, float]:
    """{client_id: rounds elapsed from attack start to first exclusion};
    missing entries (never detected) are omitted -- report separately via
    detection_recall."""
    out = {}
    for cid in malicious_ids:
        if cid in first_excluded_round:
            out[cid] = first_excluded_round[cid] - attack_start_round
    return out
