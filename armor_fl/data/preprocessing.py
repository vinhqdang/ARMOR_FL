"""CICIDS2017 / CICIDS2018 loading and preprocessing.

Follows the pipeline described in FedSE-1DSqueezeNet (Cluster Computing, 2026),
Sec. 4.2: infinite -> NaN -> mean imputation (per feature) -> Min-Max scaling to [0, 1].
Raw per-attack labels are grouped into a standard CICIDS taxonomy (documented in
LABEL_GROUPS below) to give a stable multi-class target.
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard CICIDS2017 label grouping: BENIGN + 7 attack families.
# Rare sub-types (Heartbleed, Infiltration) are kept as their own class but are
# extremely small (11 / 36 raw rows) -- expect them to be dropped by min-samples
# filtering in most non-IID partitions; this is a known dataset limitation, not
# a preprocessing bug, and should be reported as such.
CICIDS2017_LABEL_GROUPS: dict[str, str] = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack � Brute Force": "WebAttack",
    "Web Attack � XSS": "WebAttack",
    "Web Attack � Sql Injection": "WebAttack",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}
CICIDS2017_CLASSES = [
    "BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot",
    "Infiltration", "Heartbleed",
]

LABEL_COL_CANDIDATES = ["Label", " Label", "label", "Attack", "Attack_type"]


@dataclass
class DatasetBundle:
    X: np.ndarray            # (N, F) float32, min-max scaled to [0, 1]
    y: np.ndarray             # (N,) int64 class indices
    feature_names: list[str]
    classes: list[str]
    dataset_name: str


def _find_label_col(columns: list[str]) -> str:
    for cand in LABEL_COL_CANDIDATES:
        if cand in columns:
            return cand
    raise ValueError(f"No label column found among {columns[:5]}...")


def load_cicids2017(raw_dir: str, sample_frac: float | None = None,
                     seed: int = 0) -> DatasetBundle:
    """Load and concatenate all 8 CICIDS2017 MachineLearningCVE CSVs.

    Parameters
    ----------
    raw_dir: directory containing the *.pcap_ISCX.csv files (or a parent dir
        that contains a MachineLearningCVE/ subfolder -- both are handled).
    sample_frac: if set, stratified-sample this fraction of rows per file
        (for smoke tests). None = use all rows.
    """
    search_dirs = [raw_dir, os.path.join(raw_dir, "MachineLearningCVE")]
    files: list[str] = []
    for d in search_dirs:
        files = sorted(glob.glob(os.path.join(d, "*.csv")))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"No CICIDS2017 CSVs found under {raw_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if sample_frac is not None:
            df = df.groupby("Label", group_keys=False)[df.columns].apply(
                lambda g: g.sample(frac=sample_frac, random_state=seed)
                if len(g) > 0 else g
            )
        frames.append(df)
        logger.info("Loaded %s: %d rows", os.path.basename(f), len(df))
    df = pd.concat(frames, ignore_index=True)

    label_col = _find_label_col(list(df.columns))
    df["__group__"] = df[label_col].map(CICIDS2017_LABEL_GROUPS)
    unmapped = df["__group__"].isna().sum()
    if unmapped:
        unknown_labels = df.loc[df["__group__"].isna(), label_col].unique()
        logger.warning("Dropping %d rows with unmapped labels: %s",
                        unmapped, unknown_labels)
        df = df.dropna(subset=["__group__"])

    feature_cols = [c for c in df.columns if c not in (label_col, "__group__")]
    # Drop identifier / non-numeric leakage columns if present.
    for junk in ["Flow ID", "Source IP", "Destination IP", "Timestamp",
                 "Src IP", "Dst IP", "SimillarHTTP"]:
        if junk in feature_cols:
            feature_cols.remove(junk)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_str = df["__group__"].values

    class_to_idx = {c: i for i, c in enumerate(CICIDS2017_CLASSES)}
    y = np.array([class_to_idx[c] for c in y_str], dtype=np.int64)

    X = clean_and_scale(X)
    return DatasetBundle(
        X=X, y=y, feature_names=feature_cols,
        classes=CICIDS2017_CLASSES, dataset_name="CICIDS2017",
    )


def clean_and_scale(X: pd.DataFrame) -> np.ndarray:
    """Inf -> NaN -> per-feature mean impute -> Min-Max to [0, 1] (paper Eq. 11)."""
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True))
    # Any column that's entirely NaN (all-inf originally) -> fill with 0.
    X = X.fillna(0.0)
    X = X.to_numpy(dtype=np.float64)

    xmin = X.min(axis=0, keepdims=True)
    xmax = X.max(axis=0, keepdims=True)
    denom = np.where((xmax - xmin) == 0, 1.0, xmax - xmin)
    X = (X - xmin) / denom
    return X.astype(np.float32)


def train_test_split_stratified(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2,
                                 seed: int = 0):
    """sklearn's stratified split requires >=2 samples per class; extremely
    rare classes (e.g. Heartbleed: 11 rows in the full CICIDS2017, fewer
    still under sub-sampling) can violate that. Such rows are dropped from
    BOTH splits with a warning rather than crashing -- a real property of
    this dataset's class imbalance, not a preprocessing bug."""
    from sklearn.model_selection import train_test_split

    classes, counts = np.unique(y, return_counts=True)
    rare = classes[counts < 2]
    if len(rare):
        logger.warning(
            "Dropping %d rows from classes with <2 samples (can't stratify): %s",
            int(np.isin(y, rare).sum()), rare.tolist(),
        )
        keep = ~np.isin(y, rare)
        X, y = X[keep], y[keep]

    return train_test_split(X, y, test_size=test_frac, random_state=seed,
                             stratify=y)
