"""Centralized XGBoost baseline: a fast, non-federated upper-bound sanity
check on each dataset before spending compute on the full ARMOR-FL grid.

This answers a different question than the FL simulation grid: "is the
dataset/feature set learnable at all, and how far below a strong classical
baseline does the (federated, robust-aggregation, tiny-backbone) approach
sit in the no-attack/IID case?" It intentionally sees *all* training data
pooled (no client partitioning, no attacks) -- it is a ceiling reference,
not a competing method in the robustness comparison.

Uses the same loaders, sample_frac, and stratified train/test split as
scripts/run_experiment.py, and reports the same metric set as
armor_fl.fl.client.evaluate (accuracy, weighted/macro F1, weighted
precision/recall) so numbers are directly comparable to the FL results.

Usage:
    python scripts/xgboost_baseline.py --dataset cicids2017
    python scripts/xgboost_baseline.py --dataset cicids2018 --sample-frac 0.05 --device cuda
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")

from armor_fl.data.preprocessing import (
    load_cicids2017, load_cicids2018, load_ciciot2023, train_test_split_stratified,
)

DATASET_LOADERS = {
    "cicids2017": (load_cicids2017, "data_raw/cicids2017"),
    "cicids2018": (load_cicids2018, "data_raw/cicids2018"),
    "ciciot2023": (load_ciciot2023, "data_raw/ciciot2023"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS))
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--out-dir", default="results/baselines")
    args = p.parse_args()

    loader, default_raw_dir = DATASET_LOADERS[args.dataset]
    raw_dir = args.raw_dir or default_raw_dir

    t0 = time.time()
    bundle = loader(raw_dir, sample_frac=args.sample_frac, seed=args.seed)
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        bundle.X, bundle.y, test_frac=args.test_frac, seed=args.seed)
    load_dt = time.time() - t0
    print(f"Dataset: {bundle.dataset_name} | train={X_train.shape[0]} "
          f"test={X_test.shape[0]} features={X_train.shape[1]} "
          f"classes={len(bundle.classes)} (loaded in {load_dt:.1f}s)")

    import xgboost as xgb
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
    )

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(bundle.classes),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        tree_method="hist",
        device=args.device,
        eval_metric="mlogloss",
        random_state=args.seed,
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_dt = time.time() - t0

    t0 = time.time()
    preds = clf.predict(X_test)
    infer_dt = time.time() - t0

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
    }

    print(f"Trained in {train_dt:.1f}s ({args.device}), inferred in {infer_dt:.2f}s")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.dataset}_xgboost.json")
    with open(out_path, "w") as f:
        json.dump({
            "dataset": bundle.dataset_name,
            "sample_frac": args.sample_frac,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "classes": bundle.classes,
            "device": args.device,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "seed": args.seed,
            "load_seconds": load_dt,
            "train_seconds": train_dt,
            "infer_seconds": infer_dt,
            "metrics": metrics,
        }, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
