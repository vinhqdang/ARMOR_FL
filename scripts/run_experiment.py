"""CLI experiment runner: loads a dataset, sweeps over aggregator x attack x
malicious_fraction x non_iid_alpha, and writes per-run summaries + per-round
logs to results/<experiment_name>/.

Usage:
    python scripts/run_experiment.py --config configs/cicids2017_robustness.yaml
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, ".")

from armor_fl.data.preprocessing import (
    load_cicids2017, load_cicids2018, load_ciciot2023, train_test_split_stratified,
)
from armor_fl.fl.armor import ArmorConfig
from armor_fl.fl.simulate import SimulationConfig, run_simulation
from armor_fl.models.dds_backbone import SE1DSqueezeNet

DATASET_LOADERS = {
    "cicids2017": load_cicids2017,
    "cicids2018": load_cicids2018,
    "ciciot2023": load_ciciot2023,
}


def load_dataset(name: str, raw_dir: str, sample_frac: float | None, seed: int):
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset '{name}', choices: {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name](raw_dir, sample_frac=sample_frac, seed=seed)


def build_sim_config(base: dict, aggregator: str, attack_type: str | None,
                      malicious_fraction: float, non_iid_alpha: float | None,
                      seed: int) -> SimulationConfig:
    kwargs = dict(base)
    armor_cfg = None
    if aggregator == "armor":
        armor_cfg = ArmorConfig(**kwargs.pop("armor_config", {}))
    aggregator_kwargs = kwargs.pop("aggregator_kwargs", {}).get(aggregator, {})
    return SimulationConfig(
        num_clients=kwargs.get("num_clients", 10),
        client_fraction=kwargs.get("client_fraction", 0.6),
        num_rounds=kwargs.get("num_rounds", 30),
        local_epochs=kwargs.get("local_epochs", 20),
        patience=kwargs.get("patience", 5),
        lr=kwargs.get("lr", 0.01),
        batch_size=kwargs.get("batch_size", 32),
        dropout=kwargs.get("dropout", 0.5),
        non_iid_alpha=non_iid_alpha,
        aggregator=aggregator,
        aggregator_kwargs=aggregator_kwargs,
        armor_config=armor_cfg,
        attack_type=attack_type,
        malicious_fraction=malicious_fraction,
        attack_start_round=kwargs.get("attack_start_round", 1),
        seed=seed,
        device=kwargs.get("device", "cpu"),
        log_every=kwargs.get("log_every", 1),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = args.out or os.path.join("results", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    seed = cfg.get("seed", 0)
    bundle = load_dataset(cfg["dataset"], cfg.get("raw_dir", "data_raw/cicids2017"),
                           cfg.get("sample_frac"), seed)
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        bundle.X, bundle.y, test_frac=cfg.get("test_frac", 0.2), seed=seed)
    n_features, n_classes = X_train.shape[1], len(bundle.classes)
    print(f"Dataset: {bundle.dataset_name} | train={X_train.shape[0]} "
          f"test={X_test.shape[0]} features={n_features} classes={n_classes}")

    def model_factory():
        return SE1DSqueezeNet(num_features=n_features, num_classes=n_classes,
                               dropout=cfg.get("fl", {}).get("dropout", 0.5))

    grid = cfg["grid"]
    all_summaries = []
    for aggregator in grid["aggregators"]:
        for attack_type in grid.get("attack_types", [None]):
            for malicious_fraction in (grid.get("malicious_fractions", [0.0])
                                        if attack_type else [0.0]):
                for non_iid_alpha in grid.get("non_iid_alphas", [None]):
                    run_id = (f"{aggregator}__atk={attack_type}__mal={malicious_fraction}"
                              f"__alpha={non_iid_alpha}")
                    print(f"\n=== {run_id} ===")
                    sim_cfg = build_sim_config(cfg.get("fl", {}), aggregator, attack_type,
                                                malicious_fraction, non_iid_alpha, seed)
                    t0 = time.time()
                    result = run_simulation(model_factory, X_train, y_train, X_test, y_test,
                                             n_classes, sim_cfg)
                    dt = time.time() - t0

                    with open(os.path.join(out_dir, f"{run_id}.json"), "w") as f:
                        json.dump({
                            "run_id": run_id,
                            "per_round_metrics": result.per_round_metrics,
                            "detection": {k: v for k, v in result.detection.items()},
                            "wall_clock_seconds": dt,
                        }, f, indent=2, default=str)

                    last = result.per_round_metrics[-1] if result.per_round_metrics else {}
                    summary = {
                        "run_id": run_id, "aggregator": aggregator,
                        "attack_type": attack_type, "malicious_fraction": malicious_fraction,
                        "non_iid_alpha": non_iid_alpha,
                        "final_accuracy": last.get("accuracy"), "final_f1": last.get("f1"),
                        "wall_clock_seconds": dt,
                        **{f"det_{k}": v for k, v in result.detection.items()
                           if k in ("detection_precision", "detection_recall",
                                    "benign_false_exclusion_rate")},
                    }
                    all_summaries.append(summary)
                    print(f"  acc={summary['final_accuracy']} f1={summary['final_f1']} "
                          f"({dt:.1f}s)")

    import pandas as pd
    pd.DataFrame(all_summaries).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"\nWrote {len(all_summaries)} runs to {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
