"""End-to-end smoke test on small synthetic tabular data: validates the full
wiring (partition -> local train -> attack injection -> aggregate -> eval)
runs cleanly and produces sane metrics, before spending real time/compute on
CICIDS2017. Run: python scripts/smoke_test.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from armor_fl.fl.armor import ArmorConfig
from armor_fl.fl.simulate import SimulationConfig, run_simulation
from armor_fl.models.dds_backbone import SE1DSqueezeNet


def make_synthetic(n=1200, n_features=30, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, size=(n_classes, n_features))
    y = rng.integers(0, n_classes, size=n)
    X = centers[y] + rng.normal(0, 1.0, size=(n, n_features))
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-8)
    return X.astype(np.float32), y.astype(np.int64)


def model_factory(n_features, n_classes):
    return lambda: SE1DSqueezeNet(num_features=n_features, num_classes=n_classes)


def main():
    n_features, n_classes = 30, 4
    X, y = make_synthetic(n_features=n_features, n_classes=n_classes)
    X_test, y_test = make_synthetic(n=300, n_features=n_features, n_classes=n_classes, seed=99)
    mf = model_factory(n_features, n_classes)

    m = mf()
    print(f"Model params: {m.num_parameters()}")

    scenarios = [
        ("FedAvg, no attack, IID", SimulationConfig(
            num_clients=10, num_rounds=6, local_epochs=3, patience=2,
            aggregator="fedavg", malicious_fraction=0.0)),
        ("FedAvg, 30% gaussian_noise attack, IID", SimulationConfig(
            num_clients=10, num_rounds=6, local_epochs=3, patience=2,
            aggregator="fedavg", attack_type="gaussian_noise",
            malicious_fraction=0.3)),
        ("ARMOR-FL, 30% gaussian_noise attack, IID", SimulationConfig(
            num_clients=10, num_rounds=10, local_epochs=3, patience=2,
            aggregator="armor", attack_type="gaussian_noise",
            malicious_fraction=0.3,
            armor_config=ArmorConfig(burn_in_rounds=2))),
        ("ARMOR-FL, non-IID (alpha=0.5), 20% label_flip", SimulationConfig(
            num_clients=10, num_rounds=10, local_epochs=3, patience=2,
            non_iid_alpha=0.5, aggregator="armor", attack_type="label_flip",
            malicious_fraction=0.2, armor_config=ArmorConfig(burn_in_rounds=2))),
        ("Krum, 30% gaussian_noise attack, IID", SimulationConfig(
            num_clients=10, num_rounds=6, local_epochs=3, patience=2,
            aggregator="krum", aggregator_kwargs={"num_byzantine_assumed": 3},
            attack_type="gaussian_noise", malicious_fraction=0.3)),
    ]

    for name, cfg in scenarios:
        t0 = time.time()
        result = run_simulation(mf, X, y, X_test, y_test, n_classes, cfg)
        dt = time.time() - t0
        last = result.per_round_metrics[-1] if result.per_round_metrics else {}
        print(f"\n=== {name} ===")
        print(f"  wall clock: {dt:.1f}s, final round: {last.get('round')}, "
              f"acc={last.get('accuracy', float('nan')):.3f}, "
              f"f1={last.get('f1', float('nan')):.3f}")
        if cfg.malicious_fraction > 0:
            print(f"  detection: {result.detection}")

    print("\nSmoke test completed without errors.")


if __name__ == "__main__":
    main()
