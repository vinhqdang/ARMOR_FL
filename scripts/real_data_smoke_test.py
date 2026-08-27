"""Validates the pipeline on a stratified sample of real CICIDS2017 data
(preprocessing -> partition -> FL loop -> ARMOR-FL) before committing to a
full-scale run. Run: python scripts/real_data_smoke_test.py
"""
import sys
import time

sys.path.insert(0, ".")

from armor_fl.data.preprocessing import load_cicids2017, train_test_split_stratified
from armor_fl.fl.armor import ArmorConfig
from armor_fl.fl.simulate import SimulationConfig, run_simulation
from armor_fl.models.dds_backbone import SE1DSqueezeNet


def main():
    t0 = time.time()
    bundle = load_cicids2017("data_raw/cicids2017", sample_frac=0.05, seed=0)
    print(f"Loaded {bundle.X.shape[0]} rows, {bundle.X.shape[1]} features, "
          f"{len(bundle.classes)} classes in {time.time() - t0:.1f}s")

    import numpy as np
    unique, counts = np.unique(bundle.y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {bundle.classes[u]}: {c}")

    X_train, X_test, y_train, y_test = train_test_split_stratified(bundle.X, bundle.y,
                                                                     test_frac=0.2, seed=0)
    n_features, n_classes = X_train.shape[1], len(bundle.classes)

    def mf():
        return SE1DSqueezeNet(num_features=n_features, num_classes=n_classes)

    m = mf()
    print(f"Model params: {m.num_parameters()}")

    cfg = SimulationConfig(
        num_clients=10, client_fraction=0.6, num_rounds=8, local_epochs=5,
        patience=3, aggregator="armor", attack_type="gaussian_noise",
        malicious_fraction=0.3, armor_config=ArmorConfig(burn_in_rounds=2),
        non_iid_alpha=None,
    )
    t0 = time.time()
    result = run_simulation(mf, X_train, y_train, X_test, y_test, n_classes, cfg)
    print(f"\nSimulation wall clock: {time.time() - t0:.1f}s for {cfg.num_rounds} rounds")
    for row in result.per_round_metrics:
        print(f"  round {row['round']}: acc={row['accuracy']:.3f} f1={row['f1']:.3f} "
              f"excluded={row.get('excluded_this_round')}")
    print(f"Detection: {result.detection}")


if __name__ == "__main__":
    main()
