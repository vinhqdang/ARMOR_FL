"""Wall-clock benchmark for one representative FL run_id, CPU vs CUDA, at
close-to-real settings (local_epochs=20, patience=5) but fewer rounds --
used to extrapolate the full experiment grid's total compute budget before
launching it for real. See PROGRESS.md for the resulting estimate.

Usage:
    python scripts/benchmark_fl.py --sample-frac 0.1 --num-rounds 5
"""
import argparse
import sys
import time

sys.path.insert(0, ".")

from armor_fl.data.preprocessing import load_cicids2017, train_test_split_stratified
from armor_fl.fl.armor import ArmorConfig
from armor_fl.fl.simulate import SimulationConfig, run_simulation
from armor_fl.models.dds_backbone import SE1DSqueezeNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-frac", type=float, default=0.1)
    p.add_argument("--num-rounds", type=int, default=5)
    p.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    args = p.parse_args()

    bundle = load_cicids2017("data_raw/cicids2017", sample_frac=args.sample_frac, seed=0)
    X_train, X_test, y_train, y_test = train_test_split_stratified(bundle.X, bundle.y, seed=0)
    n_features, n_classes = X_train.shape[1], len(bundle.classes)
    print(f"train={X_train.shape[0]} test={X_test.shape[0]} "
          f"features={n_features} classes={n_classes}")

    def model_factory():
        return SE1DSqueezeNet(num_features=n_features, num_classes=n_classes)

    for device in args.devices:
        cfg = SimulationConfig(
            num_clients=10, client_fraction=0.6, num_rounds=args.num_rounds,
            local_epochs=20, patience=5, lr=0.01, batch_size=32,
            non_iid_alpha=0.5, aggregator="armor",
            armor_config=ArmorConfig(burn_in_rounds=3),
            attack_type="gaussian_noise", malicious_fraction=0.2,
            attack_start_round=1, seed=0, device=device, log_every=1,
        )
        t0 = time.time()
        result = run_simulation(model_factory, X_train, y_train, X_test, y_test,
                                 n_classes, cfg)
        dt = time.time() - t0
        last = result.per_round_metrics[-1] if result.per_round_metrics else {}
        per_round = dt / args.num_rounds
        print(f"\ndevice={device}: {dt:.1f}s total, {per_round:.1f}s/round "
              f"(acc={last.get('accuracy')}, f1={last.get('f1')})")
        print(f"  extrapolated to 30 rounds: {per_round * 30:.0f}s "
              f"({per_round * 30 / 60:.1f} min)")
        print(f"  extrapolated to full grid (560 run_ids x 3 datasets = 1680): "
              f"{per_round * 30 * 1680 / 3600:.1f} hours")


if __name__ == "__main__":
    main()
