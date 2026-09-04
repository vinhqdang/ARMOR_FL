"""Fast decision-grade check on the few grid cells that actually decide whether
ARMOR-FL is working, so a change can be validated in ~10-20 minutes instead of
burning ~1.3 days on the full 152-run grid.

Why these cells: the full v1 grid showed ARMOR losing to `krum` on 2/3 datasets
and losing to no-defense `fedavg` under `label_flip`. Those are the comparisons
a fix has to move. Running the whole grid to learn that is wasteful, so this
runs only:

    {fedavg, krum, armor} x {label_flip, gaussian_noise} x {mal=0.4}

on one dataset at reduced settings. `fedavg` is the "no defense" floor and
`krum` is the baseline ARMOR must at least match -- without both, an ARMOR
number in isolation means nothing.

It also prints ARMOR's detection precision/recall and benign false-exclusion
rate directly, since those are the quantities the shift-gate fix targets;
accuracy alone can hide a defense that is excluding the wrong clients.

Settings are deliberately reduced (small sample_frac, fewer rounds/epochs) --
absolute numbers are NOT comparable to the full grid. Only the RELATIVE
ordering between aggregators, measured under identical settings, is meaningful.

Usage:
    python scripts/canary_check.py                     # default: cicids2017
    python scripts/canary_check.py --dataset ciciot2023 --sample-frac 0.01
"""
import argparse
import sys
import time

sys.path.insert(0, ".")

from armor_fl.data.preprocessing import (
    load_cicids2017, load_cicids2018, load_ciciot2023, train_test_split_stratified,
)
from armor_fl.fl.armor import ArmorConfig
from armor_fl.fl.simulate import SimulationConfig, run_simulation
from armor_fl.models.dds_backbone import SE1DSqueezeNet

LOADERS = {
    "cicids2017": (load_cicids2017, "data_raw/cicids2017"),
    "cicids2018": (load_cicids2018, "data_raw/cicids2018"),
    "ciciot2023": (load_ciciot2023, "data_raw/ciciot2023"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cicids2017", choices=list(LOADERS))
    p.add_argument("--sample-frac", type=float, default=0.03)
    p.add_argument("--num-rounds", type=int, default=12)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--malicious-fraction", type=float, default=0.4)
    p.add_argument("--aggregators", nargs="+", default=["fedavg", "krum", "armor"])
    p.add_argument("--attacks", nargs="+", default=["label_flip", "gaussian_noise"])
    p.add_argument("--non-iid-alpha", default="0.5",
                    help="Dirichlet alpha, or 'none'/'iid' for true IID partitioning "
                         "(matches non_iid_alpha: null in the *_robustness.yaml configs)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.non_iid_alpha = (None if args.non_iid_alpha.lower() in ("none", "iid")
                           else float(args.non_iid_alpha))

    loader, raw_dir = LOADERS[args.dataset]
    bundle = loader(raw_dir, sample_frac=args.sample_frac, seed=0)
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        bundle.X, bundle.y, seed=0)
    n_features, n_classes = X_train.shape[1], len(bundle.classes)
    print(f"dataset={args.dataset} train={X_train.shape[0]} test={X_test.shape[0]} "
          f"features={n_features} classes={n_classes}")
    print(f"settings: rounds={args.num_rounds} local_epochs={args.local_epochs} "
          f"mal_frac={args.malicious_fraction} alpha={args.non_iid_alpha}")
    print("NOTE: reduced settings -- compare aggregators to EACH OTHER, not to "
          "full-grid numbers.\n")

    def model_factory():
        return SE1DSqueezeNet(num_features=n_features, num_classes=n_classes)

    rows = []
    for attack in args.attacks:
        for aggregator in args.aggregators:
            cfg = SimulationConfig(
                num_clients=10, client_fraction=0.6, num_rounds=args.num_rounds,
                local_epochs=args.local_epochs, patience=3, lr=0.01, batch_size=256,
                non_iid_alpha=args.non_iid_alpha, aggregator=aggregator,
                aggregator_kwargs={"num_byzantine_assumed": 3} if aggregator == "krum" else {},
                armor_config=ArmorConfig(burn_in_rounds=3) if aggregator == "armor" else None,
                attack_type=attack, malicious_fraction=args.malicious_fraction,
                attack_start_round=3, seed=0, device=args.device, log_every=1,
            )
            t0 = time.time()
            result = run_simulation(model_factory, X_train, y_train, X_test, y_test,
                                     n_classes, cfg)
            dt = time.time() - t0
            last = result.per_round_metrics[-1] if result.per_round_metrics else {}
            det = result.detection
            rows.append({
                "attack": attack, "aggregator": aggregator,
                "acc": last.get("accuracy"), "f1": last.get("f1"),
                "det_prec": det.get("detection_precision"),
                "det_rec": det.get("detection_recall"),
                "benign_fer": det.get("benign_false_exclusion_rate"),
                "secs": dt,
            })
            print(f"  {attack:16s} {aggregator:10s} acc={last.get('accuracy'):.4f} "
                  f"f1={last.get('f1'):.4f} ({dt:.0f}s)")

    print("\n" + "=" * 88)
    print(f"{'attack':16s} {'aggregator':10s} {'acc':>8s} {'f1':>8s} "
          f"{'det_prec':>9s} {'det_rec':>8s} {'benign_FER':>11s}")
    print("=" * 88)
    for r in rows:
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) and v == v else "   -"
        print(f"{r['attack']:16s} {r['aggregator']:10s} {r['acc']:8.4f} {r['f1']:8.4f} "
              f"{fmt(r['det_prec']):>9s} {fmt(r['det_rec']):>8s} {fmt(r['benign_fer']):>11s}")

    print("\nVERDICT GUIDE")
    for attack in args.attacks:
        sub = {r["aggregator"]: r for r in rows if r["attack"] == attack}
        if not {"armor", "krum", "fedavg"} <= set(sub):
            continue
        a, k, f = sub["armor"]["acc"], sub["krum"]["acc"], sub["fedavg"]["acc"]
        verdict = []
        verdict.append("armor>=krum" if a >= k - 0.02 else f"armor LOSES to krum ({a:.3f} vs {k:.3f})")
        verdict.append("armor>=fedavg" if a >= f - 0.02 else f"armor LOSES to no-defense fedavg ({a:.3f} vs {f:.3f})")
        print(f"  {attack:16s} " + " | ".join(verdict))
    print("\nIf ARMOR loses to fedavg here, the full grid will not rescue it -- "
          "fix before spending the ~1.3 days.")


if __name__ == "__main__":
    main()
