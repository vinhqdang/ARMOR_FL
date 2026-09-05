"""Computes both final-round accuracy AND mean-of-last-N-rounds accuracy per
run, since ARMOR's per-round accuracy can oscillate heavily (client subset
sampling + slow re-detection after a reinstated attacker resets its e-process)
in a way krum/fedavg/foolsgold do not -- so a single final-round number can
land on an arbitrary good or bad round for ARMOR specifically. Final-round
stays the primary/comparable-across-aggregators metric; last-N-mean is a
secondary, more stable statistic reported alongside it for ARMOR.

Usage:
    python scripts/analyze_stability.py results/cicids2017_robustness
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def summarize(path: str, last_n: int) -> dict:
    d = json.load(open(path))
    rounds = d["per_round_metrics"]
    accs = [r["accuracy"] for r in rounds]
    f1s = [r["f1"] for r in rounds]
    tail_accs = accs[-last_n:]
    tail_f1s = f1s[-last_n:]
    return {
        "run_id": d["run_id"],
        "final_accuracy": accs[-1] if accs else float("nan"),
        "final_f1": f1s[-1] if f1s else float("nan"),
        f"mean_last{last_n}_accuracy": float(np.mean(tail_accs)) if tail_accs else float("nan"),
        f"mean_last{last_n}_f1": float(np.mean(tail_f1s)) if tail_f1s else float("nan"),
        f"std_last{last_n}_accuracy": float(np.std(tail_accs)) if tail_accs else float("nan"),
        "n_rounds": len(rounds),
        **{f"det_{k}": v for k, v in d["detection"].items()
           if k in ("detection_precision", "detection_recall", "benign_false_exclusion_rate")},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--last-n", type=int, default=8)
    args = p.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        if os.path.basename(f) == "summary.csv":
            continue
        row = summarize(f, args.last_n)
        base = os.path.basename(f).replace(".json", "")
        parts = dict(kv.split("=", 1) for kv in base.split("__") if "=" in kv)
        row["aggregator"] = base.split("__")[0]
        row.update(parts)
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 300)

    print("=" * 100)
    print(f"Mean FINAL-round accuracy per aggregator (n={len(df)} runs)")
    print("=" * 100)
    print(df.groupby("aggregator")["final_accuracy"].mean().round(4))

    print()
    print("=" * 100)
    print(f"Mean of LAST-{args.last_n}-rounds accuracy per aggregator (more stable for oscillating runs)")
    print("=" * 100)
    print(df.groupby("aggregator")[f"mean_last{args.last_n}_accuracy"].mean().round(4))

    print()
    print("=" * 100)
    print("Per-run gap between final-round and last-N-mean (large gap = high oscillation)")
    print("=" * 100)
    df["stability_gap"] = (df["final_accuracy"] - df[f"mean_last{args.last_n}_accuracy"]).abs()
    print(df.groupby("aggregator")["stability_gap"].agg(["mean", "max"]).round(4))

    print()
    print("=" * 100)
    print(f"Std-dev of accuracy over last {args.last_n} rounds per aggregator "
          "(how much each aggregator's own accuracy swings round to round)")
    print("=" * 100)
    print(df.groupby("aggregator")[f"std_last{args.last_n}_accuracy"].mean().round(4))

    out = os.path.join(args.results_dir, "stability_summary.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
