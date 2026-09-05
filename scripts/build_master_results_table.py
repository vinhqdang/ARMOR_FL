"""Combines the three datasets' stability_summary.csv (produced by
analyze_stability.py) into one master table for the manuscript's Results
section, plus the comparability (paper-protocol sanity-check) results.

Usage:
    python scripts/build_master_results_table.py
"""
import pandas as pd

dfs = []
for ds in ["cicids2017", "cicids2018", "ciciot2023"]:
    df = pd.read_csv(f"results/{ds}_robustness/stability_summary.csv")
    df["dataset"] = ds
    dfs.append(df)
full = pd.concat(dfs, ignore_index=True)
full.to_csv("results/master_robustness_table.csv", index=False)

pd.set_option("display.width", 200)
print("=" * 100)
print("Mean final-round accuracy: aggregator x dataset")
print("=" * 100)
print(full.pivot_table(index="aggregator", columns="dataset", values="final_accuracy", aggfunc="mean").round(4))

print()
print("=" * 100)
print("Mean of last-8-rounds accuracy: aggregator x dataset (secondary, more stable metric)")
print("=" * 100)
print(full.pivot_table(index="aggregator", columns="dataset", values="mean_last8_accuracy", aggfunc="mean").round(4))

print()
print("=" * 100)
print("Detection precision / recall / benign false-exclusion rate: ARMOR only, by dataset")
print("=" * 100)
armor = full[full["aggregator"] == "armor"]
print(armor.groupby("dataset")[["det_detection_precision", "det_detection_recall",
                                  "det_benign_false_exclusion_rate"]].mean().round(3))

print()
print("=" * 100)
print("Comparability (paper-protocol) sanity check: FedAvg accuracy vs original paper's own numbers")
print("=" * 100)
import json
for ds in ["cicids2017", "cicids2018"]:
    print(f"\n{ds}_comparability:")
    for alpha in ["None", "1.0", "0.5", "0.1"]:
        f = f"results/{ds}_comparability/fedavg__atk=None__mal=0.0__alpha={alpha}.json"
        try:
            d = json.load(open(f))
            last = d["per_round_metrics"][-1]
            print(f"  alpha={alpha}: acc={last['accuracy']:.4f} f1={last['f1']:.4f}")
        except FileNotFoundError:
            print(f"  alpha={alpha}: (file not found)")

print("\nWrote results/master_robustness_table.csv")
