#!/usr/bin/env bash
# Runs the full experiment sequence for this pass: comparability sanity
# checks first (cheap, paper-protocol fidelity), then the three trimmed
# robustness grids (the long pole -- see PROGRESS.md's 2026-08-28 entries for
# the grid-trimming and batch_size decisions behind these configs). Run
# sequentially (not concurrently) to avoid the GPU contention documented in
# PROGRESS.md when multiple run_experiment.py processes shared the GPU.
#
# Usage: bash scripts/run_all_experiments.sh 2>&1 | tee /tmp/run_all_experiments.log
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
    configs/cicids2017_comparability.yaml
    configs/cicids2018_comparability.yaml
    configs/cicids2017_robustness.yaml
    configs/cicids2018_robustness.yaml
    configs/ciciot2023_robustness.yaml
)

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo ""
    echo "================================================================"
    echo "== $(date '+%Y-%m-%d %H:%M:%S') START: $name"
    echo "================================================================"
    conda run --no-capture-output -n py313 python -u scripts/run_experiment.py --config "$cfg"
    status=$?
    echo "== $(date '+%Y-%m-%d %H:%M:%S') END: $name (exit=$status)"
    if [ $status -ne 0 ]; then
        echo "!! $name FAILED (exit=$status) -- continuing to next config anyway"
    fi
done

echo ""
echo "================================================================"
echo "== $(date '+%Y-%m-%d %H:%M:%S') ALL CONFIGS DONE"
echo "================================================================"
