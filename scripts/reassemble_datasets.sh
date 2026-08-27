#!/usr/bin/env bash
# Reassembles the chunked dataset archives committed under data_archive/ back
# into their original zips, verifies against data_archive/checksums.txt, and
# extracts into data_raw/ in the layout armor_fl.data.preprocessing expects.
#
# The datasets are committed directly to this repo (as <100MB chunks, since
# GitHub rejects any single file >=100MB) so a fresh clone on any machine has
# everything needed without re-downloading from CIC / AWS.
#
# Usage: bash scripts/reassemble_datasets.sh [dataset ...]
#   dataset in {cicids2017, ciciot2023}; default: all available.

set -euo pipefail
cd "$(dirname "$0")/.."

CHECKSUMS="data_archive/checksums.txt"

reassemble_one() {
    local name="$1" zip_name="$2" extract_dir="$3"
    local part_dir="data_archive/${name}"
    if [ ! -d "$part_dir" ]; then
        echo "skip: no chunks found under ${part_dir}"
        return
    fi
    echo "== ${name}: reassembling ${zip_name} =="
    cat "${part_dir}"/*.part_* > "${zip_name}"

    local expected actual
    expected=$(grep "  ${zip_name}\$" "$CHECKSUMS" | cut -d' ' -f1)
    actual=$(shasum -a 256 "${zip_name}" | cut -d' ' -f1)
    if [ -z "$expected" ]; then
        echo "  WARNING: no checksum recorded for ${zip_name}, skipping verification"
    elif [ "$expected" != "$actual" ]; then
        echo "  ERROR: checksum mismatch for ${zip_name}"
        echo "    expected: ${expected}"
        echo "    actual:   ${actual}"
        exit 1
    else
        echo "  checksum OK"
    fi

    mkdir -p "${extract_dir}"
    unzip -q -o "${zip_name}" -d "${extract_dir}"
    echo "  extracted to ${extract_dir}"
}

if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(cicids2017 ciciot2023)
fi
for ds in "${DATASETS[@]}"; do
    case "$ds" in
        cicids2017)
            reassemble_one cicids2017 MachineLearningCSV_CICIDS2017.zip data_raw/cicids2017
            ;;
        ciciot2023)
            reassemble_one ciciot2023 MERGED_CSV_CICIoT2023.zip data_raw/ciciot2023
            ;;
        *)
            echo "unknown dataset: $ds (expected cicids2017 or ciciot2023)"
            exit 1
            ;;
    esac
done

echo "Done. See data_archive/README.md for dataset provenance."
