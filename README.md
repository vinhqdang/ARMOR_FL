# ARMOR-FL

**A**nytime-valid **R**obust **M**artingale-based **O**nline **R**eweighting for
**F**ederated **L**earning -- a Byzantine-robust, drift-aware aggregation layer
for federated intrusion detection, targeting *Cluster Computing* (Springer).

Builds on FedSE-1DSqueezeNet (Zhou, Mao & Chen, *Cluster Computing* 2026,
[10.1007/s10586-026-06482-2](https://doi.org/10.1007/s10586-026-06482-2)) as
the lightweight backbone / architecture baseline. ARMOR-FL's own contribution
is aggregation-layer, not backbone-layer: it replaces plain size-weighted
FedAvg with a trust weight driven by a per-client, anytime-valid sequential
test (an e-process), giving formal false-exclusion guarantees at any stopping
round, plus a second self-referential e-process that decouples a client that
is merely *drifting* (needs local adaptation) from one that is *attacking*
(needs downweighting/exclusion).

See `PROGRESS.md` for current status, findings so far, and what's next --
read that first when picking this project back up on a new machine.

## Setup

```bash
conda activate py313          # or any Python 3.13 env
pip install -r requirements.txt
```

## Datasets

Raw datasets are committed to this repo in `data_archive/` as <100MB chunks
(GitHub's hard per-file limit), so a fresh clone has everything needed
without re-downloading from CIC / AWS. Reassemble with:

```bash
bash scripts/reassemble_datasets.sh            # all available
bash scripts/reassemble_datasets.sh cicids2017 # just one
```

This extracts into `data_raw/<dataset>/`, which is what
`armor_fl.data.preprocessing` reads from. See `data_archive/README.md` for
provenance (original source URLs, checksums) of each dataset.

| Dataset | Status | Loader |
|---|---|---|
| CICIDS2017 | archived in `data_archive/cicids2017/` | `load_cicids2017` |
| CICIDS2018 | archived in `data_archive/cicids2018/` | `load_cicids2018` |
| CICIoT2023 | archived in `data_archive/ciciot2023/` | `load_ciciot2023` |

## Project layout

```
armor_fl/
  data/preprocessing.py   # CICIDS2017 / CICIoT2023 loaders: inf->NaN->mean
                           # impute->min-max scale, label grouping taxonomy
  data/partition.py       # IID and Dirichlet non-IID client partitioning
  models/dds_backbone.py  # PyTorch reimplementation of SE-1DSqueezeNet
                           # (paper's Table 1 DDS module), for direct
                           # architecture-level comparability
  models/quantization.py  # 8-bit uniform PTQ (paper Eqs. 8-10)
  fl/eprocess.py           # generic anytime-valid e-process (testing-by-
                           # betting, Waudby-Smith & Ramdas 2020)
  fl/armor.py              # ARMOR-FL aggregator: robust center + MAD +
                           # population/drift e-processes + trust weights +
                           # exclusion/probation/reinstatement
  fl/aggregators.py        # baselines: FedAvg, Krum/Multi-Krum, trimmed
                           # mean, coordinate median, FoolsGold
  fl/attacks.py             # label-flip, sign-flip, Gaussian-noise,
                           # free-rider, ALIE attack simulation
  fl/client.py              # local training w/ early stopping (paper Alg. 2)
  fl/simulate.py            # full FL simulation loop (custom, no Flower)
  eval/metrics.py           # detection precision/recall, benign
                           # false-exclusion rate

configs/       # experiment grid YAMLs (dataset x aggregator x attack x
               # malicious-fraction x non-IID-alpha)
scripts/       # smoke tests, dataset reassembly, experiment runner
tests/         # unit + statistical validation tests (pytest)
manuscript/    # Springer Nature LaTeX template (sn-jnl.cls), extracted from
               # the December 2024 template package
```

## Running things

```bash
# unit + statistical tests (validates the e-process's Ville's-inequality
# guarantee empirically, ARMOR-FL's Byzantine-exclusion / drift-decoupling
# behavior, and the baseline aggregators)
python -m pytest tests/ -v

# fast end-to-end smoke test on synthetic data (~1 min)
python scripts/smoke_test.py

# end-to-end validation on a real CICIDS2017 sample (~20 min on CPU at
# reduced local_epochs=5; full-scale runs take much longer, see PROGRESS.md)
python scripts/real_data_smoke_test.py

# full experiment grid from a config
python scripts/run_experiment.py --config configs/cicids2017_robustness.yaml
```

## Journal target

*Cluster Computing* (Springer), single-blind review, LaTeX submission using
the `sn-jnl.cls` template with the `[iicol]` option. Abstract 100-150 words,
4-6 keywords, numbered `[N]`-style citations with DOIs, mandatory
Declarations section (Funding / Competing Interests / Author Contributions /
Data Availability) after the references.
