# Progress log

Read this first when resuming on a new machine. Most-recent entry on top.
See `README.md` for setup/run commands and `data_archive/README.md` for
dataset provenance.

## 2026-08-27

**Status: harness built, unit-tested, and validated end-to-end on real
CICIDS2017 data. No manuscript writing has started yet. CICIDS2018 not yet
obtained.**

### What's done

- **Core idea locked in**: ARMOR-FL replaces FedAvg's static `n_k/N` weight
  with a trust weight driven by a per-client anytime-valid e-process
  (Waudby-Smith & Ramdas betting-martingale construction), giving a formal,
  non-inflating false-exclusion guarantee at *any* stopping round (Ville's
  inequality) -- unlike fixed-checkpoint robust-aggregation baselines, whose
  type-I error compounds the more rounds you monitor. A second,
  self-referential e-process on each client's own local-loss trajectory
  decouples "drifting" (needs local adaptation) from "attacking" (needs
  downweighting/exclusion). Full derivation lives in the code comments of
  `armor_fl/fl/eprocess.py` and `armor_fl/fl/armor.py`.
- **Full harness implemented** (`armor_fl/`): data preprocessing +
  Dirichlet non-IID partitioning, a PyTorch reimplementation of the
  FedSE-1DSqueezeNet DDS backbone matching the paper's Table 1 layer-by-layer
  (params: 4649 for CICIDS2017 vs. their reported 4732 -- close, validates
  the reimplementation), 8-bit PTQ, ARMOR-FL aggregator, classic robust
  baselines (Krum/Multi-Krum, trimmed mean, coordinate median, FoolsGold),
  attack simulators (label-flip, sign-flip, Gaussian-noise, free-rider,
  ALIE), a custom single-machine FL simulation loop (no Flower -- full
  control over the aggregation-layer statistics was worth more than Flower's
  abstraction here), and a config-driven experiment runner.
- **Tests pass (11/11)**, including two statistically important ones:
  - `tests/test_eprocess.py`: Monte Carlo verification that the e-process's
    empirical false-exclusion rate stays flat at `alpha` whether monitored
    for 10 or 1000 rounds -- the actual property motivating this whole
    approach over fixed-checkpoint z-tests.
  - `tests/test_armor.py`: verifies a persistent Byzantine client gets
    downweighted/excluded while benign clients mostly survive (statistical,
    multi-seed check against `alpha_pop`), a merely-drifting-but-honest
    client is NOT excluded (drift/attack decoupling actually works), and
    exclude->probation->reinstate works.
- **Key implementation bug found and fixed during testing**: the e-processes
  must NOT self-calibrate their null mean from the client's own burn-in
  data -- a client that's already Byzantine during "burn-in" would calibrate
  itself as normal. Fixed by using a theoretically grounded fixed null_mean
  of 1.0 (since `z = dist / median(dists)` has median exactly 1 for the
  honest population by construction of the median). See the docstring note
  in `ArmorConfig` (`armor_fl/fl/armor.py`) -- this is worth stating
  explicitly in the manuscript's Method section as a design decision, not
  just a code comment.
- **Datasets**: CICIDS2017 (224MB) and CICIoT2023 (1.6GB) downloaded,
  preprocessing loaders written (`load_cicids2017`, `load_ciciot2023`, label
  taxonomies documented in `armor_fl/data/preprocessing.py`), and both
  committed to this repo as <100MB chunks under `data_archive/` per the
  user's explicit request (so no machine has to re-download from CIC/AWS --
  see `data_archive/README.md`). CICIDS2018 is NOT yet downloaded (was
  mid-`aws s3 sync` earlier in the session but no download appears to have
  completed/persisted -- check with the user or re-run:
  `aws s3 sync --no-sign-request --region <region> "s3://cse-cic-ids2018/" data_raw/cicids2018_raw`).
- **Real-data validation** (`scripts/real_data_smoke_test.py`, 5% CICIDS2017
  sample, deliberately reduced settings for speed): pipeline runs cleanly
  end-to-end, ~1400s for 8 rounds. ARMOR-FL caught 1/3 malicious clients
  (Gaussian-noise attack) with zero false positives in that short/reduced
  run -- consistent with the statistical detection-speed test, which shows
  full detection needs closer to the paper's full local_epochs=20/patience=5
  budget and more rounds than 8. Accuracy plateaued at exactly the BENIGN
  class's marginal share (80.3%) -- expected given only 5 local epochs (vs.
  the paper's 20) on a reduced sample, not a defect; needs a real run at
  full settings to reproduce meaningful accuracy numbers.
- **Interesting finding worth reporting honestly in the paper**: on
  synthetic data, `label_flip` (a data-space attack) was NOT detected by the
  population e-process, while `gaussian_noise` (a parameter-space attack)
  was caught with 100% precision/recall. This makes sense -- a label-flipped
  client trains honestly on bad labels and produces a plausible-looking
  parameter update, which is a known hard case for any *distance-based*
  robust-aggregation defense (Krum, trimmed mean, etc. have the same
  blind spot). Worth an explicit limitation/future-work paragraph, and
  possibly motivates adding a loss-based signal to the population
  e-process as a documented extension rather than silently patching around
  it.

### Journal / manuscript context gathered

- Submission guideline (Cluster Computing, Springer): single-blind review,
  LaTeX with the Springer Nature template (`[iicol]` option), abstract
  100-150 words, 4-6 keywords, numbered `[N]` citations with DOIs, mandatory
  Declarations section (Funding / Competing Interests / Author Contributions
  / Data Availability) after references.
- Self-citation plan (moderate, per COPE guidance against excessive
  self-citation): FORTRESS-FL (closest prior Byzantine-robust FL work --
  explicit differentiation needed, 10.1016/j.array.2026.100680), ST-FedXIDS
  (drift-adaptation precedent, cite for the drift half only, 10.29284/zcrk2p65),
  Uncertainty Measures for IDS (statistical-testing framing grounding,
  10.1007/s42979-026-04923-8). CIPHER/CONFIDE only if a DP or
  conformal-calibration extension is actually built.
- A `fork` subagent was dispatched to research additional 2024-2026
  robust/Byzantine-defense FL-IDS baselines and general SOTA IDS papers on
  these three datasets (to round out Related Work beyond what FedSE-1DSqueezeNet
  itself cites, and per the user's point that CICIDS2017/2018/CICIoT2023 have
  thousands of existing baseline papers to draw comparison numbers from) --
  check for its result if picking this up shortly after 2026-08-27.

### Next steps (in rough order)

1. Resolve CICIDS2018 (re-run the `aws s3 sync`, write `load_cicids2018`
   following the same pattern as the other two loaders once its columns are
   known, chunk+commit it into `data_archive/` the same way).
2. Pull in the fork's baseline-literature research results (if not already
   in this doc) and decide which numbers are worth a head-to-head comparison
   table vs. which are just Related Work citations.
3. Run the full experiment grid (`configs/cicids2017_robustness.yaml` /
   equivalents for the other two datasets once loaders exist) at real
   settings (local_epochs=20, patience=5, num_rounds=30). NOTE: at ~1400s
   for 8 reduced-setting rounds on 5% of one dataset, the full grid (7
   aggregators x 5 attacks x 4 malicious fractions x 4 non-IID levels x 30
   rounds x full dataset) will be very slow on this M-series Mac CPU --
   estimate the real budget before launching, consider trimming the grid,
   using MPS (`device: mps` in the config), or moving to Colab GPU per the
   user's standing instruction for GPU jobs.
4. Also run `configs/cicids2017_comparability.yaml` (R=5, FedAvg only, no
   attacks) to sanity-check accuracy numbers land in the same ballpark as
   the paper's own Table 3 before trusting the robustness-grid numbers.
5. Once results exist: draft the manuscript in `manuscript/` (LaTeX,
   `sn-jnl.cls`, `[iicol]`), following the Cluster Computing guidelines
   above and this user's global writing instructions (storytelling,
   comprehensive, reasoning-driven prose -- not a technical-blog tone).
6. Per the user's standing instruction, spawn a review agent after major
   milestones (e.g. after a full experiment pass, and again after a full
   manuscript draft).
