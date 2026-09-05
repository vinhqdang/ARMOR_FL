# Progress log

Read this first when resuming on a new machine. Most-recent entry on top.
See `README.md` for setup/run commands and `data_archive/README.md` for
dataset provenance.

## 2026-09-05 -- full 152-run grid complete; master results table built; starting manuscript

The v3 grid (post all three ARMOR fixes) finished cleanly: 152/152 runs,
zero failures, across `cicids2017_comparability`, `cicids2018_comparability`,
`cicids2017_robustness`, `cicids2018_robustness`, `ciciot2023_robustness`.
Took much longer than the ~1.3-day clean estimate due to repeated
GPU-sharing contention with other sessions on this machine (meta_prompt's
`run_grid.py`, briefly CRAFT-local's CRAFX_Net) -- see the session
transcript for the coordination history; nothing left running from those
now.

**`scripts/analyze_stability.py`** was added mid-grid after noticing
individual ARMOR runs oscillate heavily round-to-round (client subsampling +
slow re-detection after a reinstated attacker's e-process resets) -- e.g.
one run showed 0.815 acc at round 15 in a canary check vs 0.065 at round 25
in the full 25-round run, for the identical scenario. It computes both
final-round accuracy (primary, comparable across all 4 aggregators) and
mean-of-last-8-rounds (secondary, more stable) per run. Turned out oscillation
is NOT unique to ARMOR -- fedavg and especially foolsgold swing just as much
or more (foolsgold's worst-case gap was 0.64 on CICIDS2017, vs ARMOR's 0.44);
krum is genuinely the most stable of the four.

**`scripts/build_master_results_table.py`** combines all three datasets'
stability summaries into `results/master_robustness_table.csv` -- this is
the primary data source for the manuscript's Results section.

### Full aggregate picture (mean final-round accuracy / mean last-8-round accuracy, across all 12 attack/mal-fraction/alpha combos per dataset)

| aggregator | CICIDS2017 | CICIDS2018 | CICIoT2023 |
|---|---|---|---|
| fedavg (no defense) | 0.727 / 0.727 | 0.825 / 0.787 | 0.413 / 0.427 |
| krum | **0.810 / 0.781** | **0.810 / 0.809** | **0.618 / 0.623** |
| foolsgold | 0.715 / 0.720 | 0.718 / 0.740 | 0.378 / 0.381 |
| armor (ours) | 0.695 / 0.714 | 0.793 / 0.731 | 0.423 / 0.422 |

**Krum wins in aggregate mean on all three datasets, even after all three
ARMOR fixes this session.** This is the honest headline finding -- NOT a
"our method wins everywhere" result. ARMOR is roughly competitive with
fedavg/foolsgold in aggregate and clearly better than doing nothing on
average, but doesn't beat krum's aggregate mean on any of the three
datasets, despite winning specific combos (e.g. `label_flip, mal=0.4,
alpha=0.5` on CICIDS2017: armor=0.815 beats krum=0.799 and fedavg=0.190).

ARMOR's own detection precision stays low across all three datasets (mean
0.167-0.312) with benign false-exclusion rates of 0.125-0.163 -- meaning
even after fixing the three false-positive bugs found this session, genuine
detection-precision limitations remain, consistent with the per-run trace
finding (see the session transcript) that a reinstated attacker's e-process
reset gives it a long grace period before possible re-detection, and that
`label_flip` remains a genuinely hard-to-detect attack for any
distance-based signal (a known, shared blind spot, not unique to ARMOR).

### Manuscript

Starting the manuscript now via the ARS `academic-pipeline` orchestrator,
entering at Stage 2 (WRITE) since research/method/results already exist.
Per explicit user instruction: run the full write -> integrity -> review ->
revise -> re-review pipeline autonomously, repeating review/fix cycles as
many times as needed until convergence, without further check-ins beyond
the pipeline's own true integrity/review gates.

## 2026-09-04 -- stopped the full v2 grid mid-run; found and fixed a THIRD
## ARMOR bug via a new cheap canary tool; relaunched

**Built `scripts/canary_check.py`** after the user pointed out that
validating a fix by re-running the full 152-run grid (~1.3 days) is too
costly. It runs `{fedavg, krum, armor}` on a couple of attack types at
reduced settings (small `sample_frac`, fewer rounds) so a fix can be
validated in ~5-15 minutes instead. Also added `scripts/trace_armor_run.py`,
a full per-round diagnostic (weights/z/log_e_pop/log_e_drift/log_e_shift/
status per client per round) for tracing exactly what ARMOR is doing on one
real scenario, used to find the two bugs below.

**Important canary calibration lesson**: the FIRST canary settings
(`sample_frac=0.03, num_rounds=12, local_epochs=5`) were too weak --
every aggregator collapsed to near-random, so the "armor loses" verdict was
noise between two failures, not signal. Always sanity-check that fedavg vs
krum actually separate under the reduced settings before trusting a
comparison; `sample_frac=0.1, num_rounds=15, local_epochs=10` does.

**`scripts/trace_armor_run.py` gotcha**: importing `torch` before
`armor_fl.data.preprocessing` (which imports pandas/pyarrow) crashes with a
native access violation on this machine -- a known Windows conflict between
torch's bundled OpenMP/MKL and pyarrow's own threading, unrelated to CUDA.
Fix: import the pandas-dependent module first. `scripts/canary_check.py`
happened to already do this by accident; `trace_armor_run.py` didn't.

### The v2 grid was stopped mid-run (~2 of 152 done) once the canary found the next bug

The canary immediately paid for itself: at real-scenario settings
(`label_flip, mal=0.4, alpha=0.5, CICIDS2017`, the exact combo that exposed
the first bug), **ARMOR still lost badly to both fedavg and krum even after
the shift-gate fix** (armor=0.117 vs fedavg=0.189, krum=0.799) -- caught in
~6 minutes instead of after burning another day-plus on the full grid.

### Bug #2: the smooth trust-weight sigmoid was applied to the raw,
unbounded log-value instead of a threshold-relative one

Traced via `trace_armor_run.py` on the failing scenario: all four malicious
(label-flipping) clients kept ~full weight the ENTIRE run (e.g. client 4:
w~13,000-13,390 consistently, `log_e_pop` staying NEGATIVE throughout --
label-flipping happened to make this client's update look statistically
CENTRAL, the already-documented blind spot). Meanwhile entirely honest
clients with only mild population deviation were crushed to near-zero
weight (client 3: w dropped to 124-681 while `log_e_pop` was only ~0.5-1.1,
nowhere near the ~3.0 hard-exclusion threshold) -- because
`sigmoid(-beta * log_value)` with `beta=4.0` applied to the RAW log-value
already gives `trust_scale~0.018` at `log_value=1.0`, far short of where
exclusion actually triggers. Net effect: ARMOR's weighted average handed
MORE relative influence to the undetected attackers than an unweighted
average would, explaining the loss to plain fedavg.

Fix: rescale the sigmoid's input by the same threshold that gates hard
exclusion (`-log(alpha_pop)`), so `trust_scale~0.98` at the population
median, `~0.5` exactly at the exclusion boundary, and only drops toward 0
well past it. See the comment above `pop_threshold`/`normalized_pop` in
`armor_fl/fl/armor.py`.

This alone did NOT fix the scenario (armor actually got marginally worse,
0.069) -- see bug #3.

### Bug #3 (the big one): z = dist/MAD blows up when the honest population
is naturally homogeneous (near-IID), independent of any attack

Broke down the ALREADY-COMPLETED v1 grid results (pre-both-fixes, archived
under `results_v1_no_shift_gate/`) by non_iid_alpha instead of averaging
across it, and found something that had been hidden inside an aggregate
mean: ARMOR was actually COMPETITIVE with krum under `label_flip` in the
non-IID case (`alpha=0.5`: armor 0.746-0.809 vs fedavg 0.485-0.799) but
**catastrophic specifically under IID** (`alpha=None`: armor 0.086-0.108 vs
fedavg 0.233-0.825, krum 0.838-0.904), with benign false-exclusion rates of
33-62.5%. This is backwards from intuition -- IID should be the EASIER
case for a population-comparison test, not the harder one.

Mechanism: `z = dist / MAD` where MAD is this round's cross-sectional
median distance from the robust center. When honest clients are naturally
similar (IID data, or just an unlucky small 6-of-10 sample with tight
agreement), MAD collapses toward zero -- and dividing ordinary training
noise (different batch orders, small-sample SGD variance) by a near-zero
MAD manufactures enormous z-scores from clients that aren't attacking at
all. This is a different failure mode from bug #1 (persistent structural
heterogeneity being mistaken for an attack) -- this one is about the
POPULATION's own natural spread shrinking, not about any one client's
persistent offset, and it hits hardest exactly when there's LESS
heterogeneity, not more.

Fix: floor MAD at a fraction (`mad_floor_fraction=0.5`) of its own recent
history (median of `ArmorAggregator.mad_history`, which stores only the RAW
per-round MAD, never the floored value, so there's no upward ratchet). This
can only ever RAISE the effective MAD relative to the raw computed value, so
it can never mask a real attack -- a genuine attack's raw MAD is already at
least as large as its own floor and passes through unchanged.

Validated via canary (`label_flip, mal=0.2, near-IID` -- used
`--non-iid-alpha 1000000` as a Dirichlet proxy for true IID before adding
proper `none`/`iid` support to the canary script): **armor=0.840 vs
krum=0.841, fedavg=0.803, benign_FER=0.000** -- complete turnaround from
v1's 0.086/62.5%-false-exclusion catastrophe on this combo. Re-validated the
original failing scenario too (`label_flip, mal=0.4, alpha=0.5`):
**armor=0.815, now BEATING krum=0.799** (was 0.117/0.069, losing to both,
earlier this session).

All 18 unit tests still pass (could not cleanly reproduce bug #3 in the
existing lightweight synthetic-client test harness -- real neural-network
training noise across clients is messier than a clean Gaussian-perturbation
model; relied on the canary's real-data validation above instead of a new
unit test for this one).

### What's still open / known-not-fixed

- `gaussian_noise` under `mal=0.4, alpha=0.5`: armor and fedavg both land on
  the exact majority-class-baseline collapse (0.8030/0.7153) while krum
  clearly wins (0.865/0.876) -- not a regression from this session's fixes,
  just a case ARMOR doesn't yet handle better than doing nothing.
- The documented bug #1 limitation stands: a from-round-1 constant attacker
  below `gross_outlier_z` is still only smoothly downweighted, not
  hard-excluded -- behavioural data alone can't separate that from a
  heterogeneous honest client.

### Relaunching

v1 results are archived under `results_v1_no_shift_gate/` (gitignored) for
comparison. Relaunching `scripts/run_all_experiments.sh` fresh (results/
directory will be overwritten) now that the canary has validated all three
fixes on the scenarios that mattered. GPU confirmed free (single process)
before launch each time -- see the process-cleanup notes in earlier entries
for why this matters (the driver script's outer bash loop has repeatedly
survived `TaskStop`/one `taskkill`, needing a second explicit kill of the
outer PID).

## 2026-09-01 -- second grid-trimming pass, GPU free again, relaunched

The other session's GPU-sharing job finished; confirmed via `nvidia-smi`
(802 MiB used, no compute processes) before doing anything else.

User felt even the already-trimmed 428-run_id grid was still too big.
Cut further (user's explicit choices, since this is a paper-narrative call):
- **Aggregators 7->4**: `fedavg, krum, foolsgold, armor` -- naive baseline +
  one classic Byzantine-robust baseline + one modern reputation-based
  baseline + our method. Dropped `multi_krum`, `trimmed_mean`,
  `coordinate_median` as redundant with `krum` for the comparison story.
- **Attack types 5->3**: `label_flip, gaussian_noise, alie` -- one
  representative per category (data-space / simple parameter-space /
  sophisticated-adaptive). Dropped `sign_flip`, `free_rider` as similar in
  kind to `gaussian_noise`.
- **Kept both `malicious_fractions` levels** ([0.2, 0.4]) and both
  `non_iid_alphas` levels ([null, 0.5]) -- user wanted to preserve those
  sweeps.
- Also removed the now-unused `multi_krum`/`trimmed_mean` entries from each
  config's `aggregator_kwargs`.

New grid: 4 x 3 x 2 x 2 = 48 run_ids/dataset x 3 datasets = 144, plus the 8
comparability run_ids = **152 total** (down from 428, down from the original
1680). At the batch_size=256 clean per-round rate (25.9s/round x 25 rounds
= ~647.5s/run_id), clean estimate is roughly **~1.3 days** for the whole
sequence -- a large improvement, assuming the GPU stays uncontended this
time.

Relaunched `bash scripts/run_all_experiments.sh` (same driver script, will
redo `cicids2017_comparability` from its first run_id since the earlier
`alpha=None` result gets naturally overwritten -- fine, it's fast and
should reproduce closely).

## 2026-09-01 -- paused: unrelated GPU-sharing job made this untenable, resume when it's done

**Deliberately stopped the experiment sequence** (`scripts/run_all_experiments.sh`,
launched 2026-08-31 08:05) at the user's explicit request, rather than letting
it keep fighting for the GPU. Do not restart it until the reason below is gone.

### Why

This machine's GPU was being shared, uncoordinated, with a `scripts/run_ablation.py`
process from a completely unrelated project/session (`super_resolution`) that
happened to start 12 minutes after ours (08:17 vs 08:05) and ran for over a day.
Confirmed directly with that session: it independently measured the same
**~20x slowdown** on its own workload, so this wasn't a bug on either side --
it's a real property of two small-kernel-launch-heavy CUDA workloads sharing
one GPU under Windows' WDDM scheduler (no true concurrent kernel execution
like Linux MPS/TCC; every switch between the two processes' contexts is an
expensive full context swap, worse when VRAM is nearly maxed at ~15.9/16.3 GB
by both jobs at once -- see the session transcript for the full explanation
given to the user).

Net effect: run 2/428 (`fedavg__atk=None__mal=0.0__alpha=1.0` on
`cicids2017_comparability`) was still not done after **~24 hours** (clean
estimate: ~35 min). Continuing to grind through the full grid under these
conditions would have taken an unpredictable multiple of the already-long
estimate, for no good reason when the fix is simply "wait for the other job
to finish, then resume with a GPU to ourselves."

### What's actually done vs. lost

- **Complete and saved**: `results/cicids2017_comparability/fedavg__atk=None__mal=0.0__alpha=None.json`
  (the IID run: acc=0.8519, f1=0.8066, 6343.9s under partial contention).
- **Lost, not resumable**: the in-progress `alpha=1.0` run (~24h of compute).
  `run_experiment.py` only writes a result JSON after a run_id fully
  completes -- there's no mid-run checkpoint, so this one has to be redone
  from scratch. Not worth adding checkpointing for a single lost run; revisit
  only if pausing mid-grid becomes a recurring need.
- Everything else in the grid (all of `cicids2018_comparability` and all
  three `*_robustness.yaml` configs, 424 run_ids) hadn't started yet --
  nothing else was lost.

### How this was stopped (for the record, in case cleanup is ever incomplete)

`TaskStop` on the tracked background task didn't actually kill the process
tree (the outer `bash scripts/run_all_experiments.sh` loop survived it and
promptly started the next config after the inner python process died) --
had to `taskkill /F /T` the actual PIDs directly, twice, once for the
in-progress python process and once for the bash driver script itself once
it was noticed still alive and having launched `cicids2018_comparability`.
Verified clean via `nvidia-smi` (memory dropped to only the other session's
usage) and `ps aux` (no more `run_experiment.py` or `run_all_experiments`
processes). The two `tail -f` log-watching Monitor tasks were stopped too.

### How to resume

1. Confirm the other GPU job is actually done: `nvidia-smi` should show
   only ~1 process using the GPU (this machine's own), not two.
2. Re-run `bash scripts/run_all_experiments.sh > /tmp/run_all_experiments.log 2>&1`
   (**as ONE properly-backgrounded call -- do not combine a trailing `&`
   with the tool's own background-tracking flag, that caused a separate
   stray-process incident earlier this session**). It will redo
   `cicids2017_comparability` from its first run_id (`fedavg`/IID) again --
   fine, it's fast and the existing JSON will just be overwritten with an
   equivalent result under clean conditions.
3. Re-benchmark one run_id first if there's any doubt the GPU is really
   uncontended before committing to the full sequence again.

## 2026-08-31 -- launched the full experiment sequence

Kicked off `bash scripts/run_all_experiments.sh` (new script -- runs
`cicids2017_comparability`, `cicids2018_comparability`, then the three
`*_robustness.yaml` grids, sequentially, since running configs concurrently
previously caused GPU contention -- see the batch_size entry below) as a
background process, logging to `/tmp/run_all_experiments.log`. Confirmed
clean startup (first run_id `fedavg__atk=None__mal=0.0__alpha=None` on
`cicids2017_comparability` began without error, single process, GPU
active).

**Status/progress check for whoever picks this up next**: tail
`/tmp/run_all_experiments.log` for the `== START:`/`== END:` markers per
config and the per-run_id `===` lines; each config also writes
`results/<config_name>/summary.csv` and one `<run_id>.json` per run as it
completes, so partial progress is inspectable even before a config finishes.
Expected order of completion: both comparability configs first (cheap, ~2-3
hours each), then `cicids2017_robustness` / `cicids2018_robustness` /
`ciciot2023_robustness` (the ~3.1-day estimate from the batch_size entry
below applies to this stage). Do not launch a second `run_experiment.py`
process concurrently with this one -- confirm via `nvidia-smi`'s process
list and `ps aux` that only one is running first (see the "process-
management mistake" note below for why this matters).

Next steps once this finishes: sanity-check the comparability results
against the paper's own Table 3 numbers before trusting the robustness
grid's accuracy figures, then start drafting the manuscript per the
Cluster Computing submission guidelines noted further down this file.

## 2026-08-28 -- batch_size was the real lever; trimmed grid now ~3 GPU-days

Follow-up to "grid trimming round 2" immediately below. Asked the user how
to close the remaining gap (37-50+ estimated GPU-days even after all four
previously-approved levers); they picked **increase batch_size**.

**Process-management mistake first, worth flagging so it isn't repeated**:
while benchmarking batch sizes, several earlier `Bash` calls had combined a
trailing shell `&` with the tool's own `run_in_background: true` on the same
command. The tool considered those "completed" the instant the outer shell
returned, but the actual `conda run ... python ...` process kept running
detached and untracked. Three separate stray Python processes ended up
alive simultaneously, all fighting over the same GPU (confirmed via
`nvidia-smi`'s process list, ~16/16.3 GB VRAM used, GPU stuck at low-power
P4 state) -- this is why an early batch-size run silently produced an empty
log (buffered stdout lost when a process is killed/contended) and why a
`local_epochs=10` re-check earlier in the session read 308.3s/round instead
of the true single-process number. Killed all three stray PIDs via
`taskkill`; every timing number below is from a single clean process.
**Lesson: never combine a trailing `&` with `run_in_background: true` on the
same call -- pick exactly one.**

### The real finding

Same representative run_id (armor aggregator, gaussian_noise, non_iid_alpha=
0.5, 10% CICIDS2017 sample), `local_epochs=10`/`patience=3`, 5 rounds, varying
only `batch_size`, single clean process, this machine:

| batch_size | s/round | trimmed-grid extrapolation (420 run_ids) |
|---|---|---|
| 32 | 211.5 | 25.7 days |
| 128 | 47.1 | 5.7 days |
| 256 | 25.9 | 3.1 days |
| 512 | 10.7 | 1.3 days |

Batch size was the dominant lever all along -- far more than the
combinatorics/epoch/round cuts combined, consistent with the original
"GPU utilization 5-18%" finding: a tiny model at `batch_size=32` is almost
entirely kernel-launch-overhead-bound, not compute-bound, on this GPU.
Accuracy/F1 were statistically indistinguishable across batch sizes in this
short check (acc=0.803, f1=0.715 at bs=32/256/512; bs=128 landed at
acc=0.805 -- within run-to-run noise for a 5-round smoke test, not a real
comparison).

### Decision: batch_size=256 for the robustness grids only

Chose **256** (not the fastest 512) as a defensible middle ground -- ~3.1
days total is comfortably runnable, without taking the most extreme
deviation from the paper's own `batch_size=32` (Table 2) when a smaller
deviation gets most of the benefit. Applied to all three
`*_robustness.yaml` files only. **`*_comparability.yaml` configs
deliberately kept at `batch_size=32`** -- their entire purpose is fidelity
to the paper's own protocol, so they shouldn't take this deviation. All 17
tests still pass after the change.

This is a hyperparameter deviation from Table 2 and needs to be stated
plainly in the manuscript's Method/Limitations section as a practical
compute-budget decision, the same way the `label_flip` detection blind spot
was flagged earlier in this file -- not quietly folded in as if it were the
paper's original protocol.

## 2026-08-28 -- grid trimming round 2, on a new (slower) GPU machine -- still infeasible, blocked again

Picked up the deferred grid-trimming decision from the entry below. The user
approved all four levers that had been offered: restructure the training
loop first, cut grid combinatorics, reduce `local_epochs`, and reduce
`num_rounds`/`sample_frac` further. This entry documents what actually
happened when those were implemented -- headline: **even after all four,
the full 3-dataset grid is still an estimated 5-8+ weeks of continuous GPU
time on this machine**, which is a materially different (worse) outcome than
"cut it down to something runnable," so this is flagged back to the user
again rather than silently pushing further cuts.

### New machine, new GPU

This session runs on a different physical machine than the one that produced
the ~102-day / 175.3 s/round estimate below -- an NVIDIA RTX 5000 Ada
Generation Laptop GPU (16 GB), driver 596.58, CUDA 13.2, PyTorch 2.6.0+cu124.
`data_raw/` had to be regenerated via `bash scripts/reassemble_datasets.sh`
(not pushed, per the note below); all three datasets reassembled cleanly and
checksums matched.

**A real, unrelated CUDA driver bug surfaced immediately**: the full test
suite crashed reliably (`Windows fatal exception: access violation`, native
crash inside `torch._dynamo`'s eval-frame hook) at the exact same test every
time, but only when CUDA tests ran after the CPU-only test files earlier in
the same pytest session -- `tests/test_device_consistency.py` alone always
passed. Confirmed via `git stash` that this reproduces identically on the
unmodified pre-session code, so it's an environment issue, not something the
training-loop changes below introduced. Root cause: CUDA's lazy module
loading (default in recent driver/CUDA combos) misbehaving with this
PyTorch build on Windows. Fix: `CUDA_MODULE_LOADING=EAGER`, which resolved it
across repeated full-suite runs. Persisted permanently for this machine via
`conda env config vars set CUDA_MODULE_LOADING=EAGER -n py313` (survives
`conda activate`, doesn't need to be remembered per-command).

### Training-loop restructure (`armor_fl/fl/client.py`, `armor_fl/fl/simulate.py`)

Implemented the "attack the actual overhead" lever from the options list
below, without changing any hyperparameter that affects what's being
reproduced:
- `local_train`: replaced `DataLoader`/`TensorDataset` with manual
  `torch.randperm` + slicing directly on already-GPU-resident tensors --
  same shuffle-every-epoch, same fixed `batch_size`, just without
  DataLoader's per-batch Python/collate overhead.
- `run_simulation`: train/test tensors are moved to `cfg.device` once up
  front (previously each client's slice was transferred fresh every
  client-round); a single local-training model is constructed once and
  reset via `load_state_dict` per client instead of calling
  `model_factory()` (and `.to(device)`) fresh every client-round.
- Found and fixed a latent crash this restructure would have hit: a
  leftover final batch of exactly 1 sample crashes `nn.BatchNorm1d` in
  train mode ("expected more than 1 value per channel"). Added an explicit
  skip for `batch_idx.numel() < 2`. (`DataLoader`'s default `drop_last=False`
  had the same latent bug -- this wasn't newly introduced, just newly
  exercised by manual batching in the test suite's tiny synthetic
  partitions.)
- All 17 existing tests still pass, including the 6 CUDA
  device-consistency tests.

### The wall-clock reality check

Same representative run_id as the original benchmark (armor aggregator,
gaussian_noise attack, non_iid_alpha=0.5, 10% CICIDS2017 sample, 226k train
rows), measured on **this machine** both before and after the restructure
(same-machine comparison, since the ~102-day estimate below was from
different hardware and isn't a fair baseline here):

| variant | local_epochs/patience | s/round |
|---|---|---|
| pre-restructure (this machine) | 20 / 5 | 564.0 |
| post-restructure | 20 / 5 | 426.0 |
| post-restructure | 10 / 3 | 308.3 |

Two findings worth internalizing:
1. **The restructure bought ~1.32x**, not the large multiple hoped for.
   This machine's laptop GPU is simply much slower per-round than whatever
   produced the earlier 175.3 s/round figure -- the earlier estimate isn't
   a fair comparison target on this hardware.
2. **Cutting `local_epochs` 20->10 only bought ~1.38x, not 2x** --
   `patience=5`/`patience=3` was already triggering early stopping before
   `max_epochs` in most client-rounds, so the epoch cap isn't the binding
   constraint most of the time; the fixed per-round cost (client selection,
   aggregation, eval, model reset) is a bigger share of the total than
   expected.

### What was actually applied to the configs

All three `*_robustness.yaml` (and both `*_comparability.yaml`) now set
`device: cuda`. Robustness grids: `local_epochs: 10`, `patience: 3`,
`num_rounds: 25` (matches the statistical floor `test_armor.py` validates),
`malicious_fractions: [0.2, 0.4]` (was 4 levels), `non_iid_alphas: [null,
0.5]` (was 4 levels) -- combinatorics 560->140 run_ids/dataset (420 total,
down from 1680). `sample_frac`: cicids2017 null->0.1 (matches the benchmark
exactly -- was a previously-unaccounted ~10x hidden cost), cicids2018
0.05->0.02, ciciot2023 0.02->0.01.

**New estimate at these settings: ~899 GPU-hours (~37.5 days) just for a
single CICIDS2017-scale run through the trimmed 420-run_id grid** (140
run_ids/dataset x 25 rounds x 308.3 s/round, extrapolated across all 3
datasets assuming similar per-row cost -- CICIDS2018 and CICIoT2023 will
likely run somewhat higher since their trimmed sample_fracs still leave
larger absolute row counts than CICIDS2017's 226k). **This is not "something
runnable" in any casual sense** -- it's 5-8+ weeks of continuous GPU time on
a laptop, which is a different problem (can the machine even stay on and
undisturbed that long?) than the ~102-day number this was originally framed
against.

### Blocked again -- flagged back to the user rather than cutting further unilaterally

The four approved levers are now implemented as far as they reasonably go
without a new kind of decision:
- Combinatorics can be cut further (e.g. drop to 3-4 aggregators or 3
  attack types) but that directly shapes which comparisons the manuscript
  can make -- a paper-narrative call, not an engineering one.
- `batch_size` (currently 32, matching Table 2) was never actually the
  lever exercised -- GPU utilization is still well under 100% at this batch
  size on this GPU, so increasing it would likely help a lot, but it's a
  hyperparameter deviation from the paper's own protocol and wasn't part of
  what was approved.
- Running multiple run_ids concurrently as separate processes on the same
  GPU (no science change, since GPU utilization has headroom) is untested
  and might not scale well if the bottleneck is Python/CPU-side overhead
  rather than raw GPU compute -- worth trying but unproven.
- Simply accepting a multi-week unattended background run is also a valid
  choice, just one with real practical constraints on a laptop.

Not picking one of these unilaterally; see next-steps/decision needed at the
top of this file structure going forward.

## 2026-08-28 -- session paused here, decision needed on the next machine

Everything below (2026-08-27, "latest 4" through "latest") is committed and
pushed (`main` at `8b56745`). **Stopped at exactly the decision point in
"Options for trimming" below** -- the user was asked to pick which lever(s)
to pull to cut the full grid down from its current ~102-day estimate, and
deferred that choice to continue on a different machine rather than answer
inline. Nothing has been trimmed yet; the grid configs (`configs/*_
robustness.yaml`) are unchanged from what's described below. **Read the
"The wall-clock finding" and "Options for trimming" sections just below
before running anything at real settings** -- picking up and just running
`scripts/run_experiment.py` on a robustness config as-is will not finish in
any reasonable time.

Practical note for picking this up: `data_raw/` (all three datasets,
reassembled via `bash scripts/reassemble_datasets.sh`) and the leftover
`*.tar.gz`/`*.zip` reassembly intermediates are gitignored and were never
pushed -- a fresh machine needs to re-run the reassemble script (fast, no
re-download, just cat+extract from the committed `data_archive/` chunks)
before any dataset loader will find files.

## 2026-08-27 (latest 4)

**Found and fixed a real, previously-unexercised CUDA bug while running the
wall-clock benchmark**, then got the actual GPU numbers -- headline finding:
**the full experiment grid as currently specified is infeasible on this
machine (~102+ days of continuous GPU compute), even after the bug fix.**
Blocked on the user for how to trim it -- see below.

### The bug

Every existing config/test used `device: cpu`, so a real device-consistency
bug had zero coverage: `global_vector` (the running aggregate, built once
outside the per-round loop from a freshly-constructed, CPU-resident
`model_factory()`) was never moved to `cfg.device`, while each round's local
training moved its own model to `cfg.device` inside `local_train`. First
crash: `honest_delta = local_vector - global_vector` in
`armor_fl/fl/simulate.py` ("Expected all tensors to be on the same device").
Chasing it down surfaced the same class of bug in three more places, all
fixed together (see `armor_fl/fl/simulate.py`, `robust_stats.py`,
`attacks.py`, `client.py`):
- `simulate.py`: `global_model = model_factory().to(cfg.device)` (was
  missing `.to()` entirely).
- `robust_stats.py`: `weighted_average` and `geometric_median` built their
  weights tensor with `torch.tensor(weights, dtype=torch.float32)` --
  defaults to CPU regardless of the vectors being combined; now derives the
  device from the (already-device-consistent) stacked vectors.
  `krum_select`'s `dist_matrix` got the same treatment for consistency.
- `attacks.py`: `gaussian_noise` and `free_rider` called `torch.randn(...)`
  without a `device=` kwarg, so the injected noise was always CPU-resident;
  fixed to inherit the delta's device.
- `client.py::evaluate`: didn't move `model` to `device` at all (only `X`/
  `y`) -- added `model = model.to(device)` for defense-in-depth, since it
  had been silently relying on the caller already having placed the model
  correctly.

Added `tests/test_device_consistency.py` (6 tests, `skipif(not
torch.cuda.is_available())`) covering the exact crash path
(`run_simulation` end-to-end on cuda for fedavg/krum/foolsgold/armor) plus
the lower-level helpers directly. All 17 tests (11 original + 6 new) pass.
This is exactly the kind of bug that stays invisible indefinitely on a
CPU-only machine -- worth remembering: CPU is also PyTorch's default
device, so a missing `.to(device)` is silent until a second device enters
the picture.

### The wall-clock finding

`scripts/benchmark_fl.py` (new): one representative run_id (aggregator=
armor, gaussian_noise attack, non_iid_alpha=0.5, num_clients=10,
client_fraction=0.6, **local_epochs=20, patience=5** -- i.e. real settings,
not smoke-test-reduced), on a 10% CICIDS2017 sample (226k train rows), GPU:

- **175.3 s/round** (3-round measurement, post-bug-fix)
- extrapolated to the real `num_rounds=30`: **~87.6 min per run_id**
- extrapolated to the full grid (7 aggregators x 5 attacks x 4 malicious
  fractions x 4 non-IID alphas = 560 run_ids/dataset x 3 datasets = 1680
  run_ids): **~2454 GPU-hours (~102 days) of continuous compute**

And this is an *underestimate* for two reasons: (1) `cicids2017_robustness.
yaml` actually sets `sample_frac: null` (full 2.26M rows, ~10x this
benchmark's sample), and (2) GPU utilization during the benchmark was ~5-18%
-- confirmed via `nvidia-smi` and `Get-Process` CPU-time sampling while
investigating why it looked idle -- meaning the bottleneck is per-round/
per-client Python and DataLoader overhead across 10 clients x up to 20
local epochs, not matrix-multiply throughput. **A faster GPU would not fix
this**; the levers that matter are grid size, `num_rounds`, `local_epochs`,
and possibly restructuring the per-client training loop itself.

### Options for trimming (needs a decision, not made unilaterally here)

- Cut grid combinatorics (e.g. non_iid_alphas 4->2, malicious_fractions
  4->2 -- a 4x cut lands near ~26 days, still likely too slow alone).
- Cut `local_epochs` (currently 20, matching the paper's Table 2 -- lowering
  this changes what's being reproduced, so this specifically needs the
  user's sign-off, not just an engineering call).
- Cut `num_rounds` below 30 (the e-process needs enough rounds to
  accumulate evidence past `burn_in_rounds=3`, so there's a floor below
  which the detection-speed story breaks -- test_armor.py's statistical
  checks pass at `num_rounds=25`, so ~25-30 is probably close to the real
  floor already).
- Reduce `sample_frac` further across all three datasets.
- Restructure local training to reduce per-round Python overhead (batch
  multiple clients' training instead of a fresh model+DataLoader per client
  per round) -- a real engineering investment, not a config change.
- Parallelize across multiple runs/machines (doesn't reduce total compute-
  hours, only wall-clock if run concurrently).

Realistically this needs several of these combined, and the right mix
depends on which grid cells the manuscript's narrative actually needs a
head-to-head number for -- flagged to the user rather than choosing
unilaterally.

## 2026-08-27 (latest 3)

**Added a centralized XGBoost baseline (`scripts/xgboost_baseline.py`)
per the user's suggestion**, before committing compute to the full
ARMOR-FL grid: "run it to test if it is really good, then we fully run
it, so we need some baseline models." This is *not* a robustness
comparison method (it sees all training data pooled, no client
partitioning, no attacks) -- it's a fast ceiling reference to sanity-check
each dataset is learnable at all and to know how much headroom the FL
approach has below a strong classical baseline in the clean/IID case.
Same loaders, `sample_frac`, stratified split, and metric set
(accuracy/weighted-F1/weighted-precision/weighted-recall/macro-F1, matching
`armor_fl.fl.client.evaluate`) as the FL grid, for direct comparability.
`xgboost>=3.4` (3.4.1 installed) added to `requirements.txt`. Also
reassembled all three datasets locally (`bash scripts/reassemble_datasets.sh`)
and found `.gitignore` had `*.zip` but not `*.tar.gz` -- the reassembled
CICIDS2018 tar.gz landed as an untracked 1.6GB file at repo root; fixed the
gitignore gap and deleted the (regenerable) leftover archives.

Results (GPU, `results/baselines/*.json`):

| Dataset | sample_frac | train rows | accuracy | F1 (weighted) | F1 (macro) |
|---|---|---|---|---|---|
| CICIDS2017 | full | 2.26M | 0.9990 | 0.9990 | 0.9316 |
| CICIDS2018 | 0.05 | 649k | 0.9842 | 0.9806 | 0.7992 |
| CICIoT2023 | 0.02 | 720k | 0.8561 | 0.8377 | 0.6601 |

**Takeaways**: CICIDS2017 is nearly saturated by a classical baseline (macro-F1
still capped by the extremely rare Heartbleed/Infiltration classes noted
earlier). CICIDS2018 and especially CICIoT2023 have substantially more
headroom (macro-F1 0.80 and 0.66) -- consistent with known literature
findings that these are harder multi-class problems (CICIoT2023's 8-class
grouping from 34 raw attack types, some with very similar flow signatures).
This sets realistic accuracy-ceiling expectations before evaluating
ARMOR-FL's own numbers: hitting >90% macro-F1 on CICIoT2023 would be
suspicious, not impressive. Worth citing these baseline numbers directly in
the manuscript's experimental setup as an "is this dataset/feature set
learnable" sanity anchor, separate from the robustness-comparison story.

Loading `_load_and_group` is the dominant cost even with `sample_frac` set
(CICIDS2018 sample took 397s to load vs 23s to train) -- it reads full CSVs
before subsampling per-file. Not a blocker for now, but worth optimizing
(e.g. `pandas.read_csv` with `skiprows`/chunked reading, or caching a
subsampled parquet) if it becomes a bottleneck for repeated grid runs.

## 2026-08-27 (latest 2)

- **Step 2 of the next-steps list done**: `configs/cicids2018_robustness.yaml`
  and `configs/cicids2018_comparability.yaml` added (mirror the CICIDS2017
  ones), and `cicids2018` -> `load_cicids2018` wired into
  `scripts/run_experiment.py`'s `DATASET_LOADERS` registry. Robustness config
  uses `sample_frac: 0.05` (~810k rows of the 16.2M-row full dataset, same
  order of magnitude as the CICIoT2023 sampled set); comparability config
  uses `sample_frac: 0.2` (~2.6M rows, roughly CICIDS2017's full size) since
  even an R=5 FedAvg-only sanity check isn't worth the wall-clock at full
  16.2M-row scale. README's dataset table was also stale (still said
  CICIDS2018 "not yet downloaded/archived" from before the prior commit
  archived it) -- fixed. All 11 tests still pass; registry import verified.
  `data_raw/` is not populated on this machine yet (gitignored, needs
  `bash scripts/reassemble_datasets.sh`) so the CICIDS2018 loader itself was
  not re-exercised end-to-end here, only via the existing test suite + a
  dry import check. Next up is still step 3: estimate the real compute
  budget and run the full grids.
- **Compute note for step 3**: picking this up on a Windows machine
  (`C:\work\ARMOR_FL`) with an NVIDIA RTX A2000 8GB laptop GPU, CUDA
  available and `torch.cuda.is_available()` True (torch 2.11.0.dev+cu128).
  All configs currently default to `device: cpu` for portability (the prior
  session's M-series Mac had no CUDA) -- pass `device: cuda` in the config
  (or override) when actually launching the full grids from this machine.
  The backbone is tiny (4649 params) so most of the wall-clock is probably
  data loading / per-round Python overhead rather than matmul-bound, but
  worth benchmarking one run_id both ways before committing to the full
  grid's time budget.

## 2026-08-27 (latest)

**Status: harness built and unit-tested; all three datasets downloaded,
loaders written, and archived to this repo. Related-work/baseline literature
research done. No manuscript writing has started yet. No full-scale
experiment run yet -- that's the next real blocker.**

### Since the entry below

- **All three datasets now archived in `data_archive/`**: CICIDS2018 (6.5GB
  raw -> 1.6GB gzipped) synced via `aws s3 sync --no-sign-request` from just
  the `Processed Traffic Data for ML Algorithms/` prefix (NOT the raw
  pcap/log half of the bucket, which is far larger and not needed), 10 daily
  CSVs, 16.2M rows. Known data-quality quirk confirmed and handled: ~59 rows
  have `Label=="Label"` (CICFlowMeter's header re-emitted mid-file on one
  day) -- these are silently and correctly dropped by the existing
  unmapped-label path in `_load_and_group`, not a special case needed.
  `load_cicids2018` added to `armor_fl/data/preprocessing.py`, 7-class
  taxonomy (BENIGN/DoS/DDoS/BruteForce/WebAttack/Bot/Infiltration -- no
  PortScan or Heartbleed in this dataset), verified on a 2% sample.
  Re-archived as `.tar.gz` (source is loose CSVs, not a zip) since gzip
  compresses this text data ~4x.
- **Baseline literature research complete**: see
  `manuscript/related_work_candidates.md` for two full tables (12
  robust/Byzantine-defense FL-IDS papers, 8 recent general SOTA IDS papers)
  plus a gap analysis. Headline finding worth building the paper's novelty
  framing around: **no robust-aggregation FL-IDS paper was found with
  reported attack-robustness numbers on CICIoT2023 specifically** -- ARMOR-FL
  would be a first mover there. Also: almost none of the 12 defense papers
  give formal statistical guarantees (they're empirical clustering/distance/
  reputation heuristics), and none disambiguate "drifting" from "attacking"
  -- both are ARMOR-FL's actual differentiators, now backed by a literature
  gap rather than just first-principles reasoning.
- **Agent-orchestration note**: the first `fork` subagent dispatched for this
  research task went out of scope (wrote/committed/pushed unrelated files
  across two separate turns, then got confused about its own identity when
  asked to just report findings as text -- see feedback drafts sent this
  session). Abandoned it and used a fresh `general-purpose` agent instead,
  which worked cleanly. Worth remembering if delegating research-only tasks
  again: a fresh agent may be more reliable than a fork when the task
  shouldn't touch files/git at all.

## 2026-08-27 (earlier)

**Status at this point: harness built, unit-tested, and validated end-to-end
on real CICIDS2017 data. CICIDS2018 not yet obtained.**

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

1. Decide which Table 1 entries in `manuscript/related_work_candidates.md`
   are worth an actual head-to-head numerical comparison (vs. just Related
   Work citations) -- most only have partial/unconfirmed metrics, so this
   needs a closer read of the actual papers, not just the abstracts.
2. Write `configs/cicids2018_robustness.yaml` and
   `configs/cicids2018_comparability.yaml` (mirror the CICIDS2017 ones;
   `configs/ciciot2023_robustness.yaml` already exists). Wire `cicids2018`
   into `scripts/run_experiment.py`'s `DATASET_LOADERS` registry.
3. Run the full experiment grids at real settings (local_epochs=20,
   patience=5, num_rounds=30) across all three datasets. NOTE: at ~1400s for
   8 reduced-setting rounds on 5% of CICIDS2017 alone, the full grid (7
   aggregators x 5 attacks x 4 malicious fractions x 4 non-IID levels x 30
   rounds x full dataset, x3 datasets) will be very slow on this M-series Mac
   CPU -- estimate the real budget before launching, consider trimming the
   grid, using MPS (`device: mps` in the config), or moving to Colab GPU per
   the user's standing instruction for GPU jobs. CICIDS2018's
   Tuesday-20-02-2018 file alone is 3.9GB/one day, so sample_frac is close to
   mandatory there too (see the CICIoT2023 config's sample_frac=0.02 pattern).
4. Also run the `*_comparability.yaml` configs (R=5, FedAvg only, no attacks)
   per dataset to sanity-check accuracy numbers land in the same ballpark as
   the paper's own Table 3 before trusting the robustness-grid numbers.
5. Once results exist: draft the manuscript in `manuscript/` (LaTeX,
   `sn-jnl.cls`, `[iicol]`), following the Cluster Computing guidelines above,
   this user's global writing instructions (storytelling, comprehensive,
   reasoning-driven prose -- not a technical-blog tone), and the novelty
   framing in `manuscript/related_work_candidates.md`'s gap analysis.
6. Per the user's standing instruction, spawn a review agent after major
   milestones (e.g. after a full experiment pass, and again after a full
   manuscript draft).
