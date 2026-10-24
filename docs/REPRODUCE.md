# Runbook — from the raw corpora to the camera-ready numbers

For the person running it (on your own machine). **Nothing in this repository has seen the real corpora yet**: every dataset host was
unreachable from the environment where the code was prepared, so the loaders were tested on stand-in files that copy the documented
column headers. The runbook is ordered so that every step that can fail does so early and cheaply.

## 0. Before you start (≈ 15 min)

1. `pip install -e .[dev]` (Python ≥ 3.9). PyTorch on the CPU is enough: the model is a 64-32 MLP and every worker uses one thread.
2. `pytest tests/ -q` → expect **all tests passed, 5 skipped** (the skips are the tests that need the real corpora). If anything fails, stop.
3. **Commit the patch on a branch before running anything** (`git switch -c camera-ready`, `git add -A`, `git commit`). Every run file records
   the git revision, and `make report` warns when the runs come from mixed or dirty revisions. Do not edit code between the first and the last run.
4. Disk: the corpora themselves, plus ≈ 100 MB for results and a few hundred MB for the sampled-pool cache in `DIR/.fedids_cache`.
5. `make` needs a Unix shell. On plain Windows use WSL / Git-Bash, or the explicit commands in section 8.

## 1. Put the corpora in place

Follow [`DATA.md`](DATA.md) (file names and folder layout per corpus). Set once: `export DATA=/path/to/corpora` (PowerShell: `$env:DATA="D:\corpora"`).

## 2. Look at the data before training anything — `make data-check DATA=$DATA`

Runs `describe_datasets.py` (rows, classes, attack share, native columns, which harmonized fields were found, duplicate rate) and
`signal_check.py --strict` (does a logistic regression / boosted-tree model find *any* signal in the 31 features?). Minutes, except the first
read of CSE-CIC-IDS2018.

| You see | Meaning | Do |
|---|---|---|
| `Data for … not found` | wrong folder or file names | fix the layout (DATA.md); nothing else is wrong |
| a harmonized field "missing" that DATA.md says should exist | the column mapping in `src/fedids_bench/data/features.py` (`SOURCE_COLUMNS`) does not match your copy | **stop**; copy the first line of the CSV header (`head -1 file.csv`) and fix or send it — training on it would silently use constant features |
| attack share ≈ 0 % or ≈ 100 %, or very few classes | you pointed at a single-day file | put all day files in the folder and use `--pool all` |
| signal verdict `OK` (boosted-tree validation AUROC ≥ 0.75) | features carry signal | continue |
| `WEAK` (0.60–0.75) | learnable but hard | continue; report it as hard |
| `NO SIGNAL` (< 0.60) | the harmonized features do not separate attacks on that corpus; **no FL tuning will fix that** | stop; read the per-feature-set rows of `results/signal_check.csv` (`core5` → `core22` → `core`). If the full core is also at chance the corpus must be reported as "no learnable signal in the shared flow features" |

## 3. Rehearsal — `make reproduce-quick DATA=$DATA JOBS=4` (≈ 30–60 min)

Small settings (grids: 5 000 rows, 5 rounds, 2 seeds; tuning: 3 trials, 1 seed, 3 000 rows, 5 rounds) through every script — the `main` grid and the transfer study stand in for the five grids — writing only to `results/rehearsal/`. The report it builds
carries a "NOT PAPER-GRADE" banner on every table and figure. Check that `results/rehearsal/report/claims.md` lists most cells as above the
majority baseline. If the rehearsal is wrong, the full run will be wrong — fix it first.

## 4. Hyper-parameters — `make tune DATA=$DATA JOBS=8`

Random search with the **same budget for every algorithm** (24 trials per corpus × algorithm, trial 0 = the old defaults), scored on the
**validation split of the tuning rows**, which can never appear in a reported test set. Output: `results/tuned_hparams.json` and every trial in
`results/tuning/trials.csv`. Exit status 2 means that for some (corpus, algorithm) no trial reached a validation balanced accuracy above 0.5 + 0.02 (the majority baseline's value): read
`results/tuning/selected.csv` before going on. Continuing is allowed — the report will flag those cells — but the paper must then say so.

## 5. Grids — `make grids DATA=$DATA JOBS=8`, then `make transfer`, `make edge`

| Grid | Runs | Feeds |
|---|---:|---|
| `main` | 250 | FL-algorithm table, convergence figures |
| `partitioning` | 350 | partition ablation (IID, Dirichlet, Pareto, quantity skew, single-domain; count-weighted vs equal weights) |
| `robustness` | 900 | attacks × defenses (clean cost, accuracy drop, backdoor ASR) |
| `inflation` | 300 | sample-count spoofing, count-weighted vs equal weights |
| `sweeps` | 675 | sensitivity to model-poisoning strength, Byzantine noise, malicious fraction |
| `transfer` | 25 models | zero-shot 5 × 5 matrix, target-scaler ablation, shuffled-column control |

Every run is one JSON file in `results/runs/<grid>/`; re-running resumes where it stopped and only repeats runs whose stored settings
changed. Stop and restart freely (Ctrl-C is safe).

**Run time.** Measured on the machine where the code was prepared (2 vCPU, 2.1 GHz Xeon, one thread per run, 120 000-row stand-in
file): one grid run (10 clients, 20 rounds, 50 000 rows, 64-32 MLP) takes **6.5 s** (1 local epoch, batch 128), **21.7 s** (3 epochs,
batch 64) or **55.6 s** (5 epochs, batch 32). Which of these you get depends on what `make tune` selects, so the table brackets it.
Wall-clock assumes `JOBS=8` on 8 real cores; with 4 cores double it. Treat all of it as an order of magnitude, not a promise.

| Stage | Runs | CPU-hours (light / typical / heavy) | Wall-clock at 8 jobs |
|---|---:|---|---|
| `make tune` (federated trials; the centralized trials were not timed and are cheap) | 960 | 0.8 / 2.6 / 6.7 | 0.1 / 0.3 / 0.8 h |
| `make grids` (five grids) | 2 475 | 4.5 / 15 / 38 | 0.6 / 1.9 / 4.8 h |
| `make transfer` | 25 | ≈ 0.1 | minutes |
| first read of each corpus (cached afterwards) | — | **not measured**: the real files were never available here; expect minutes per corpus, the most for CSE-CIC-IDS2018 and IoT-23 | — |

## 6. Report — `make report`

Writes `results/report/`: LaTeX and CSV tables, **vector PDF** figures, `claims.md`, `completeness.json`. The generator *refuses* to build from a grid
with missing runs, from runs whose stored settings differ from the grid spec or the tuned file, or from rehearsal settings; it never fills a gap and prints
absent cells as `--`. Quote `claims.md` (negative findings are listed first) instead of retyping numbers.

## 7. What to send back

The report can be regenerated from the run files alone — **the corpora are not needed** — so the most useful hand-back is a branch with the results:

```bash
git add results/runs results/tuning results/tuned_hparams.json results/dataset_description.* results/signal_check.csv results/edge_profile.csv results/report
git commit -m "camera-ready runs"
git push origin HEAD:camera-ready-results
```

(tens of MB). Also keep the console output of `make data-check` and `make tune`.

## 8. The same commands without `make`

```bash
python scripts/describe_datasets.py --data-dir DATA --pool all
python scripts/signal_check.py --data-dir DATA --strict
python scripts/tune_hparams.py --data-dir DATA --n-jobs 8
python scripts/run_grid.py --spec configs/grids/main.yaml --data-dir DATA --n-jobs 8      # likewise partitioning, robustness, inflation, sweeps
python scripts/run_transfer.py --spec configs/grids/transfer.yaml --data-dir DATA --n-jobs 5
python scripts/profile_edge.py --from-run results/runs/main/main__unsw_nb15__fl__fedavg__iid__s0.json --measure-host
python scripts/make_report.py
```

## 9. Reducing the budget (honestly)

Edit the **grid YAML** (`configs/grids/*.yaml`: `base.dataset.n_samples`, `base.federated.rounds`, `seeds`) rather than passing `--set`: the settings are
then part of the spec, every run is verified against it, and the report accepts it. Command-line overrides (`--set`, `--datasets`, `--seeds`, `--limit`) are
for rehearsals only; their runs go to `results/rehearsal/` and the report refuses them. Whatever you shrink, say so in the paper (rows, rounds, seeds).
Cutting seeds below 5 widens every confidence interval; cutting rows or rounds changes absolute numbers.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `UnicodeDecodeError` | CIC files contain cp1252 bytes; the loaders read `latin-1`, so a re-encoded copy is fine, a differently-named column set is not |
| first run per corpus takes minutes | the pool cache is being built (`DIR/.fedids_cache`); later runs reuse it. Delete the folder to rebuild |
| `MemoryError` / swap thrashing | lower `--n-jobs`; each worker holds one pool (< 1 GB) |
| `results/tuned_hparams.json not found` | run `make tune` first (or `--no-tuned` for the spec's base values — rehearsal only) |
| `make report` says a grid is incomplete / has different settings | run `make grids` again (it resumes); if you changed a spec, `--force` the affected grid |
| run files exist but report complains about the revision | commit before running; do not edit code between runs |

## 11. Checks before any number goes into the paper

* Every cell of `claims.md` marked "NOT above majority" is reported as such (not hidden, not dropped).
* Confidence intervals are quoted wherever a mean is; differences are claimed only if they survive the Holm-corrected paired test.
* Equal-weight FedAvg is "neutral to count inflation" **by construction**; the finding is its clean-data cost under quantity skew (`partition_delta`).
* The column-shuffled transfer control collapses toward chance; if it does not, the transfer matrix is not measuring what the paper says.
* Edge latency/energy are analytical **ranges**; Raspberry Pi 4 and workstation parameters are assumptions unless cited (`--device-file`).
* Dataset sizes, class counts and feature counts in the paper come from `results/dataset_description.*`, not from memory.
