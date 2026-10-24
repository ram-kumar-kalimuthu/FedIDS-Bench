# FedIDS-Bench: A Standardized Cross-Dataset, Cross-Algorithm, and Cross-Threat-Model Benchmark for Federated Intrusion Detection Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official Repository for the Paper**:  
> *"FedIDS-Bench: A Standardized Cross-Dataset, Cross-Algorithm, and Cross-Threat-Model Benchmark for Federated Intrusion Detection Systems"*  
> Authors: Ms. Sowmiya .S, Ram kumar .K, Rahgul .S.A, Naveen Kumar .R, Ranjan Kumar .S.M  
> Affiliation: Department of B.Tech AI&DS, Sri Eshwar College of Engineering, Tamil Nadu, India

---

## Overview

**FedIDS-Bench** is an end-to-end, standardized benchmarking platform for Federated Learning for Network Intrusion Detection Systems (FL-NIDS). It targets the methodological problems that make published FL-NIDS numbers hard to compare: normalization statistics computed before the train/test split, synthetic heterogeneity, single-threat evaluation without defensive baselines, hyper-parameters never tuned or tuned on the test set, and edge-cost figures that are typed in rather than computed.

- **5 real corpora**: UNSW-NB15, CICIDS2017, CSE-CIC-IDS2018, IoT-23 and ToN_IoT, mapped onto one harmonized **31-feature flow core** (log-scaled volumes, packet sizes, byte/packet asymmetry and rates, one-hot protocol, service group, destination-port range and connection outcome). Fields a corpus does not ship become an explicit `unknown` category, never an invented number. The old 5-volume-feature core (`core5`) is kept only as an ablation.
- **Leakage control**: de-duplication before splitting, stratified train/validation/test splits, `StandardScaler` fitted on the training rows only, and a *content-hashed* `tune` / `report` row split so that the rows used to choose hyper-parameters can never appear in a reported test set.
- **Partitions**: IID, Dirichlet label skew (alpha 0.1 / 0.5), Pareto quantity skew (beta 1.5), plain quantity skew and a *single-domain* partitioner (each client holds benign traffic plus at most `L-1` attack categories). `label_skew` is a stratified IID split and is documented as such.
- **FL algorithms**: FedAvg, FedProx and FedAdam (plus a centralized reference), with hyper-parameters selected per (corpus, algorithm) on the validation split by one shared random-search protocol.
- **Threat engine**: label flipping, model poisoning, Byzantine noise, backdoor (with attack-success-rate), and **sample-count spoofing** (a client reports an inflated `n_k`).
- **Defenses**: coordinate-wise trimmed mean, coordinate median, norm clipping, Krum, and equal-weight (uniform) FedAvg.
- **Metrics**: balanced accuracy, MCC, F1 on the attack class, a real AUROC computed from model scores, and the majority-class baseline next to every result. Statistics are mean +/- 95 % Student-t confidence interval over 5 seeds, with paired comparisons and an explicit "beats the majority baseline" test.
- **Cross-dataset transfer**: strict zero-shot evaluation (source-corpus scaler), the target-scaler convention as an ablation, and a column-shuffled control that must collapse if the model uses feature semantics.
- **Edge cost model**: parameters, FLOPs, latency, energy and per-round traffic are *computed* from the trained model and reported as ranges (`eta` in [0.01, 0.10]); a wall-clock / RAPL measurement on the local machine can be added. Raspberry Pi 4 and workstation figures are assumptions unless you supply cited values (`--device-file`).
- **Determinism**: all RNGs are seeded per run. Results are repeatable for the same seed, library versions and thread configuration; bitwise identity across different hardware is *not* claimed.

> **Status of results.** Every number, table and figure of the camera-ready paper is produced by `make reproduce` from the run files in `results/runs/` (`make report` refuses incomplete grids and never fills a gap). The CSV/JSON files at the top level of `results/` are from the first submission, were produced by the older pipeline and are **superseded — do not cite them** (see `docs/audit_findings.md`).

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                      Stage 1: Multi-Corpora Ingestion + Harmonization              |
|  [UNSW-NB15] [CICIDS2017] [CSE-CIC-2018] [IoT-23] [ToN_IoT]  -->  canonical flow   |
|  fields (duration, bytes/packets fwd+bwd, protocol, dst port, service, outcome)    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|        Stage 2: De-duplication, Stratified Splits, Zero-Leakage Scaling            |
|   tune/report row roles | Fit: mu_train, sigma_train on S_train ONLY => transform  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   Stage 3: Non-IID Partitioner                                     |
|   IID | Dirichlet(alpha) | Pareto / quantity skew | Single-domain                   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   Stage 4: Clients & Adversarial Threat Engine                     |
|   FedAvg / FedProx / FedAdam | label flip | poison | Byzantine | backdoor | n_k lie |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   Stage 5: Coordinator & Robust Aggregators                        |
|        Norm clipping | Trimmed mean | Coordinate median | Krum | Uniform weights  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|        Stage 6: Evaluation, Statistics, Edge Cost Model, Report Generator          |
|   bal. acc | MCC | F1 | AUROC | majority baseline | mean +/- 95 % CI | edge table  |
+-----------------------------------------------------------------------------------+
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/ram-kumar-kalimuthu/FedIDS-Bench.git
cd FedIDS-Bench

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

pip install -e .[dev]
```

### 2. Fast smoke test (synthetic telemetry, < 30 s)

```bash
python scripts/run_smoke.py
# or
fedids-bench run --config configs/smoke.yaml
```

### 3. Unit tests

```bash
pytest tests/ -q          # 5 tests are skipped unless the real corpora are available (FEDIDS_DATA_DIR=...)
```

---

## Reproducing the paper

The corpora are **not** shipped (size and licenses); see [`docs/DATA.md`](docs/DATA.md) for the files each loader expects and
[`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the step-by-step runbook, expected run times and troubleshooting.

```bash
# 0. 30-60 minute end-to-end check on small settings; writes only to results/rehearsal/ and stamps its report NOT PAPER-GRADE
make reproduce-quick DATA=/path/to/corpora JOBS=4

# 1. the real thing (resumable; hours, see docs/REPRODUCE.md)
make reproduce DATA=/path/to/corpora JOBS=8
#    = data-check -> tune -> grids -> transfer -> edge -> report
```

| Step | Command | Output |
|---|---|---|
| Inspect the data | `make data-check` | `results/dataset_description.{json,csv}`, `results/signal_check.csv` |
| Select hyper-parameters (validation split, tuning rows only) | `make tune` | `results/tuned_hparams.json`, `results/tuning/{trials,selected}.csv` |
| Five run grids | `make grids` | `results/runs/{main,partitioning,robustness,inflation,sweeps}/*.json` (2,475 runs) |
| Zero-shot transfer | `make transfer` | `results/runs/transfer/*.json` |
| Edge cost | `make edge` | `results/edge_profile.csv` |
| Tables, figures, claims | `make report` | `results/report/*.tex`, `*.csv`, `*.pdf` (vector), `claims.md` |

`results/report/claims.md` is written by the report generator: for every (corpus, method, partition) cell it states whether
balanced accuracy is above the majority baseline with 95 % confidence, lists the paired differences that survive Holm correction,
and summarizes the equal-weight, inflation and transfer findings — negative findings first. Write the paper text from that file,
not the other way round.

---

## Repository Structure

```
FedIDS-Bench/
├── configs/                  # smoke / standard / large manifests and grids/ (the five paper grids + transfer)
├── docs/                     # DATA.md (corpora), REPRODUCE.md (runbook), audit_findings.md, codebase catalog
├── notebooks/                # interactive notebooks
├── results/                  # runs/, tuning/, report/ (generated); top-level legacy files are superseded
├── scripts/                  # describe_datasets, signal_check, tune_hparams, run_grid, run_transfer, profile_edge, make_report
├── src/fedids_bench/
│   ├── attacks/              # label flip, poison, Byzantine, backdoor, sample-count spoofing
│   ├── data/                 # features.py (harmonized core), loaders, partitioners
│   ├── defenses/             # Byzantine-robust aggregators
│   ├── evaluation/           # metrics, cross-dataset transfer, edge cost model, backdoor success
│   ├── experiments/          # runner, grid executor, tuning protocol, transfer study, statistics
│   ├── federated/            # clients and server (FedAvg, FedProx, FedAdam)
│   ├── models/               # SmallMLP, DeepMLP
│   └── reporting/            # tables, vector figures, claim checks
├── tests/                    # pytest suite
├── LICENSE
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Determinism check

```bash
pytest tests/test_reproducibility.py -v
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
