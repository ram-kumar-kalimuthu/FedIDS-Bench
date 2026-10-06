# FedIDS-Bench: Architecture, Module Hierarchy, and Directory Guide

This document provides a clean, human-readable structural guide to the FedIDS-Bench framework, explaining how modules interact and how the repository is structured.

---

## 1. High-Level Architectural Flow

```
                      +---------------------------------------+
                      | Raw Telemetry Ingestion (loaders.py)  |
                      | 5 Corpora: UNSW, CICIDS, CSE, IoT, ToN|
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Split-Isolated Feature Transformation |
                      | Fit mu, sigma on S_train ONLY         |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Heterogeneity Partitioner             |
                      | Dirichlet (alpha) | Pareto (beta)     |
                      +---------------------------------------+
                                          |
                                          v
           +-----------------------------------------------------+
           | Distributed Client Workers (client.py / trainer.py) |
           | Deep-copy isolation, local SGD / Proximal training  |
           +-----------------------------------------------------+
                                          |
                                          v
           +-----------------------------------------------------+
           | Adversarial Attack Engine (attacks/*.py)            |
           | Label flip, Model poison, Byzantine noise, Backdoor |
           +-----------------------------------------------------+
                                          |
                                          v
           +-----------------------------------------------------+
           | Central Aggregation Server (server.py)              |
           | Uniform / Weighted FedAvg, FedProx, FedAdam         |
           | Byzantine Defenses (defenses/*.py: Trim, Med, Krum) |
           +-----------------------------------------------------+
                                          |
                                          v
           +-----------------------------------------------------+
           | Evaluation & Profiling (evaluation/*.py)            |
           | Detection metrics, Bandwidth, Analytical Latency/mJ |
           +-----------------------------------------------------+
```

---

## 2. Directory Map & Responsibilities

### `src/fedids_bench/` (Framework Engine)
- **`attacks/`**: Implements 5 adversarial threat models (targeted label flip, model sign-flipping/amplification, Byzantine Gaussian perturbation, flow backdoor watermarking, sample-count spoofing).
- **`data/`**: Zero-leakage data loaders for all five corpora, feature schema definitions, and non-IID partitioners (Dirichlet, Single-Domain, Pareto).
- **`defenses/`**: Byzantine-tolerant aggregation operators (Norm Clipping, Coordinate Trimmed Mean, Coordinate Median, Krum).
- **`evaluation/`**: Performance metrics (Accuracy, Balanced Accuracy, Macro/Weighted F1, MCC, FPR/FNR), communication accounting, and analytical hardware profiling.
- **`federated/`**: Client worker trainer (with state encapsulation), central aggregation coordinator, and optimization algorithms (FedAvg, FedProx, FedAdam).
- **`models/`**: Neural network architectures (`SmallMLP` with 754 parameters / 3,016 bytes, and `DeepMLP`).
- **`utils/`**: Deterministic random seed locking, structured logging, and serialization.

### `scripts/` (Harnesses & Reproducibility)
- **`run_all_paper_experiments.py`**: Executes the master benchmark suite, regenerating all results files.
- **`generate_paper_figures.py`**: Publication-quality figure generator reading directly from committed CSVs.
- **`verify_all_phases.py`**: Full multi-stage framework verification script.
- **`run_smoke.py`**: Rapid 30-second CI/CD sanity test.

### `results/` (Committed Empirical Evidence)
- Stores all precomputed benchmark CSVs (`table3_fl_algorithms.csv`, `table4_robustness.csv`, `table6_transferability.csv`, `partitioning_ablation.csv`, `device_profiling.csv`, `figure3_convergence.csv`) and raw JSON logs.

### `tests/` (Test Suite)
- 46 comprehensive unit and integration tests verifying all features, loaders, attacks, defenses, metrics, and bitwise reproducibility.
