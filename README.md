# FedIDS-Bench: A Standardized Cross-Dataset, Cross-Algorithm, and Cross-Threat-Model Benchmark for Federated Intrusion Detection Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Bitwise%20Verified-brightgreen.svg)]()

> **Official Repository for the Paper**:  
> *"FedIDS-Bench: A Standardized Cross-Dataset, Cross-Algorithm, and Cross-Threat-Model Benchmark for Federated Intrusion Detection Systems"*  
> Authors: Ms. Sowmiya .S, Ram kumar .K, Rahgul .S.A, Naveen Kumar .R, Ranjan Kumar .S.M  
> Affiliation: Department of B.Tech AI&DS, Sri Eshwar College of Engineering, Tamil Nadu, India

---

## Overview

**FedIDS-Bench** is an end-to-end, standardized benchmarking platform designed to eliminate methodological discrepancies and reproducibility failures in Federated Learning for Network Intrusion Detection Systems (FL-NIDS).

Published research in FL-NIDS frequently suffers from data leakage (computing normalization moments over entire datasets prior to train-test splitting), synthetic heterogeneity that fails to capture real telemetry dynamics, single-threat evaluations without defensive baselines, and a lack of physical edge benchmarking. **FedIDS-Bench** systematically resolves these issues across:

- **5 Real Network & IoT Telemetry Corpora**: UNSW-NB15, CICIDS2017, CSE-CIC-IDS2018, IoT-23, and ToN_IoT (standardized to a canonical 5-feature NetFlow core).
- **Strict Zero-Leakage Preprocessing**: Fit-once normalization on training splits only, preventing statistical leakage.
- **Controlled Non-IID Partitions**: Dirichlet label priors ($\alpha \in \{0.1, 0.5, \infty\}$), disjoint label allocations, and Pareto volume skew ($\beta=1.5$).
- **Decoupled FL Optimizers**: FedAvg, FedProx, and FedAdam with strict client parameter encapsulation.
- **Adversarial Threat Engine**: Label inversion, model sign-flipping/amplification, Byzantine Gaussian noise, flow watermarking (backdoor), and sample-count spoofing.
- **Byzantine-Robust Defenses**: Coordinate-wise trimmed mean, coordinate median, norm-bounded gradient clipping, Krum, and uniform-weight FedAvg.
- **Analytical Edge Hardware Profiling**: Model inference latency, bandwidth, and energy consumption analytically derived from peak FLOPs and manufacturer TDP across Raspberry Pi 4 (ARM Cortex-A72), NVIDIA Jetson Orin Nano, and workstation CPUs.
- **Bitwise Multi-Process Determinism**: Hardened random seeding, determinism locks, and multi-process state isolation.

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                            Stage 1: Multi-Corpora Ingestion                        |
|   [UNSW-NB15 (39f)] [CICIDS2017 (78f)] [CSE-CIC-2018 (78f)] [IoT-23 (11f)] [ToN] |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|               Stage 2: Strict Zero-Leakage Train-Test Isolation                   |
|           Fit: mu_train, sigma_train on S_train ONLY  ==> Transform S_test        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   Stage 3: Multi-Domain Non-IID Partitioner                       |
|   Dirichlet Class Skew Dir(alpha) | Single-Domain Bounds | Pareto Volume Imbalance |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   Stage 4: Edge Workers & Adversarial Threat Engine               |
|   Clients 1..K (FedAvg / FedProx / FedAdam) | Model Inversion | Byzantine Noise   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   Stage 5: Central Coordinator & Robust Aggregators               |
|      Norm Clipping | Trimmed Mean | Coordinate Median | Multi-Krum                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|              Stage 6: Multi-Tier Edge Profiler & Reproducibility Engine           |
|      F1/Precision/Recall | Network Uplink/Downlink | Latency (ms) | Energy (mJ)   |
+-----------------------------------------------------------------------------------+
```

---

## Quick Start

### 1. Installation

Clone the repository and install requirements in an editable virtual environment:

```bash
git clone https://github.com/ram-kumar-kalimuthu/FedIDS-Bench.git
cd FedIDS-Bench

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies and CLI
pip install -e .[dev]
```

### 2. Fast Smoke Test (Synthetic Telemetry)

Verify all components in less than 30 seconds:

```bash
python scripts/run_smoke.py
```

Or execute via the `fedids-bench` CLI:

```bash
fedids-bench run --config configs/smoke.yaml
```

---

## Reproducing Paper Results
 
To reproduce all experimental figures, tables, and findings reported in the manuscript:

### 1. Execute End-to-End Paper Evaluation Suite
Runs the complete multi-dataset experimental matrix across all five corpora, the adversarial defense grid, Pareto volume ablation, transferability matrix, and device profiling, outputting committed CSVs to `results/`:

```bash
python scripts/run_all_paper_experiments.py
```

Generated outputs include:
- `results/table3_fl_algorithms.csv` (Table 3: Multi-Dataset FL Benchmarks across all 5 corpora)
- `results/table4_robustness.csv` (Table 4: Adversarial Defense Benchmarks with Uniform FedAvg)
- `results/partitioning_ablation.csv` (Pareto $\beta=1.5$ & Label Skew Ablations)
- `results/table6_transferability.csv` (Table 6: Cross-Dataset Transfer Matrix under aligned 5-feature core)
- `results/figure3_convergence.csv` (Figure 3: Round-by-Round Convergence Trajectories)
- `results/device_profiling.csv` (Table 5 & Figure 5: Edge Latency and Energy Analytical Estimates)

### 2. Generate Publication-Quality Figures (Figs 2, 3, 4, 5)
Directly renders high-resolution 300-DPI paper figures in `paper/figures/` from the committed result CSV files (fails fast if any CSV is missing):

```bash
python scripts/generate_paper_figures.py
```

---

## Repository Structure

```
FedIDS-Bench/
├── configs/                  # Benchmark YAML manifests (smoke, standard, large)
├── docs/                     # Detailed architectural specifications & audit logs
├── notebooks/                # Google Colab / Jupyter interactive notebooks
├── results/                  # Pre-computed benchmark outputs, CSVs, and JSON logs
├── scripts/                  # Reproducibility runners and evaluation harnesses
├── src/                      # Modular Python package source
│   └── fedids_bench/
│       ├── attacks/          # Adversarial attack implementations
│       ├── data/             # Split-isolated loaders & non-IID partitioners
│       ├── defenses/         # Byzantine-robust aggregators
│       ├── evaluation/       # Performance, bandwidth & edge profilers
│       ├── federated/        # Client/Server coordinators (FedAvg, FedProx, FedAdam)
│       └── models/           # Neural network architectures (SmallMLP, DeepMLP)
├── tests/                    # Unit & integration test suite (pytest)
├── LICENSE                   # MIT License
├── Makefile                  # Automation shortcuts
├── pyproject.toml            # Python packaging manifest
└── requirements.txt          # Python dependency specifications
```

---

## Verified Bitwise Determinism

To verify that model weights, partitioning permutations, and gradient updates match across separate hardware runs at the bitwise level:

```bash
pytest tests/test_reproducibility.py -v
```

---



## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---


