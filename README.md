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

- **5 Real Network & IoT Telemetry Corpora**: UNSW-NB15, CICIDS2017, CSE-CIC-IDS2018, IoT-23, and ToN_IoT.
- **Strict Zero-Leakage Preprocessing**: Fit-once normalization on training splits only, preventing statistical leakage.
- **Controlled Non-IID Partitions**: Dirichlet label priors ($\alpha \in \{0.1, 0.5, 1.0\}$), disjoint label allocations, and Pareto volume skew.
- **Decoupled FL Optimizers**: FedAvg, FedProx, and FedAdam with strict client parameter encapsulation.
- **Adversarial Threat Engine**: Label inversion, model sign-flipping/amplification, Byzantine Gaussian noise, flow watermarking (backdoor), and sample-count spoofing.
- **Byzantine-Robust Defenses**: Coordinate-wise trimmed mean, coordinate median, norm-bounded gradient clipping, and Multi-Krum.
- **Physical Edge Profiling**: Multi-tier telemetry on Raspberry Pi 4 (ARM Cortex-A72), NVIDIA Jetson Orin Nano, and workstation CPUs.
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

### 1. Execute Comprehensive Verification Harness
Runs all phases (data ingestion, zero-leakage preprocessing, non-IID partitioning, optimization, poisoning defenses, and edge profiling):

```bash
python scripts/verify_all_phases.py
```

### 2. Evaluate Baseline Optimization Algorithms (Table 3 & Fig. 3)
Evaluates FedAvg, FedProx, and FedAdam across IID and Dirichlet ($\alpha=0.5, \alpha=0.1$) distributions:

```bash
python scripts/test_optimizer_fl.py
```

### 3. Evaluate Adversarial Threat & Defense Matrix (Table 4 & Fig. 4)
Simulates label flipping, model sign inversion, Gaussian noise, backdoor trigger embedding, and sample inflation across all four Byzantine aggregators:

```bash
python scripts/evaluate_baselines.py
```

### 4. Re-generate High-Resolution Paper Figures
Generates publication-quality 300-DPI figures in `paper/figures/`:

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

## Contact & Correspondence

- **Ms. Sowmiya .S** (*Assistant Professor & Corresponding Author*): `sowmiya.s@sece.ac.in`  
- **Department of B.Tech AI&DS**, Sri Eshwar College of Engineering, Coimbatore, Tamil Nadu, India.
