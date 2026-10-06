# FedIDS-Bench: Comprehensive File-by-File Codebase Catalog

**Date:** October 2026  
**Auditor:** Lead Systems Architect & Peer Reviewer  
**Repository:** [FedIDS-Bench](https://github.com/ram-kumar-kalimuthu/FedIDS-Bench)  
**Scope:** Complete structural, functional, and organizational audit of every file and directory.

---

## 1. Directory Structure Overview

```
fedids-bench/
├── configs/                  # Benchmark YAML configuration manifests
├── docs/                     # Specifications, audit findings, and codebase analysis
│   └── codebase_analysis/    # File-by-file catalog, architecture map, and cleanup records
├── paper/                    # Paper figures and camera-ready rebuttal matrix
│   └── figures/              # High-resolution 300-DPI publication figures
├── results/                  # Committed empirical benchmark CSVs and run JSONs
├── scripts/                  # Master execution harnesses and reproducibility runners
├── src/                      # Production Python package source
│   └── fedids_bench/         # Core library modules (attacks, data, defenses, eval, fl, models)
├── tests/                    # Comprehensive unit and integration test suite (pytest)
├── LICENSE                   # MIT License
├── Makefile                  # Build and test automation
├── pyproject.toml            # Modern Python packaging specification (PEP 517/518)
├── README.md                 # Project landing page and reproduction guide
└── requirements.txt          # Frozen dependency specifications
```

---

## 2. Exhaustive File-by-File Catalog

### 2.1. Root Configuration & Packaging Files

| File Path | Purpose & Responsibility | Key Entities / Definitions | Status |
|---|---|---|:---:|
| `pyproject.toml` | Standardized Python build system specification (PEP 517/518) using `setuptools`. Declares dependencies, CLI entrypoints, and test configurations. | `[build-system]`, `[project]`, `project.scripts: fedids-bench = "fedids_bench.cli:main"` | Active |
| `requirements.txt` | Minimal pinned dependencies for virtual environment setup (`torch`, `numpy`, `scikit-learn`, `pandas`, `pyyaml`, `matplotlib`). | Dependency list | Active |
| `Makefile` | Developer workflow shortcuts for installing, testing, linting, and running smoke tests. | `install`, `test`, `smoke`, `clean` | Active |
| `LICENSE` | Standard MIT Open Source License granting unrestricted academic and commercial reuse. | Copyright notice | Active |
| `README.md` | Primary documentation containing system architecture diagrams, quickstart commands, and instructions for reproducing paper tables/figures. | Architecture schematic, reproduction commands, author affiliations | Active |
| `.gitignore` | Prevents heavy raw dataset CSVs, temporary Python caches, virtual environments, and intermediate files from polluting version control while whitelisting `results/*.csv` and `paper/figures/*`. | Exclusion rules and whitelist directives | Active |

---

### 2.2. Core Package (`src/fedids_bench/`)

#### 2.2.1. Root Package & CLI
| File Path | Purpose & Responsibility | Key Entities / Definitions | Status |
|---|---|---|:---:|
| `src/fedids_bench/__init__.py` | Package initialization; exposes semantic versioning. | `__version__ = "0.1.0"` | Active |
| `src/fedids_bench/cli.py` | Command-line interface allowing users to launch benchmarks directly via `fedids-bench run --config <path>`. | `main()`, `run_command()`, argument parser | Active |
| `src/fedids_bench/config.py` | Strongly typed Pydantic configuration schemas validating experiment settings, algorithm parameters, non-IID distributions, threat models, and defense operators. | `ExperimentConfig`, `FederatedConfig`, `PartitioningConfig`, `AttackConfig`, `DefenseConfig`, `ModelConfig` | Active |

#### 2.2.2. Adversarial Threat Engine (`src/fedids_bench/attacks/`)
| File Path | Purpose & Responsibility | Key Entities / Definitions | Status |
|---|---|---|:---:|
| `src/fedids_bench/attacks/__init__.py` | Exposes attack primitives for modular imports. | Package exports | Active |
| `src/fedids_bench/attacks/base.py` | Abstract base class defining interface contracts for data poisoning and model parameter poisoning. | `BaseAttack`, `apply_data_poisoning()`, `apply_model_poisoning()` | Active |
| `src/fedids_bench/attacks/label_flip.py` | Targeted and random label inversion attack; swaps attack and benign binary targets ($y \gets 1 - y$). | `LabelFlipAttack` | Active |
| `src/fedids_bench/attacks/feature_poison.py` | Injects localized attribute noise into network flow features. | `FeaturePoisonAttack` | Active |
| `src/fedids_bench/attacks/model_poison.py` | Inverts gradient trajectory and scales update magnitude ($\Delta w_m = -\gamma \Delta w$) to diverge global model. | `ModelPoisonAttack` | Active |
| `src/fedids_bench/attacks/byzantine.py` | Replaces client model delta with isotropic Gaussian noise rescaled to match honest $L_2$-norm ($\sigma \|\Delta w\|_2 \frac{\xi}{\|\xi\|_2}$). | `ByzantineNoiseAttack` | Active |
| `src/fedids_bench/attacks/backdoor.py` | Injects deterministic watermark trigger pattern into flow features ($x'[0] = 999.0$) paired with benign target label $y=0$. | `BackdoorFlowAttack` | Active |
| `src/fedids_bench/attacks/factory.py` | Dynamic factory for instantiating attacks based on YAML configuration manifests. | `AttackFactory.create_attack()` | Active |

#### 2.2.3. Data Ingestion & Partitioning (`src/fedids_bench/data/`)
| File Path | Purpose & Responsibility | Key Entities / Definitions | Status |
|---|---|---|:---:|
| `src/fedids_bench/data/__init__.py` | Exposes loaders and partitioners. | Package exports | Active |
| `src/fedids_bench/data/schema.py` | Standard NetFlow feature definitions, column indices, and canonical 5-feature schema (`duration`, `fwd_bytes`, `bwd_bytes`, `fwd_pkts`, `bwd_pkts`). | `CANONICAL_FEATURES`, `DatasetSchema` | Active |
| `src/fedids_bench/data/loaders.py` | Split-isolated dataset loaders for UNSW-NB15, CICIDS2017, CSE-CIC-IDS2018, IoT-23 (parsing `orig_pkts` and `resp_pkts`), and ToN-IoT. Enforces fit-once training normalization to eliminate train-test leakage. | `load_dataset()`, `FEATURE_MAPPING`, `StandardScalerWrapper` | Active |
| `src/fedids_bench/data/partition.py` | Multi-faceted heterogeneity partitioners: IID, Dirichlet label prior ($\alpha \in \{0.1, 0.5, \infty\}$), Single-Domain disjoint class bounds, and Pareto power-law quantity skew ($\beta = 1.5$). | `IIDPartitioner`, `DirichletPartitioner`, `SingleDomainPartitioner`, `ParetoPartitioner` | Active |
| `src/fedids_bench/data/synthetic.py` | High-speed, deterministic synthetic network flow generator for fast unit tests and smoke validation without requiring multi-GB raw pcap traces. | `generate_synthetic_telemetry()` | Active |

#### 2.2.4. Byzantine Defenses (`src/fedids_bench/defenses/`)
| File Path | Purpose & Responsibility | Key Entities / Definitions | Status |
|---|---|---|:---:|
| `src/fedids_bench/defenses/__init__.py` | Exposes robust aggregation defense operators. | Package exports | Active |
| `src/fedids_bench/defenses/base.py` | Abstract interface contract for server-side robust aggregators. | `BaseDefense`, `aggregate()` | Active |
| `src/fedids_bench/defenses/clipping.py` | Norm-bounded gradient clipping projecting parameter updates onto an $L_2$ ball of radius $\tau$. | `NormClippingDefense` | Active |
| `src/fedids_bench/defenses/trimmed_mean.py` | Coordinate-wise trimmed mean aggregator discarding $\beta$-tails along each weight dimension before averaging. | `TrimmedMeanDefense` | Active |
| `src/fedids_bench/defenses/median.py` | Coordinate-wise median aggregator selecting middle value per parameter coordinate across workers. | `CoordinateMedianDefense` | Active |
| `src/fedids_bench/defenses/krum.py` | Multi-Krum Byzantine aggregation selecting candidate vectors that minimize cumulative Euclidean distance to nearest neighbors. | `KrumDefense` | Active |
| `src/fedids_bench/defenses/factory.py` | Factory instantiating defense operators based on configuration. | `DefenseFactory.create_defense()` | Active |

#### 2.2.5. Evaluation & Edge Profiling (`src/fedids_bench/evaluation/`)
| File Path | Purpose & Responsibility | Key Entities / Definitions | Status |
|---|---|---|:---:|
| `src/fedids_bench/evaluation/__init__.py` | Exposes evaluation profilers and metric calculators. | Package exports | Active |
| `src/fedids_bench/evaluation/classification.py` | Robust intrusion metrics: Accuracy, Balanced Accuracy, Precision, Recall, Macro-F1, Weighted-F1, MCC, FPR, and FNR with zero-division protection. | `compute_classification_metrics()`, `derive_binary_rates()` | Active |
| `src/fedids_bench/evaluation/communication.py` | Precise bandwidth accountant tracking uplink, downlink, and round-trip byte exchanges per client round. | `CommunicationTracker` | Active |
| `src/fedids_bench/evaluation/cross_dataset.py` | Evaluates zero-shot cross-corpus model generalizability under aligned feature schemas, reporting AUROC and Balanced Accuracy. | `evaluate_cross_domain_transfer()` | Active |
| `src/fedids_bench/evaluation/device_profiles.py` | Hardware specifications (cores, frequency, peak GFLOP/s, manufacturer TDP) for Raspberry Pi 4, Jetson Orin Nano, and Laptop CPU. | `DEVICE_PROFILES`, `DeviceProfile` | Active |
| `src/fedids_bench/evaluation/resource_cost.py` | Analytical hardware cost estimator calculating per-sample inference latency and per-batch energy draw based on theoretical FLOPs and TDP. | `estimate_execution_cost()`, `calculate_model_flops()` | Active |

#### 2.2.6. Federated Orchestration & Optimizers (`src/fedids_bench/federated/`)
| File Path | Purpose & Responsibility | Key Entities / Definitions | Status |
|---|---|---|:---:|
| `src/fedids_bench/federated/__init__.py` | Exposes federated client, server, and trainer abstractions. | Package exports | Active |
| `src/fedids_bench/federated/client.py` | Decentralized edge worker simulating local SGD training with deep-copy parameter state encapsulation to prevent memory leaks across rounds. | `FederatedClient` | Active |
| `src/fedids_bench/federated/server.py` | Central aggregation server orchestrating client sampling, update collection, attack injection, defense filtering, and `trust_client_sample_counts` control. | `FederatedServer` | Active |
| `src/fedids_bench/federated/trainer.py` | Local mini-batch training loop executing forward, loss calculation, backward propagation, and proximal regularization. | `LocalTrainer` | Active |
| `src/fedids_bench/federated/algorithms/fedavg.py` | Classical Federated Averaging with options for sample-weighted aggregation and uniform-weighted aggregation. | `FedAvgAggregator` | Active |
| `src/fedids_bench/federated/algorithms/fedprox.py` | Federated Proximal algorithm adding $\frac{\mu}{2}\|w - w_t\|^2$ regularizer to penalize client drift. | `FedProxClientTrainer` | Active |
| `src/fedids_bench/federated/algorithms/fedadam.py` | Adaptive server-side optimization computing first and second momentum vectors ($\beta_1, \beta_2, \tau$) to stabilize updates under acute skew. | `FedAdamAggregator` | Active |

#### 2.2.7. Neural Network Models (`src/fedids_bench/models/`)
| File Path | Purpose & Responsibility | Key Entities / Definitions | Status |
|---|---|---|:---:|
| `src/fedids_bench/models/__init__.py` | Exposes model architectures. | Package exports | Active |
| `src/fedids_bench/models/base.py` | Abstract model interface defining weight extraction, loading, parameter counting, and FLOP calculation. | `BaseModel` | Active |
| `src/fedids_bench/models/mlp.py` | Multi-Layer Perceptrons: `SmallMLP` ($d \to 32 \to 16 \to 2$, exactly 754 float32 parameters / 3,016 bytes) and `DeepMLP` ($d \to 128 \to 64 \to 32 \to 2$). | `SmallMLP`, `DeepMLP` | Active |

#### 2.2.8. Utilities & Reporting (`src/fedids_bench/utils/` & `reporting/`)
| File Path | Purpose & Responsibility | Key Entities / Definitions | Status |
|---|---|---|:---:|
| `src/fedids_bench/utils/logging.py` | Standardized colored logging with timestamps and module namespaces. | `get_logger()`, `setup_logging()` | Active |
| `src/fedids_bench/utils/reproducibility.py` | Determinism engine locking seeds across PyTorch CPU, NumPy, and Python runtime. | `set_seed()`, `enable_deterministic_mode()` | Active |
| `src/fedids_bench/utils/serialization.py` | JSON encoder/decoder handling NumPy arrays, tensors, and dataclasses. | `to_json()`, `from_json()` | Active |
| `src/fedids_bench/reporting/tables.py` | Automated formatter outputting LaTeX `booktabs` and Markdown comparison tables. | `generate_latex_table()`, `generate_markdown_table()` | Active |

---

### 2.3. Test Suite (`tests/`)

Every test in this suite is verified passing (46/46 passed in 33s):

| Test File | Verification Scope | Status |
|---|---|:---:|
| `tests/conftest.py` | Pytest fixtures providing synthetic data, temporary directories, and model instances. | Passing |
| `tests/test_adversarial_fl.py` | Validates end-to-end FL training under model poisoning and label flip attacks. | Passing |
| `tests/test_attacks.py` | Unit tests for all attack primitives: label flip, feature poison, model poison, Byzantine noise, backdoor trigger. | Passing |
| `tests/test_communication.py` | Validates uplink and downlink byte tracking per client round. | Passing |
| `tests/test_config.py` | Tests Pydantic config validation, boundary checks, and invalid parameter rejection. | Passing |
| `tests/test_cross_dataset.py` | Validates canonical feature alignment and cross-dataset evaluation harness. | Passing |
| `tests/test_defenses.py` | Unit tests for robust aggregators: clipping, trimmed mean, median, and Krum. | Passing |
| `tests/test_device_profiles.py` | Validates analytical latency and energy estimation formulas against known bounds. | Passing |
| `tests/test_fedavg.py` | Tests weighted vs. uniform FedAvg mathematics and single-client identity. | Passing |
| `tests/test_loaders.py` | Validates all 5 dataset loaders; asserts $\ge 5$ features and zero null values. | Passing |
| `tests/test_metrics.py` | Tests precision, recall, F1, MCC, and FPR/FNR under balanced and imbalanced predictions. | Passing |
| `tests/test_models.py` | Validates forward pass dimensions, parameter counting, and deterministic initialization. | Passing |
| `tests/test_partition.py` | Validates IID, Dirichlet label skew, and held-out test splits. | Passing |
| `tests/test_phase2.py` | Tests split-isolated scaler non-leakage, sample-count defense, Pareto quantity skew, FedProx, and FedAdam execution. | Passing |
| `tests/test_real_datasets.py` | Validates real telemetry ingestion on actual CSV files for all 5 corpora. | Passing |
| `tests/test_reporting.py` | Tests LaTeX table formatting and output generation. | Passing |
| `tests/test_reproducibility.py` | Subprocess-level end-to-end bitwise reproducibility verification across independent processes. | Passing |
| `tests/test_synthetic.py` | Tests synthetic telemetry generation and deterministic seeding. | Passing |

---

### 2.4. Production Scripts (`scripts/`)

| Script Path | Purpose & Responsibility | Output / Artifact Produced | Status |
|---|---|---|:---:|
| `scripts/run_all_paper_experiments.py` | Master evaluation suite executing all 5 benchmarks across 5 corpora, Pareto quantity skew, adversarial defenses, and edge profiling. | Generates all committed CSVs in `results/` | Active |
| `scripts/generate_paper_figures.py` | Publication figure generator strictly reading from committed CSVs, rendering Figures 2, 3, 4, and 5 at 300 DPI. | `paper/figures/*.png` | Active |
| `scripts/verify_all_phases.py` | Comprehensive multi-stage verification script validating ingestion, normalization, FL, attacks, and defenses. | Verification summary | Active |
| `scripts/run_smoke.py` | Ultra-fast 30-second synthetic smoke test for quick continuous integration verification. | Terminal report | Active |
| `scripts/test_optimizer_fl.py` | Benchmarks FedAvg, FedProx, and FedAdam across learning rates and Dirichlet tiers. | `results/fl_*.json` | Active |
| `scripts/evaluate_baselines.py` | Evaluates dummy baselines (majority class, stratified random) and centralized SGD baselines. | `results/baseline_metrics.json` | Active |
| `scripts/prepare_data.py` | Data validation and integrity checking utility for raw dataset directories. | Preprocessing logs | Active |
| `scripts/generate_synthetic_data.py` | Standalone CLI script for generating synthetic traffic files. | Synthetic CSVs | Active |
| `scripts/aggregate_results.py` | Aggregates individual JSON run logs into summary statistics. | Consolidated metrics | Active |
| `scripts/diagnose_learning.py` | Diagnostic script for inspecting round-by-round training convergence curves. | Diagnostic logs | Active |

---

### 2.5. Committed Benchmark Results (`results/`)

All figures and tables in the paper are dynamically backed by these committed files:

| File Name | Description & Contents | Backed Manuscript Element |
|---|---|---|
| `table3_fl_algorithms.csv` | Full cross-dataset benchmark across all 5 corpora $\times$ {FedAvg, FedProx, FedAdam} $\times$ {IID, $\alpha=0.5, 0.1$} with Majority Class baselines. | Table 3 in Section 6.1 |
| `table4_robustness.csv` | Adversarial poisoning grid (Label Flip, Model Poisoning, Byzantine Noise, Backdoor, Sample Spoofing) across 5 Byzantine defenses and Uniform FedAvg. | Table 4 in Section 6.2 |
| `table6_transferability.csv` | Cross-domain transfer matrix across all 5 datasets under the canonical 5-feature NetFlow core, reporting Accuracy, Balanced Accuracy, AUROC, and F1. | Table 6 in Section 6.4 |
| `partitioning_ablation.csv` | Empirical ablation of Pareto power-law quantity skew ($\beta=1.5$, $N_{\min}=113, N_{\max}=363$ samples) comparing weighted vs uniform FedAvg. | Section 3.2, Section 6.2 |
| `device_profiling.csv` | Analytical hardware execution metrics (latency, batch energy, model size, round traffic) across Raspberry Pi 4, Jetson Orin Nano, and Laptop CPU. | Table 5 & Fig. 5 in Section 6.3 |
| `figure3_convergence.csv` | Round-by-round Global Test F1 convergence trajectory over 20 communication rounds for FedAvg, FedProx, and FedAdam. | Figure 3 in Section 6.1 |
| `dataset_summary.csv` | Native feature/class counts vs harmonized 5-feature NetFlow core dimensions across the 5 corpora. | Table 2 & Fig. 2 in Section 5.1 |
| `fl_*.json` (25 files) | Individual JSON execution logs recording complete round-by-round client metrics for every dataset and Dirichlet tier. | Raw empirical audit logs |
| `adv_*.json` (30 files) | Individual JSON execution logs recording complete round-by-round metrics for every threat model $\times$ defense operator. | Raw empirical audit logs |

---

## 3. Summary of Cleanup Actions Performed

During our audit, we identified and removed the following redundant and obsolete files:

1. **`scripts/generate_figures.py` (REMOVED)**: An outdated figure generation script that previously contained hardcoded dummy fallback curves (which reviewer R1-6 correctly criticized). This has been completely superseded by `scripts/generate_paper_figures.py`, which strictly reads from committed CSVs.
2. **`scripts/time_one_run.py` (REMOVED)**: A temporary scratch timing script from development.
3. **`scripts/fedids-bench/` and `scripts/paper/` (REMOVED)**: Accidental nested directories created during early shell execution.
4. **`results/default/` (REMOVED)**: Stray scratch output folder from preliminary smoke runs.

All obsolete files have been purged, leaving the repository in a clean, human-readable, and well-organized state.
