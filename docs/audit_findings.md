# FedIDS-Bench — Methodological Audit & Resolution Report

**Auditor Profile**: Lead FL/IDS Peer Reviewer & Methodological Auditor  
**Scope**: FedIDS-Bench Codebase & Experimental Harness  
**Status**: **VERIFIED & RESOLVED — All Methodological Fixes Implemented, Validated, and Merged**

---

## Executive Summary of Audit Resolutions

All critical, high, and medium severity findings identified during preliminary auditing have been systematically resolved, empirically verified, and integrated into the test suite.

| Finding ID | Severity | Description | Resolution Status |
|---|:---:|---|:---:|
| **CRITICAL-1** | Critical | Train-Test Data Leakage in Global Feature Scaling | **RESOLVED**: Split-isolated scaling fitted strictly on training partition (`split == 0`). |
| **CRITICAL-2** | Critical | Unvalidated Client Sample Count ($n_k$) Inflation | **RESOLVED**: Added `trust_client_sample_counts=False` (Uniform FedAvg) & verified defense. |
| **HIGH-1** | High | Division-by-Zero Metric Distortions | **RESOLVED**: Replaced with robust classification metrics (`zero_division=0`, Macro/Weighted F1, MCC). |
| **MEDIUM-1** | Medium | Extreme Dirichlet Zero-Sample Partitions | **RESOLVED**: Proportional stratified allocation with fallback guarantees minimum sample floor. |
| **MEDIUM-2** | Medium | Heldout Client Count Floor Mismatch | **RESOLVED**: Added explicit ceiling and floor checks in `_calculate_heldout_count`. |
| **MEDIUM-3** | Medium | Feature Alignment across Heterogeneous Corpora | **RESOLVED**: Implemented unified 5-feature NetFlow core (Sarhan et al.) across all 5 loaders. |
| **LOW-1** | Low | Deterministic Multiprocessing Execution | **RESOLVED**: Verified bitwise reproducibility across isolated seed controls. |

---

## Detailed Resolutions & Empirical Verifications

### 1. [CRITICAL-1] Split-Isolated Feature Standardization
* **Affected Components**: `src/fedids_bench/data/loaders.py`, `src/fedids_bench/data/synthetic.py`
* **Resolution**: Feature normalization parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are computed exclusively from the local training split (`split == 0`) before partitioning across clients. Test features and held-out evaluation splits are transformed using these stored parameters without re-estimation.
* **Empirical Validation**: Validated in `tests/test_real_datasets.py` across UNSW-NB15, CICIDS2017, CSE-CIC-IDS2018, IoT-23, and ToN-IoT.

---

### 2. [CRITICAL-2] Client-Reported Sample Count Verification
* **Affected Components**: `src/fedids_bench/federated/algorithms/fedavg.py`, `src/fedids_bench/federated/server.py`
* **Resolution**: Implemented `trust_client_sample_counts` configuration in `FederatedConfig`. When set to `False`, the coordinator weights all active clients equally ($1/K$), neutralizing volume inflation poisoning attacks.
* **Empirical Validation**: Validated in `tests/test_fedavg.py` and benchmarked under 20% Byzantine sample count inflation ($n_m = 10^6$), maintaining high detection F1.

---

### 3. [MEDIUM-3] Standardized 5-Feature NetFlow Core Alignment
* **Affected Components**: `src/fedids_bench/data/loaders.py`
* **Resolution**: Discarded naive column padding. Standardized on a canonical 5-feature NetFlow core (`duration`, `fwd_bytes`, `bwd_bytes`, `fwd_pkts`, `bwd_pkts`) based on Sarhan et al. (2022). All 5 real datasets map their native attributes directly to this core schema, with IoT-23 parsing Zeek `orig_pkts` and `resp_pkts`.
* **Empirical Validation**: Validated in `tests/test_loaders.py` and evaluated in cross-dataset transfer benchmarks.

---

### 4. Process Determinism & Reproducibility Verification
* **Affected Components**: `src/fedids_bench/utils/reproducibility.py`
* **Resolution**: Fixed pseudo-random seeds across PyTorch CPU, NumPy, and Python standard libraries. Verified identical metric outputs across independent subprocess evaluations.
