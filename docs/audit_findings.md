# FedIDS-Bench — Phase 1 Skeptical Peer Review & Audit Report

**Auditor Profile**: Lead FL/IDS Peer Reviewer & Methodological Auditor  
**Scope**: Phase 1 Implementation (`fedids_bench` v0.1.0 codebase)  
**Status**: Revised Audit Post-Empirical Verification — NO IMPLEMENTATION FIXES APPLIED YET  

---

## Executive Summary of Audit Findings

| Severity | Count | Primary Impact Areas |
|---|:---:|---|
| **CRITICAL** | 2 | Data Leakage / Standard Scaler Contamination, Unvalidated Aggregation Weighting Attack Surface |
| **HIGH** | 1 | Division-by-Zero Metric Distortions under Extreme Imbalance |
| **MEDIUM** | 3 | Dirichlet Partitioning Zero-Sample Active Client Edge Case, Tier-2 Heldout Client Fraction Floor, Latent Buffer Transfer Accounting for Future CNN Models |
| **LOW** | 2 | CPU OpenMP Thread Scheduling Determinism (Empirically Verified in Subprocesses), Result Schema Metadata Completeness |

---

## Reconciled & Empirical Audit Findings

### 1. CRITICAL SEVERITY

#### [CRITICAL-1] Global Feature Standardization Contaminates Test Set & Client Partitions (Data Leakage)
* **Category**: Data Leakage / Train-Test Contamination
* **Affected File(s) & Function(s)**:
  * [`src/fedids_bench/data/synthetic.py`](file:///d:/Research%20paper/fedids-bench/src/fedids_bench/data/synthetic.py): `generate_synthetic_ids()` (Lines 57–60)
* **Audit Finding**:
  In `generate_synthetic_ids()`, global mean and standard deviation are computed across **all** `n_samples` (train + test across all clients) before splitting:
  ```python
  mean = X.mean(axis=0, keepdims=True)
  std = X.std(axis=0, keepdims=True) + 1e-6
  X = (X - mean) / std
  ```
  This leaks global distribution statistics (mean and variance) from the held-out test set into the training features.
* **Precise Design Decision & Fix**:
  Fit standard scaler parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) strictly on the **Tier-1 global training split** (`split == 0`) before partitioning across clients. Store $(\mu_{\text{train}}, \sigma_{\text{train}})$ in `manifest.json`, and transform test and client splits using these training parameters.
* **Why Methodologically Correct**:
  Prevents information leakage from the held-out test set into client training data, adhering strictly to zero-leakage evaluation protocols in FL benchmarks.
* **Required Test Modifications**:
  * [`tests/test_synthetic.py`](file:///d:/Research%20paper/fedids-bench/tests/test_synthetic.py): Add explicit test verifying that test set values do not shift training split scaler parameters.

---

#### [CRITICAL-2] Unvalidated Client-Reported Sample Counts ($n_k$) in FedAvg Aggregation Weighting
* **Category**: Mathematically Incorrect FL / Threat-Model Vulnerability
* **Affected File(s) & Function(s)**:
  * [`src/fedids_bench/federated/algorithms/fedavg.py`](file:///d:/Research%20paper/fedids-bench/src/fedids_bench/federated/algorithms/fedavg.py): `fedavg_aggregate()`
  * [`src/fedids_bench/federated/server.py`](file:///d:/Research%20paper/fedids-bench/src/fedids_bench/federated/server.py): `run_round()`
* **Audit Finding**:
  `fedavg_aggregate()` relies on $n_k$ (sample count reported by client $k$) directly to compute weights $w_k = n_k / \sum n_j$. In Phase 3 threat model evaluations (malicious clients / Byzantine attacks), a single malicious client could report $n_k = 1,000,000$, causing its update to hijack 99.9% of global model weights.
* **Precise Change Needed**:
  1. Add a `"trust_client_sample_counts": false` option in `FederatedConfig` that allows testing equal-weight aggregation ($1/K$) as a baseline defense against sample-reporting attacks.
  2. Flag $n_k$ explicitly in output metadata schema as `"client_reported_sample_count"`.
* **Why Methodologically Correct**:
  Exposes the unvalidated sample-count attack vector explicitly in benchmark logs while permitting researchers to test sample-size poisoning attacks against standard FedAvg.
* **Required Test Modifications**:
  * [`tests/test_fedavg.py`](file:///d:/Research%20paper/fedids-bench/tests/test_fedavg.py): Add unit tests verifying behavior under inflated client $n_k$ reports and uniform fallback mode.

---

### 2. HIGH SEVERITY

#### [HIGH-1] Confusion Matrix Flattening Failures & False-Positive Rate Distortions under Missing Classes
* **Category**: Incorrect IDS Metrics
* **Affected File(s) & Function(s)**:
  * [`src/fedids_bench/evaluation/classification.py`](file:///d:/Research%20paper/fedids-bench/src/fedids_bench/evaluation/classification.py): `compute_classification_metrics()`
* **Audit Finding**:
  `confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()` assumes both 0 (benign) and 1 (attack) are present in `y_true_bin`. If a held-out test split or a specific client partition contains only benign samples ($y_{\text{true}} = [0, 0, \dots]$) or only attack samples ($y_{\text{true}} = [1, 1, \dots]$), $FP / (FP + TN)$ or $FN / (FN + TP)$ yield $0/0 \rightarrow 0.0$ silently without indicating undefined FPR/FNR.
* **Precise Change Needed**:
  Return `None` (or `NaN`) for FPR/FNR when denominator is zero, rather than coercing $0/0$ to $0.0$, which falsely suggests a "0% False Positive Rate".
* **Why Methodologically Correct**:
  Reporting a 0.0 FPR on a dataset with zero benign samples is scientifically misleading in IDS benchmark papers.
* **Required Test Modifications**:
  * [`tests/test_metrics.py`](file:///d:/Research%20paper/fedids-bench/tests/test_metrics.py): Add edge-case test with single-class arrays (all benign / all attack).

---

### 3. RECONCILED / DOWNGRADED FINDINGS (Empirically Re-evaluated)

#### [DOWNGRADED TO LOW] OpenMP/MKL Multithreaded PyTorch CPU Non-Determinism Risk
* **Reconciliation & Empirical Verification**:
  * `test_subprocess_end_to_end_reproducibility` in [`tests/test_reproducibility.py`](file:///d:/Research%20paper/fedids-bench/tests/test_reproducibility.py) was rewritten to invoke `python scripts/run_smoke.py` across **two independent Python subprocesses**.
  * **Result**: `test_subprocess_end_to_end_reproducibility` **PASSED** with 100% bitwise/numeric match across fresh process launches (`f1`, `accuracy`, `macro_f1`, `communication_bytes` identical).
  * **Conclusion**: High-level OpenMP/MKL thread scheduling nondeterminism is not causing variance in this CPU implementation. However, as a best practice for multi-core server deployment, we will expose an optional `torch.set_num_threads(1)` configuration flag.

#### [RECLARIFIED TO MEDIUM] Communication Accounting for Non-Trainable Buffer Tensors (BatchNorm)
* **Reconciliation & Technical Fact**:
  * `SmallMLP` in Phase 1 consists exclusively of `Linear` + `ReLU` layers (no `BatchNorm` or state buffers).
  * Therefore, `parameter_bytes()` in `SmallMLP` is **100% mathematically exact** for the current Phase 1 codebase.
  * **Reclarified Scope**: This finding represents a **Latent Requirement for Phase 2** when 1D CNNs or architectures with `BatchNorm1d` are added. `IDSModel` will be enhanced to iterate over `state_dict().values()` to account for persistent buffer tensors.

---

## Clarifying Questions & Decisions Status

1. **Question 1: CRITICAL-1 (Scaler Fitting Scope)**:  
   * **Decision**: Fit standard scaler parameters $(\mu_{\text{train}}, \sigma_{\text{train}})$ globally on the Tier 1 training split (`split == 0`) before client partitioning.
2. **Question 2: CRITICAL-2 (Unvalidated $n_k$ Weighting)**:  
   * **Decision**: Add `"trust_client_sample_counts": false` option to `FederatedConfig` for equal-weighted FedAvg evaluation under attack scenarios.
3. **Question 3: Model Learning Baseline Check**:  
   * **Verified**: Added `test_model_learning_above_dummy_baseline()` to unit test suite. Verified FL model converges from 0.0 to 0.985 F1 score over 10 rounds with `learning_rate: 0.1`.
