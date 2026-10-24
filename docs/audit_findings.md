# FedIDS-Bench — Methodological Audit (v2)

**Scope**: code, committed results and manuscript claims of FedIDS-Bench as submitted for ICEAI 2026 (repository state `f11beb7`).  
**Supersedes**: the v1 audit, which declared every finding "VERIFIED & RESOLVED". Several of those resolutions held for the *code
path* but not for the *numbers* the repository had committed; the points below are what the second audit found and what was changed.  
**This file makes no claim about results.** Results exist only after `make reproduce` has been run on the real corpora
(`docs/REPRODUCE.md`); the report generator then states, per cell, whether the model beats the majority baseline.

---

## A. Findings of the second audit and their resolution

| ID | Severity | Finding (evidence) | Resolution | Guarded by |
|---|:---:|---|---|---|
| **A1** | Critical | **The committed federated results carry no evidence of learning.** 19 of the 25 federated cells in `results/table3_fl_algorithms.csv` equal the majority-class accuracy to three decimals (e.g. CSE-CIC-IDS2018: all five cells 0.9575 = majority; ToN_IoT: all five 0.7125 = majority). Every attack x defense cell in `results/table4_robustness.csv` lies between 0.8304 and 0.8321, i.e. a constant predictor. A defense cannot be compared on a model that predicts one class. | New harmonized 31-feature core (log-scaled volumes, sizes, asymmetry, rates, protocol, service group, port range, connection outcome); validation-split tuning of the client learning rate, local epochs, batch size, class weighting and momentum (and `mu` / server step / `tau`) with one shared protocol; `signal_check.py` tests whether *any* simple model finds signal **before** federated training; `make_report` prints which cells are not above the majority baseline instead of hiding them. | `tests/test_features_pool.py`, `tests/test_grid_aggregate.py::test_beats_majority_false_when_at_chance`, `scripts/signal_check.py --strict` |
| **A2** | Critical | **CICIDS2017 column mapping matched names with a leading space after the headers had already been stripped**, so 4 of the 5 harmonized volume features were constant zero on that corpus (reproduced on a fixture with the real header layout). | Matching on `strip().lower()` for every corpus; `describe_datasets.py` prints the harmonized fields found/missing per corpus from the files. | `test_cicids_leading_space_headers_are_mapped`, `test_cicids_end_to_end_no_constant_volume_features` |
| **A3** | Critical | **Loaders read the first N rows of time-ordered captures and sampled with replacement**: single-day, single-attack subsets, and duplicate rows that can land in both train and test. | Uniform sampling over the whole file(s) without replacement (chunked Bernoulli pool), all day-files pooled (`pool: all`), exact duplicates removed *before* any split, mixed header spellings and `latin-1` handled. | `test_sampling_is_uniform_over_the_whole_file`, `test_exact_duplicates_are_removed_before_splitting`, `test_multi_file_pooling_different_headers_and_latin1` |
| **A4** | High | **`auroc` was a placeholder equal to balanced accuracy** (`classification.py`: `"auroc": bin_balanced_acc`). | Real ROC-AUC from model scores (`predict_proba`), `None` when no scores exist; the transfer study reports it too. | `tests/test_audit_regressions.py::test_auroc_is_computed_from_scores_not_copied_from_balanced_accuracy`, `tests/test_cross_dataset.py` |
| **A5** | High | **`label_skew` is a stratified IID split** (every class is divided evenly among clients), so the "label skew" ablation measured nothing. | Documented as such; a genuine `single_domain` partitioner is added (benign traffic plus at most `L-1` attack categories per client). | `tests/test_single_domain.py` |
| **A6** | High | **Edge-cost and figure data were typed into `scripts/run_all_paper_experiments.py`** as dictionary literals (the docstring says "based on FLOPs formula", the formula is never executed); committed figures pre-date the CSVs they claim to be drawn from. | `evaluation/edge_profile.py` computes parameters, FLOPs, latency, energy and per-round traffic from the model that was trained and reports ranges (`eta` in [0.01, 0.10]); a host measurement can be added. Tables and vector figures are produced by `scripts/make_report.py` from run files only. | `tests/test_edge_profile.py::test_profile_is_computed_not_typed_in`, `tests/test_reporting.py`, `tests/test_features_pool.py::test_describe_reports_nothing_hand_typed` |
| **A7** | High | **Hyper-parameters were untuned and shared by every algorithm; no validation split existed**, so any later tuning would have been done on the test set. | Stratified train / validation / test splits; a *content-hashed* `tune` / `report` row split so tuning rows can never appear in a reported test set; tuning scores the validation split only and uses seeds disjoint from the reporting seeds; every trial is saved (`results/tuning/trials.csv`). | `tests/test_row_roles.py` |
| **A8** | Medium | **Single seed, no uncertainty.** | 5 reporting seeds; mean +/- 95 % Student-t CI; paired comparisons with Holm correction; explicit "beats majority with 95 % confidence" test. | `tests/test_grid_aggregate.py` |
| **A9** | Medium | **Cross-dataset transfer was not zero-shot**: the target corpus' own scaler was applied to the target data, and there was no control. | Strict zero-shot (source-corpus scaler) is the headline; the target-scaler convention is an ablation; a column-shuffled control must collapse if the model uses feature semantics. | `tests/test_backdoor_transfer.py::test_column_shuffle_control_destroys_a_model_that_uses_feature_semantics`, `::test_transfer_study_and_aggregation` |
| **A10** | Medium | **Backdoor success was never measured** (only clean accuracy was reported under the attack). | Attack-success-rate on triggered attack flows is computed and reported for backdoor runs only. | `tests/test_backdoor_transfer.py::test_backdoor_success_separates_a_backdoored_from_a_clean_model` |
| **A11** | Medium | **Sample-count spoofing had no implementation** (only the defense flag `trust_client_sample_counts=False` existed), so the claim that uniform weighting defends against it was untested. | `sample_inflation` attack (reported `n_k` = 10^6, optionally combined with another attack); the `inflation` grid compares count-weighted and uniform aggregation. *Caveat in the paper text:* equal weighting ignores `n_k` by construction, so it is neutral to inflation trivially; the empirical question is its cost on clean data under quantity skew (`partitioning` grid). | `tests/test_audit_regressions.py::test_inflated_count_dominates_count_weighted_fedavg_but_not_uniform_fedavg`, `::test_attackers_are_active_clients_even_when_clients_are_held_out` |
| **A12** | Medium | **Reported numbers could silently come from different settings** than the ones stated in the paper (grids re-run with different options, cached files reused). | Every run file stores its full settings and code revision; `make_report` verifies each run against the grid spec and the tuned file and refuses incomplete, rehearsal or mixed-revision grids. | `tests/test_run_integrity.py` |
| **A13** | Low | **"Bitwise reproducibility across hardware" was claimed without evidence.** | Claim replaced by what is tested: same seed, same library versions and thread settings give the same results. | `tests/test_reproducibility.py` |
| **A14** | High | **Attackers were drawn from the held-out clients whenever `heldout_client_fraction` > 0.** The server marked `set(range(num_malicious))` as malicious, but held-out clients are the *first* ids, so with the shipped `smoke` / `standard` / `large` configs (`heldout_client_fraction: 0.2`) no participating client was ever attacked and any "robustness" run on those configs measured a clean federation. (The paper's own adversarial runs and all new grids use `heldout_client_fraction: 0.0`, so they were not affected by *this* bug — they are affected by A1.) | Attackers are the first `int(n_active x fraction)` **active** clients. | `tests/test_audit_regressions.py::test_attackers_are_active_clients_even_when_clients_are_held_out` (fails on the old logic) |
| **A15** | Medium | **Random streams collided across runs.** Client training used `seed + 10000 x client + round`, client selection `seed + round`, and the loaders `seed + chunk`: (seed 0, round 1) and (seed 1, round 0) of the same client shared one stream, so the "independent" reporting seeds behind every confidence interval were partly correlated. | `derive_seed(*parts)` (numpy `SeedSequence`) for client training, client selection and loader sampling; `LOADER_VERSION` bumped so cached pools are rebuilt. | `tests/test_audit_regressions.py::test_derived_seeds_do_not_collide_across_runs_clients_and_rounds` |
| **A16** | Medium | **The backdoor poisoned the *first* rows of a client's shard.** Partitioners that lay classes out one after another (`label_skew`, `single_domain`) then poison a single class, e.g. relabel benign flows as benign, which is no backdoor at all. | The poisoned rows are a seeded random subset of the shard (the old "first rows" behavior remains available when no seed is given). | `tests/test_attacks.py::test_backdoor_attack` |
| **A17** | Low | **`median` was biased downwards for an even number of clients**: `torch.median` returns the *lower* middle value, and the grids use 10 clients. | Coordinate-wise median averages the two middle values. | `tests/test_audit_regressions.py::test_median_of_an_even_number_of_clients_averages_the_two_middle_values` |
| **A18** | Medium | **Krum feasibility was checked against the nominal client count**, not against the clients that actually take part (held-out clients do not train), and the attacker count was computed in two places. | One helper (`config.n_malicious`) defines the number of attackers for the server, the validation and the result files; Krum validation uses participating clients and `max(krum_f, simulated attackers)`. | `tests/test_audit_regressions.py::test_krum_feasibility_counts_participating_clients_and_simulated_attackers`, `::test_heldout_rule_is_shared_by_config_validation_and_partitioner` |

## B. v1 findings that still hold

| Finding | Status |
|---|---|
| CRITICAL-1 — scaler fitted on all data before splitting | Fixed and kept: statistics come from the training rows only (`test_loader_shapes_splits_and_zero_leakage`). |
| CRITICAL-2 — client-reported `n_k` trusted | The *defense* (`trust_client_sample_counts=False`) is kept; see A11 for the missing attack and the correct reading of the result. |
| HIGH-1 — division-by-zero metric distortions | Kept (`zero_division=0`, MCC, balanced accuracy). |
| MEDIUM-1 / MEDIUM-2 — empty Dirichlet shards, held-out client floor | Kept. |
| MEDIUM-3 — feature alignment across corpora | **Superseded.** The 5-feature core is replaced by the 31-feature core (A1); `core5` remains as an ablation. |

## C. Known limitations (state them in the paper)

* **Real data were never available to the build environment.** Every dataset host was unreachable from the sandbox in which this code
  was prepared, so the loaders were tested on stand-in files that copy the documented column headers. The first run on the real files can
  still reveal a header variant; `make data-check` is designed to show that *before* any long run.
* **Missing fields are explicit.** UNSW-NB15 (partitioned CSVs) has no destination port; CICIDS2017 (MachineLearningCVE) has no protocol
  or service column. Those features are the `unknown` category, which limits what cross-corpus transfer can show.
* **Edge numbers are analytical bounds**, not measurements of a Raspberry Pi or Jetson. Raspberry Pi 4 and workstation parameters are
  assumptions unless you supply cited values through `--device-file`; a host measurement describes the host only.
* **One small architecture** (MLP with hidden layers 64-32) and one family of attacks per type; conclusions do not extend to other
  architectures or adaptive attackers.
* **Statistical power**: five seeds give wide intervals for small effects; the report marks which differences survive Holm correction.

## D. How to re-verify

```bash
pytest tests/ -q                                  # 176 passed, 5 skipped without the real corpora
python scripts/describe_datasets.py --data-dir DIR --pool all
python scripts/signal_check.py --data-dir DIR --strict
make reproduce-quick DATA=DIR                     # small end-to-end run, stamped NOT PAPER-GRADE
```
