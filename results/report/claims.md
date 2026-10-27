# Machine-generated findings

Every number below is computed from the run files by `scripts/make_report.py`; quote these, do not retype them.

## FL algorithms

- Cells (corpus x method x partition) whose balanced accuracy is above the majority baseline with 95% confidence: 50/50.
- Central. / --: balanced accuracy 0.874-0.992 across 5 corpora.
- FedAvg / IID: balanced accuracy 0.861-0.988 across 5 corpora.
- FedAvg / Dir(0.5): balanced accuracy 0.856-0.984 across 5 corpora.
- FedAvg / Dir(0.1): balanced accuracy 0.766-0.978 across 5 corpora.
- FedProx / IID: balanced accuracy 0.861-0.987 across 5 corpora.
- FedProx / Dir(0.5): balanced accuracy 0.855-0.984 across 5 corpora.
- FedProx / Dir(0.1): balanced accuracy 0.812-0.977 across 5 corpora.
- FedAdam / IID: balanced accuracy 0.853-0.986 across 5 corpora.
- FedAdam / Dir(0.5): balanced accuracy 0.839-0.978 across 5 corpora.
- FedAdam / Dir(0.1): balanced accuracy 0.729-0.977 across 5 corpora.
- Paired differences with Holm-adjusted p < 0.05: 1/35 comparisons.
  fedadam - fedavg / IID / CSE-CIC-IDS2018: -0.007 ± 0.002 (p_holm = 0.0456)

## Partitioning

- Equal-weight minus count-weighted FedAvg, Dir(0.1) (macro over corpora): +0.028 ± 0.034.
- Equal-weight minus count-weighted FedAvg, Dir(0.5) (macro over corpora): +0.001 ± 0.002.
- Equal-weight minus count-weighted FedAvg, Domain-1 (macro over corpora): +0.037 ± 0.063.
- Equal-weight minus count-weighted FedAvg, Domain-2 (macro over corpora): -0.025 ± 0.030.
- Equal-weight minus count-weighted FedAvg, IID (macro over corpora): +0.000 ± 0.000.
- Equal-weight minus count-weighted FedAvg, Pareto(1.5) (macro over corpora): -0.001 ± 0.001.
- Equal-weight minus count-weighted FedAvg, Qty-skew (macro over corpora): -0.002 ± 0.002.

## Partition heterogeneity

- IID partition (macro over corpora): JS divergence 0.000, 0% of clients hold a single class, 0% hold no attack flow, largest/smallest client 1.0x, per-client attack share 0.40-0.43.
- Dir(0.5) partition (macro over corpora): JS divergence 0.124, 3% of clients hold a single class, 2% hold no attack flow, largest/smallest client 510.9x, per-client attack share 0.03-0.94.
- Dir(0.1) partition (macro over corpora): JS divergence 0.247, 45% of clients hold a single class, 27% hold no attack flow, largest/smallest client 6783.6x, per-client attack share 0.00-1.00.
- Qty-skew partition (macro over corpora): JS divergence 0.004, 1% of clients hold a single class, 0% hold no attack flow, largest/smallest client 573.2x, per-client attack share 0.36-0.48.
- Pareto(1.5) partition (macro over corpora): JS divergence 0.000, 0% of clients hold a single class, 0% hold no attack flow, largest/smallest client 7.7x, per-client attack share 0.40-0.43.
- Domain-2 partition (macro over corpora): JS divergence 0.096, 0% of clients hold a single class, 0% hold no attack flow, largest/smallest client 4.3x, per-client attack share 0.01-0.65.
- Domain-1 partition (macro over corpora): JS divergence 0.403, 98% of clients hold a single class, 9% hold no attack flow, largest/smallest client 8534.2x, per-client attack share 0.00-1.00.

## Robustness

- Backdoor: best defense Equal-w. (0.946), worst Krum (0.883) (macro balanced accuracy).
- Byzantine (σ=1): best defense Equal-w. (0.942), worst Krum (0.856) (macro balanced accuracy).
- None: best defense Equal-w. (0.947), worst Krum (0.881) (macro balanced accuracy).
- Inflate+poison: best defense Trim (0.921), worst Clip (0.499) (macro balanced accuracy).
- Label flip: best defense Trim (0.939), worst FedAvg (0.837) (macro balanced accuracy).
- Poison (γ=5): best defense Trim (0.921), worst Equal-w. (0.500) (macro balanced accuracy).

## Sample-count inflation

- Inflation / honest / Dir(0.5): count-weighted 0.939, equal-weight 0.947.
- Inflation / honest / Pareto(1.5): count-weighted 0.947, equal-weight 0.951.
- Inflation / labelflip / Dir(0.5): count-weighted 0.121, equal-weight 0.860.
- Inflation / labelflip / Pareto(1.5): count-weighted 0.084, equal-weight 0.889.
- Inflation / poison5 / Dir(0.5): count-weighted 0.500, equal-weight 0.500.
- Inflation / poison5 / Pareto(1.5): count-weighted 0.500, equal-weight 0.498.

## Transfer

- Off-diagonal zero-shot cells above chance (0.5) with 95% confidence: 4/20.
- Mean drop from the in-domain score of the same model: +0.426.
- Mean gap to the column-shuffled control (should be > 0 if column semantics carry the transfer): +0.018.
