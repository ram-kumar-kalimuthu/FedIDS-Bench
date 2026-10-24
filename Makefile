.PHONY: install test smoke clean data-check tune grids transfer edge report reproduce reproduce-quick

PY    ?= python
DATA  ?= data/raw
# worker processes (each one is single-threaded torch); default: half of the CPUs
JOBS  ?= $(shell $(PY) -c "import os; print(max(1, (os.cpu_count() or 2) // 2))")
GRIDS := main partitioning robustness inflation sweeps

install:
	pip install -e .[dev]

test:
	pytest tests/ -v

smoke:
	$(PY) scripts/run_smoke.py

# ------------------------------------------------------------------------------------------------------------------------
# Paper pipeline.  Every table and figure of the paper is produced by `make report` from the run files below; nothing is typed
# by hand.  Details, timings and troubleshooting: docs/REPRODUCE.md.   Usage:  make reproduce DATA=/path/to/corpora JOBS=8
# All steps are resumable: finished runs are kept and only re-run if their settings changed.
# ------------------------------------------------------------------------------------------------------------------------

# 1. Look at the data BEFORE training anything (rows, classes, harmonized fields found, does a simple model beat chance?)
data-check:
	$(PY) scripts/describe_datasets.py --data-dir $(DATA) --pool all
	$(PY) scripts/signal_check.py --data-dir $(DATA) --strict

# 2. Hyper-parameters, chosen on the validation split of rows that are never reported (results/tuned_hparams.json)
tune:
	$(PY) scripts/tune_hparams.py --data-dir $(DATA) --n-jobs $(JOBS)

# 3. The five grids (results/runs/<grid>/*.json) and the transfer study
grids:
	@for g in $(GRIDS); do \
	  $(PY) scripts/run_grid.py --spec configs/grids/$$g.yaml --data-dir $(DATA) --n-jobs $(JOBS) || exit 1; \
	done

transfer:
	$(PY) scripts/run_transfer.py --spec configs/grids/transfer.yaml --data-dir $(DATA) --n-jobs $(JOBS)

# 4. Edge cost estimates for the model that was actually trained (analytical ranges + a measurement on this machine)
edge:
	$(PY) scripts/profile_edge.py --from-run results/runs/main/main__unsw_nb15__fl__fedavg__iid__s0.json --measure-host

# 5. Tables (LaTeX + CSV), vector figures and claims.md -> results/report/ ; refuses incomplete or rehearsal grids
report:
	$(PY) scripts/make_report.py

reproduce: data-check tune grids transfer edge report

# 30-60 minute end-to-end check on small settings: confirms that the loaders read YOUR copy of the data, that the models learn,
# and that every script runs.  Writes ONLY to results/rehearsal/ (never to results/runs), and the report it builds is stamped
# NOT PAPER-GRADE.  Run this first; if it looks wrong, fix that before starting the multi-hour `make reproduce`.
QUICK_SET := --set dataset.n_samples=5000 --set federated.rounds=5
reproduce-quick:
	$(PY) scripts/describe_datasets.py --data-dir $(DATA) --pool all --out results/rehearsal/dataset_description
	$(PY) scripts/signal_check.py --data-dir $(DATA) --n-samples 5000 --out results/rehearsal/signal_check.csv
	$(PY) scripts/tune_hparams.py --data-dir $(DATA) --quick --n-jobs $(JOBS) --no-fail
	$(PY) scripts/run_grid.py --spec configs/grids/main.yaml --data-dir $(DATA) --seeds 0,1 $(QUICK_SET) \
	      --tuned results/rehearsal/tuned_hparams.json --n-jobs $(JOBS)
	$(PY) scripts/run_transfer.py --spec configs/grids/transfer.yaml --data-dir $(DATA) --seeds 0,1 $(QUICK_SET) \
	      --tuned results/rehearsal/tuned_hparams.json --n-jobs $(JOBS)
	$(PY) scripts/profile_edge.py --from-run results/rehearsal/main/main__unsw_nb15__fl__fedavg__iid__s0.json \
	      --out results/rehearsal/edge_profile.csv
	$(PY) scripts/make_report.py --only main,transfer --runs results/rehearsal --out results/rehearsal/report \
	      --tuned results/rehearsal/tuned_hparams.json --tuning-dir results/rehearsal/tuning \
	      --edge results/rehearsal/edge_profile.csv --description results/rehearsal/dataset_description.json --allow-rehearsal

# removes build products and caches only; results/ is never touched (the old target deleted it)
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
