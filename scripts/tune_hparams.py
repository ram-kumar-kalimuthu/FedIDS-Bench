#!/usr/bin/env python3
"""
Select hyper-parameters on the VALIDATION split of the TUNING rows (never the test split, never a reporting row).

  python scripts/tune_hparams.py --data-dir /path/to/datasets --n-jobs 4        # full protocol
  python scripts/tune_hparams.py --data-dir /path/to/datasets --quick           # 5-minute dry run

Output: results/tuned_hparams.json (consumed by run_grid.py / run_transfer.py), results/tuning/trials.csv (EVERY trial,
for the supplement), results/tuning/selected.csv (winners, default-vs-tuned gain, majority baseline).
If no trial of a (corpus, algorithm) beats the majority baseline the script says so and exits with status 2.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys

import pandas as pd

from fedids_bench.config import deep_merge
from fedids_bench.experiments import grid
from fedids_bench.experiments.aggregate import load_runs, t_ci95
from fedids_bench.experiments.cli_utils import csv_list, default_n_jobs, parse_overrides, write_provenance
from fedids_bench.experiments.tuning import GROUPS, build_spec, flatten, sample_trials, select_best

ROOT = _bootstrap.ROOT
DATASETS = ["unsw_nb15", "cicids2017", "cse_cic_ids2018", "iot23", "ton_iot"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--groups", default="fedavg,fedprox,fedadam,centralized")
    ap.add_argument("--trials", type=int, default=24, help="trials per (corpus, algorithm), trial 0 = repository default")
    ap.add_argument("--seeds", default="1001,1002", help="tuning seeds (must differ from the reporting seeds)")
    ap.add_argument("--n-samples", type=int, default=30_000)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--hidden", default="64,32", help="fixed architecture (hidden layer sizes)")
    ap.add_argument("--search-hidden", action="store_true", help="also search the hidden sizes")
    ap.add_argument("--pool", choices=["first", "all"], default="all")
    ap.add_argument("--space-seed", type=int, default=2026)
    ap.add_argument("--min-gain", type=float, default=0.02, help="best validation balanced accuracy must exceed 0.5 by this much")
    ap.add_argument("--out-dir", default=None, help="default results/tuning (results/rehearsal/tuning for --quick / --set)")
    ap.add_argument("--tuned-out", default=None, help="default results/tuned_hparams.json (results/rehearsal/... for --quick / --set)")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE", help="override a config value in every trial")
    ap.add_argument("--quick", action="store_true", help="dry run: 3 trials, 1 seed, 3,000 rows, 5 rounds")
    ap.add_argument("--n-jobs", type=int, default=default_n_jobs())
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-fail", action="store_true", help="exit 0 even if a corpus shows no signal")
    args = ap.parse_args()

    datasets = csv_list(args.datasets)
    groups = csv_list(args.groups)
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        ap.error(f"unknown groups {bad}; choose from {GROUPS}")
    seeds = csv_list(args.seeds, int)
    trials, n_samples, rounds = args.trials, args.n_samples, args.rounds
    if args.quick:
        trials, seeds, n_samples, rounds = 3, seeds[:1], 3000, 5
    reporting_clash = [s for s in seeds if 0 <= s < 100]
    if reporting_clash:
        ap.error(f"tuning seeds {reporting_clash} fall in the reporting range 0..99; use e.g. 1001,1002")

    spec = build_spec(datasets, groups, seeds, trials, n_samples, rounds, csv_list(args.hidden, int), args.data_dir,
                      args.search_hidden, args.space_seed, args.pool)
    over = parse_overrides(args.sets)
    rehearsal = args.quick or bool(grid.comparable_overrides(over))
    base = os.path.join("results", "rehearsal") if rehearsal else "results"
    if args.out_dir is None:
        args.out_dir = os.path.join(base, "tuning")
    if args.tuned_out is None:
        args.tuned_out = os.path.join(base, "tuned_hparams.json")
    if rehearsal:
        print(f"REHEARSAL (--quick / --set): outputs go to {args.out_dir} and {args.tuned_out}, NOT to results/tuning; never paper results.")
    jobs = grid.expand_spec(spec, None)
    for j in jobs:
        j["overrides"] = deep_merge(j["overrides"], over)
    run_dir = os.path.join(args.out_dir, "runs")
    print(f"tuning: {len(jobs)} runs = {len(datasets)} corpora x {len(groups)} algorithms x {trials} trials x {len(seeds)} seeds "
          f"(n_samples={n_samples}, rounds={rounds}) -> {run_dir}")
    write_provenance(args.out_dir, "tuning", {"spec": spec, "overrides_cli": over}, repo_dir=ROOT)
    counts = grid.execute(jobs, run_dir, n_jobs=args.n_jobs, force=args.force)
    if counts["error"]:
        print(f"{counts['error']} trial run(s) failed; see *.error.txt in {run_dir}", file=sys.stderr)
        return 1

    final, _ = load_runs(run_dir)
    table = select_best(final, {g: trials for g in groups})
    trial_defs = {g: sample_trials(g, trials, args.space_seed, args.search_hidden) for g in groups}
    flat = table.apply(lambda r: pd.Series(flatten(trial_defs[r["group"]][int(r["trial"])])), axis=1)
    table = pd.concat([table, flat], axis=1)
    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "trials.csv"), index=False)

    tuned, rows, no_signal = {"_meta": {"trials": trials, "seeds": seeds, "n_samples": n_samples, "rounds": rounds,
                                        "hidden": csv_list(args.hidden, int), "space_seed": args.space_seed,
                                        "row_role": "tune", "selection": "mean validation balanced accuracy"}}, [], []
    for (ds, g), sub in table.groupby(["dataset", "group"]):
        best = sub[sub["selected"]].iloc[0]
        default = sub[sub["trial"] == 0].iloc[0]
        tuned.setdefault(ds, {})[g] = trial_defs[g][int(best["trial"])]
        ok = best["val_balanced_accuracy"] >= 0.5 + args.min_gain
        if not ok:
            no_signal.append((ds, g))
        rows.append({"dataset": ds, "group": g, "best_trial": int(best["trial"]),
                     "val_balanced_accuracy_best": best["val_balanced_accuracy"], "val_mcc_best": best["val_mcc"],
                     "val_balanced_accuracy_default": default["val_balanced_accuracy"],
                     "gain_over_default": best["val_balanced_accuracy"] - default["val_balanced_accuracy"],
                     "majority_balanced_accuracy": 0.5, "beats_majority": bool(ok), **flatten(trial_defs[g][int(best["trial"])])})
    sel = pd.DataFrame(rows)
    sel.to_csv(os.path.join(args.out_dir, "selected.csv"), index=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.tuned_out)), exist_ok=True)
    with open(args.tuned_out, "w", encoding="utf-8") as fh:
        json.dump(tuned, fh, indent=2)

    pd.set_option("display.width", 200)
    print("\nselected (validation split of tuning rows):")
    print(sel[["dataset", "group", "val_balanced_accuracy_default", "val_balanced_accuracy_best", "gain_over_default", "beats_majority"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {args.tuned_out}, {args.out_dir}/trials.csv, {args.out_dir}/selected.csv")
    if no_signal:
        print(f"\nWARNING: no trial beat the majority baseline by {args.min_gain} for {no_signal}. Do not run the grids for these "
              f"corpora blindly: run scripts/signal_check.py and look at the features first.", file=sys.stderr)
        return 0 if args.no_fail else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
