#!/usr/bin/env python3
"""
Run a declarative experiment grid (configs/grids/*.yaml).

  python scripts/run_grid.py --spec configs/grids/main.yaml --data-dir /path/to/datasets --n-jobs 4

* every run is one cached JSON in --out (default results/runs/<spec name>); re-running resumes where it stopped,
* hyper-parameters come from results/tuned_hparams.json (scripts/tune_hparams.py, validation split of tuning rows only),
* --datasets / --seeds / --set key=value narrow or override the spec (use them for quick rehearsals, never for the paper).
"""
import _bootstrap  # noqa: F401  (adds ../src to sys.path)

import argparse
import json
import os
import sys
from collections import Counter

from fedids_bench.config import deep_merge
from fedids_bench.experiments import grid
from fedids_bench.experiments.cli_utils import (
    csv_list, default_n_jobs, parse_overrides, sha256_file, write_provenance,
)

ROOT = _bootstrap.ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="grid YAML (configs/grids/*.yaml)")
    ap.add_argument("--out", default=None, help="run directory (default results/runs/<spec name>)")
    ap.add_argument("--tuned", default=os.path.join("results", "tuned_hparams.json"), help="tuned hyper-parameters (JSON)")
    ap.add_argument("--no-tuned", action="store_true", help="ignore tuned hyper-parameters (spec base values only)")
    ap.add_argument("--data-dir", default=None, help="folder that holds the corpora (sets dataset.data_dir)")
    ap.add_argument("--datasets", default=None, help="comma list, narrows the spec")
    ap.add_argument("--seeds", default=None, help="comma list of integers, replaces the spec's seeds")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value in EVERY run, e.g. dataset.n_samples=3000 (repeatable)")
    ap.add_argument("--n-jobs", type=int, default=default_n_jobs(), help="worker processes (default: half the CPUs)")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N jobs")
    ap.add_argument("--force", action="store_true", help="re-run runs that already have a result file")
    ap.add_argument("--dry-run", action="store_true", help="print the job list and exit")
    args = ap.parse_args()

    spec = grid.load_spec(args.spec)
    if args.datasets:
        spec["datasets"] = [d for d in spec["datasets"] if d in csv_list(args.datasets)]
        if not spec["datasets"]:
            ap.error(f"--datasets matches none of the spec's datasets")
    if args.seeds:
        spec["seeds"] = csv_list(args.seeds, int)

    tuned = None
    needs_tuned = any(v.get("group") for v in spec["variants"])
    if needs_tuned and not args.no_tuned:
        if not os.path.exists(args.tuned):
            ap.error(f"{args.tuned} not found: run scripts/tune_hparams.py first (or pass --no-tuned to use the spec's base values)")
        tuned = grid.load_tuned(args.tuned)
        missing = sorted({(ds, v["group"]) for ds in spec["datasets"] for v in spec["variants"]
                          if v.get("group") and v["group"] not in tuned.get(ds, {})})
        if missing:
            ap.error(f"tuned file has no entry for (dataset, group): {missing}")

    cli_over = parse_overrides(args.sets)
    # --set / --limit change WHAT is computed: such runs are rehearsals and go to their own folder, so they can neither
    # replace paper runs nor be mistaken for them (--datasets / --seeds only pick a subset of the canonical spec and stay in results/runs)
    rehearsal = bool(grid.comparable_overrides(cli_over)) or args.limit is not None
    if args.data_dir:
        cli_over = deep_merge(cli_over, {"dataset": {"data_dir": args.data_dir}})

    jobs = grid.expand_spec(spec, tuned)
    for j in jobs:
        j["overrides"] = deep_merge(j["overrides"], cli_over)

    out_dir = args.out or os.path.join("results", "rehearsal" if rehearsal else "runs", spec["name"])
    if rehearsal and not args.out:
        print(f"REHEARSAL (--set / --limit): runs go to {out_dir}, NOT to results/runs; they are never paper results.")
    per_ds = Counter(j["dataset"] for j in jobs)
    print(f"grid '{spec['name']}': {len(jobs)} runs = {len(spec['datasets'])} datasets x {len(spec['variants'])} variants "
          f"x {len(spec['seeds'])} seeds  -> {out_dir}")
    print("  per dataset:", dict(per_ds))
    if args.dry_run:
        print(json.dumps(jobs[0]["overrides"], indent=2))
        return 0

    os.makedirs(out_dir, exist_ok=True)
    write_provenance(out_dir, spec["name"], {
        "spec_path": os.path.abspath(args.spec), "spec": spec, "overrides_cli": cli_over,
        "tuned_path": None if tuned is None else os.path.abspath(args.tuned),
        "tuned_sha256": None if tuned is None else sha256_file(args.tuned),
    }, repo_dir=ROOT)
    counts = grid.execute(jobs, out_dir, n_jobs=args.n_jobs, force=args.force, limit=args.limit)
    if counts["error"]:
        print(f"{counts['error']} run(s) failed: see *.error.txt in {out_dir}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
