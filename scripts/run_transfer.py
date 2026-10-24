#!/usr/bin/env python3
"""
Strict zero-shot cross-dataset transfer (paper Table 7).

  python scripts/run_transfer.py --spec configs/grids/transfer.yaml --data-dir /path/to/datasets --n-jobs 5

Trains one federated model per (source corpus, seed) and evaluates it on the held-out test split of EVERY corpus
(strict zero-shot standardization with the SOURCE statistics, the legacy target-scaler convention, and a
column-shuffled control).  One JSON per (source, seed); finished ones are skipped on re-runs.
"""
import _bootstrap  # noqa: F401

import argparse
import os
import sys

from fedids_bench.config import deep_merge
from fedids_bench.experiments import grid
from fedids_bench.experiments.cli_utils import (
    csv_list, default_n_jobs, parse_overrides, sha256_file, write_provenance,
)
from fedids_bench.experiments.transfer import execute_transfer

ROOT = _bootstrap.ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=os.path.join("configs", "grids", "transfer.yaml"))
    ap.add_argument("--out", default=None, help="default results/runs/<spec name>")
    ap.add_argument("--tuned", default=os.path.join("results", "tuned_hparams.json"))
    ap.add_argument("--no-tuned", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--datasets", default=None, help="comma list: source AND target corpora")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--n-jobs", type=int, default=default_n_jobs(), help="processes (one per seed)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import yaml
    with open(args.spec, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    for key in ("name", "datasets", "seeds"):
        if key not in spec:
            ap.error(f"{args.spec} is missing '{key}'")
    if args.datasets:
        spec["datasets"] = [d for d in spec["datasets"] if d in csv_list(args.datasets)]
    if args.seeds:
        spec["seeds"] = csv_list(args.seeds, int)

    tuned = None
    if spec.get("group") and not args.no_tuned:
        if not os.path.exists(args.tuned):
            ap.error(f"{args.tuned} not found: run scripts/tune_hparams.py first (or pass --no-tuned)")
        tuned = grid.load_tuned(args.tuned)
        missing = [ds for ds in spec["datasets"] if spec["group"] not in tuned.get(ds, {})]
        if missing:
            ap.error(f"tuned file has no '{spec['group']}' entry for: {missing}")

    over = parse_overrides(args.sets)
    rehearsal = bool(grid.comparable_overrides(over))             # --set changes what is computed -> separate folder
    if args.data_dir:
        over = deep_merge(over, {"dataset": {"data_dir": args.data_dir}})
    spec["overrides"] = deep_merge(spec.get("overrides", {}), over)

    out_dir = args.out or os.path.join("results", "rehearsal" if rehearsal else "runs", spec["name"])
    if rehearsal and not args.out:
        print(f"REHEARSAL (--set): runs go to {out_dir}, NOT to results/runs; they are never paper results.")
    print(f"transfer '{spec['name']}': {len(spec['datasets'])} corpora x {len(spec['seeds'])} seeds -> {out_dir}")
    write_provenance(out_dir, spec["name"], {
        "spec_path": os.path.abspath(args.spec), "spec": spec,
        "tuned_path": None if tuned is None else os.path.abspath(args.tuned),
        "tuned_sha256": None if tuned is None else sha256_file(args.tuned),
    }, repo_dir=ROOT)
    counts = execute_transfer(spec, out_dir, tuned=tuned, n_jobs=args.n_jobs, force=args.force)
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
