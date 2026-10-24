#!/usr/bin/env python3
"""
Build every table (LaTeX + CSV) and figure (vector PDF) of the paper from the run files -- nothing is typed by hand.

  python scripts/make_report.py                       # results/runs/* + results/tuning + results/edge_profile.csv -> results/report/
  python scripts/make_report.py --only main,transfer  # a subset of the grids
  python scripts/make_report.py --allow-partial       # build from an incomplete grid (every artifact is stamped NOT PAPER-GRADE)

The script REFUSES to build from a grid that is missing runs the spec declares, or that was made with command-line overrides /
narrowed datasets / different seeds (those are rehearsals) unless the corresponding flag is given -- and then stamps a visible banner on
every table caption and figure.  It never invents or interpolates a missing number: absent cells print as "--".
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys

import pandas as pd

from fedids_bench.experiments.aggregate import load_partition_stats, load_runs, load_transfer
from fedids_bench.experiments.cli_utils import csv_list
from fedids_bench.reporting import claims as claimlib
from fedids_bench.reporting import paper as P
from fedids_bench.reporting import paper_figures as F
from fedids_bench.reporting import paper_tables as T

ROOT = _bootstrap.ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join("results", "runs"), help="folder with one sub-folder per grid")
    ap.add_argument("--specs", default=os.path.join("configs", "grids"))
    ap.add_argument("--out", default=os.path.join("results", "report"))
    ap.add_argument("--only", default=None, help="comma list of grids (default: every grid that has a run folder)")
    ap.add_argument("--tuning-dir", default=os.path.join("results", "tuning"))
    ap.add_argument("--tuned", default=os.path.join("results", "tuned_hparams.json"),
                    help="tuned hyper-parameters the grids were run with (every run's settings are verified against spec + this file)")
    ap.add_argument("--edge", default=os.path.join("results", "edge_profile.csv"))
    ap.add_argument("--description", default=os.path.join("results", "dataset_description.json"))
    ap.add_argument("--metric", default="balanced_accuracy")
    ap.add_argument("--allow-partial", action="store_true", help="build from grids with missing runs (stamped NOT PAPER-GRADE)")
    ap.add_argument("--allow-rehearsal", action="store_true", help="accept runs made with --set / --datasets / --seeds overrides (stamped)")
    ap.add_argument("--style", choices=["booktabs", "hline"], default="booktabs")
    args = ap.parse_args()

    wanted = csv_list(args.only) or list(P.PAPER_GRIDS)
    statuses, present = [], []
    for name in wanted:
        run_dir = os.path.join(args.runs, name)
        spec = os.path.join(args.specs, f"{name}.yaml")
        if not os.path.isdir(run_dir):
            if args.only:
                print(f"error: no run folder {run_dir}", file=sys.stderr)
                return 2
            continue
        st = P.grid_status(name, run_dir, spec, tuned_path=args.tuned)
        statuses.append(st)
        present.append(name)
        print(f"{name:13s} {st.n_found}/{st.n_expected} runs" + (f"  MISSING {len(st.missing)}" if st.missing else "")
              + (f"  ERRORS {len(st.errors)}" if st.errors else "") + (f"  FLAGS: {'; '.join(st.flags)}" if st.flags else ""))
        for w in st.warnings:
            print(f"{'':13s} warning: {w}")
    if not statuses:
        print(f"error: no grid folders under {args.runs}", file=sys.stderr)
        return 2
    refuse = []
    for st in statuses:
        if not st.complete and not args.allow_partial:
            refuse.append(f"{st.name}: {len(st.missing)} missing run(s) (e.g. {st.missing[:2]}) / {len(st.errors)} failed run(s)")
        if st.flags and not args.allow_rehearsal:
            refuse.append(f"{st.name}: {'; '.join(st.flags)}")
    if refuse:
        print("\nREFUSING to build the report:\n  " + "\n  ".join(refuse) +
              "\n(finish the grids, or pass --allow-partial / --allow-rehearsal to build a stamped, non-paper report)", file=sys.stderr)
        return 2
    banner = P.banner(statuses, args.allow_partial, args.allow_rehearsal)
    if banner:
        print(f"\n{banner}")

    tdir, fdir = os.path.join(args.out, "tables"), os.path.join(args.out, "figures")
    font = F.setup_style()
    print(f"figure font: {font}")
    tables, figures, frames = [], [], {}

    def keep(ts):
        for t in ts:
            tables.append(t)
            if t.data is not None and len(t.data):
                os.makedirs(tdir, exist_ok=True)
                t.data.to_csv(os.path.join(tdir, t.name + ".csv"), index=False)
            if t.tex:
                P.write_artifact(os.path.join(tdir, t.name + ".tex"), t.tex)

    def fig(path_name, fn, *a, **kw):
        out = fn(*a, os.path.join(fdir, path_name), banner=banner, **kw)
        if out:
            figures.append(out)

    metric = args.metric
    for name in present:
        run_dir = os.path.join(args.runs, name)
        if name == "transfer":
            tdf = load_transfer(run_dir)
            frames["transfer"] = tdf
            keep(T.build_transfer(tdf, banner, metric))
            fig("fig_transfer.pdf", F.fig_transfer, tdf, metric=metric)
            fig("fig_transfer_legacy.pdf", F.fig_transfer, tdf, modes=("target_scaler",), metric=metric)
            continue
        final, rounds = load_runs(run_dir)
        frames[name] = final
        if name == "main":
            keep(T.build_main(final, banner))
            for part in ("iid", "dir0.5", "dir0.1"):
                fig(f"fig_convergence_{part}.pdf", F.fig_convergence, final, rounds, part, metric=metric)
        elif name == "partitioning":
            keep(T.build_partitioning(final, banner, metric))
            keep(T.build_partition_heterogeneity(load_partition_stats(run_dir), banner))
            fig("fig_partition.pdf", F.fig_partition, final, metric=metric)
        elif name == "robustness":
            keep(T.build_robustness(final, banner, metric))
            fig("fig_robustness.pdf", F.fig_robustness, final, metric=metric)
            fig("fig_robustness_by_dataset.pdf", F.fig_robustness_by_dataset, final, metric=metric)
        elif name == "inflation":
            keep(T.build_inflation(final, banner, metric))
            fig("fig_inflation.pdf", F.fig_inflation, final, metric=metric)
        elif name == "sweeps":
            keep(T.build_sweeps(final, banner, metric))
            fig("fig_sweeps.pdf", F.fig_sweeps, T.sweep_frame(final, metric), metric=metric)

    keep(T.build_edge(args.edge, banner))
    fig("fig_edge.pdf", F.fig_edge, args.edge)
    keep(T.build_tuning(os.path.join(args.tuning_dir, "selected.csv"), os.path.join(args.tuning_dir, "trials.csv"), banner))
    keep(T.build_datasets(args.description, os.path.join(args.runs, "main"), banner))

    by_name = {t.name: t for t in tables}
    sections = {}
    if "main" in frames:
        sections["FL algorithms"] = claimlib.main_claims(frames["main"], by_name.get("main_paired").data if "main_paired" in by_name else None, metric)
    if "partitioning" in frames:
        sections["Partitioning"] = claimlib.partition_claims(frames["partitioning"], by_name.get("partition_delta").data if "partition_delta" in by_name else None)
        sections["Partition heterogeneity"] = claimlib.heterogeneity_claims(
            by_name.get("partition_heterogeneity").data if "partition_heterogeneity" in by_name else None)
    if "robustness" in frames:
        sections["Robustness"] = claimlib.robustness_claims(frames["robustness"], metric)
    if "inflation" in frames:
        sections["Sample-count inflation"] = claimlib.inflation_claims(frames["inflation"], metric)
    if "transfer" in frames:
        sections["Transfer"] = claimlib.transfer_claims(by_name.get("transfer_diagnostics").data if "transfer_diagnostics" in by_name else None)
    P.write_artifact(os.path.join(args.out, "claims.md"), claimlib.render(sections, banner))
    P.write_artifact(os.path.join(args.out, "numbers.tex"), P.numbers_tex(statuses, frames))
    P.write_artifact(os.path.join(args.out, "completeness.json"), json.dumps(
        {"banner": banner, "grids": [s.to_dict() for s in statuses], "tables": sorted(t.name for t in tables), "figures": sorted(os.path.basename(f) for f in figures)},
        indent=2))
    print(f"\nwrote {len([t for t in tables if t.tex])} LaTeX tables, {len(tables)} CSV tables, {len(figures)} figures, claims.md -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
