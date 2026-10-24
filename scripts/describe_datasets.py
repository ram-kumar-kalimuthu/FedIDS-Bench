#!/usr/bin/env python3
"""
Describe the five corpora from the FILES (nothing typed by hand): rows, classes, attack share, native column count,
which harmonized fields were found, and the exact-duplicate rate of the harmonized fields.  Feeds paper Table 3.

  python scripts/describe_datasets.py --data-dir /path/to/datasets --pool all
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys

import pandas as pd

from fedids_bench.config import DatasetConfig
from fedids_bench.data.loaders import get_dataset_loader

DATASETS = ["unsw_nb15", "cicids2017", "cse_cic_ids2018", "iot23", "ton_iot"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--pool", choices=["first", "all"], default="all", help="'all' = every day/file of the corpus")
    ap.add_argument("--out", default=os.path.join("results", "dataset_description"), help="output prefix (.json and .csv)")
    ap.add_argument("--no-duplicates", action="store_true", help="skip the (slower) duplicate-rate estimate")
    ap.add_argument("--allow-missing", action="store_true", help="do not fail when a corpus is not found")
    args = ap.parse_args()

    out, missing = {}, []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"== {name}", flush=True)
        cfg = DatasetConfig(name=name, data_dir=args.data_dir, pool=args.pool)
        try:
            info = get_dataset_loader(name).describe(cfg, with_duplicates=not args.no_duplicates)
        except FileNotFoundError as e:
            print(f"   NOT FOUND: {e}")
            missing.append(name)
            continue
        out[name] = info
        top = ", ".join(f"{k}: {v:,}" for k, v in list(info["label_counts_top20"].items())[:5])
        print(f"   {info['n_files']} file(s), {info['n_rows']:,} rows, {info['n_native_columns']} columns, "
              f"{info['native_label_classes']} label classes, attack share {info['attack_share']:.3f}")
        print(f"   top labels: {top}")
        print(f"   harmonized fields missing: {info['fields_missing'] or 'none'}"
              + (f"; duplicate rate {info['canonical_duplicate_rate']}" if "canonical_duplicate_rate" in info else ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    rows = [{
        "dataset": k, "files": v["n_files"], "rows": v["n_rows"], "native_columns": v["n_native_columns"],
        "label_classes": v["native_label_classes"], "attack_share": v["attack_share"],
        "canonical_duplicate_rate": v.get("canonical_duplicate_rate"), "fields_missing": ";".join(v["fields_missing"]),
    } for k, v in out.items()]
    pd.DataFrame(rows).to_csv(args.out + ".csv", index=False)
    print(f"\nwrote {args.out}.json / .csv for {len(out)} corpus/corpora")
    if missing and not args.allow_missing:
        print(f"missing: {missing} (see docs/DATA.md for the expected folder and file names)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
