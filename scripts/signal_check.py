#!/usr/bin/env python3
"""
Does the harmonized feature set carry signal?  A few minutes, no federated training.

For every corpus and feature set (core5 = old 5 volume features, core22, core = 31 features) a logistic regression and
a gradient-boosted tree model are fitted on the TRAIN rows of the TUNING row-role and scored on its VALIDATION rows.
If even the boosted trees stay near AUROC 0.5 the features are the problem, not the federated algorithms -- and no
amount of FL tuning will make the benchmark learn.  Run this BEFORE the grids.

  python scripts/signal_check.py --data-dir /path/to/datasets
"""
import _bootstrap  # noqa: F401

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

from fedids_bench.config import DatasetConfig
from fedids_bench.data.loaders import get_dataset_loader

DATASETS = ["unsw_nb15", "cicids2017", "cse_cic_ids2018", "iot23", "ton_iot"]


def verdict(auroc: float) -> str:
    if not np.isfinite(auroc):
        return "n/a"
    return "OK" if auroc >= 0.75 else ("WEAK" if auroc >= 0.60 else "NO SIGNAL")


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=2).astype(float)
    w = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    return w[y]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--feature-sets", default="core5,core22,core")
    ap.add_argument("--n-samples", type=int, default=30_000)
    ap.add_argument("--pool", choices=["first", "all"], default="all")
    ap.add_argument("--seed", type=int, default=1001, help="tuning-range seed (reporting seeds are 0..4)")
    ap.add_argument("--out", default=os.path.join("results", "signal_check.csv"))
    ap.add_argument("--strict", action="store_true", help="exit 2 if any corpus has NO SIGNAL with the full feature set")
    args = ap.parse_args()

    rows, missing = [], []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for fs in [f.strip() for f in args.feature_sets.split(",") if f.strip()]:
            cfg = DatasetConfig(name=name, n_samples=args.n_samples, val_fraction=0.15, test_fraction=0.15, feature_set=fs,
                                pool=args.pool, row_role="tune", data_dir=args.data_dir)
            try:
                ds = get_dataset_loader(name).load(cfg, seed=args.seed)
            except FileNotFoundError as e:
                if name not in missing:
                    print(f"{name}: NOT FOUND ({e})")
                    missing.append(name)
                break
            tr, va = ds.split == 0, ds.split == 1
            Xtr, ytr, Xva, yva = ds.X[tr], ds.y[tr], ds.X[va], ds.y[va]
            if len(np.unique(yva)) < 2 or len(np.unique(ytr)) < 2:
                print(f"{name}/{fs}: only one class in the split -- skipped")
                continue
            lr = LogisticRegression(max_iter=500, class_weight="balanced").fit(Xtr, ytr)
            gb = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xtr, ytr, sample_weight=_balanced_weights(ytr))
            for mname, mdl in (("logreg", lr), ("boosted_trees", gb)):
                p = mdl.predict_proba(Xva)[:, 1]
                pred = (p >= 0.5).astype(int)
                auroc = float(roc_auc_score(yva, p))
                rows.append({
                    "dataset": name, "feature_set": fs, "n_features": ds.X.shape[1], "model": mname,
                    "n_train": int(tr.sum()), "n_val": int(va.sum()), "attack_prevalence_val": float(yva.mean()),
                    "val_balanced_accuracy": float(balanced_accuracy_score(yva, pred)), "val_mcc": float(matthews_corrcoef(yva, pred)),
                    "val_auroc": auroc, "verdict": verdict(auroc),
                })
            last = rows[-2:]
            print(f"{name:16s} {fs:7s} ({ds.X.shape[1]:2d} feats)  " + "   ".join(
                f"{r['model']}: bal-acc {r['val_balanced_accuracy']:.3f}  AUROC {r['val_auroc']:.3f} [{r['verdict']}]" for r in last), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    if df.empty:
        return 1
    best = df[df["model"] == "boosted_trees"]
    full = best[best["feature_set"] == best["feature_set"].iloc[-1]] if len(best) else best
    bad = full[full["verdict"] == "NO SIGNAL"]["dataset"].tolist()
    if bad:
        print(f"\nWARNING: no signal in the full feature set for {bad}. Federated learning cannot beat the majority baseline "
              f"there; fix the feature extraction (docs/DATA.md) before running any grid.", file=sys.stderr)
    return 2 if (bad and args.strict) else (1 if missing else 0)


if __name__ == "__main__":
    sys.exit(main())
