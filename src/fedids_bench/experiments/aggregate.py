"""
Turn the per-run JSON files produced by ``grid.py`` into the numbers that appear in the paper.

Every table/figure number is a mean over seeds with a Student-t 95% confidence interval; method comparisons
are PAIRED over seeds (same data subsample, split, partition and initialization for both methods).
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

METRICS: List[str] = [
    "accuracy", "f1", "balanced_accuracy", "mcc", "auroc", "precision", "recall",
    "attack_recall", "benign_recall", "false_positive_rate", "false_negative_rate",
]


def _num(x: object) -> float:
    try:
        return float(x)           # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def load_runs(run_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (final-metrics frame, per-round frame) from every ``*.json`` run file in ``run_dir``."""
    final_rows: List[Dict[str, object]] = []
    round_rows: List[Dict[str, object]] = []
    for fn in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        if os.path.basename(fn).startswith("_"):      # _provenance_<grid>.json is metadata, not a run
            continue
        with open(fn, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        res = d["result"]
        m = res.get("metrics", {}) or {}
        maj = res.get("baseline_majority", {}) or {}
        row: Dict[str, object] = {
            "run_id": d["run_id"], "grid": d.get("grid"), "dataset": d["dataset"], "variant": d["variant"],
            "group": d.get("group"), "seed": d["seed"], "wall_time_sec": d.get("wall_time_sec"),
        }
        for fname, level in (d.get("factors") or {}).items():          # matrix factor levels -> columns f_<factor>
            row[f"f_{fname}"] = level
        for k in METRICS:
            row[k] = _num(m.get(k))
            row[f"maj_{k}"] = _num(maj.get(k))
        bd = res.get("backdoor") or {}
        row["bd_asr"], row["bd_clean_miss"], row["bd_lift"] = _num(bd.get("asr")), _num(bd.get("clean_miss")), _num(bd.get("asr_lift"))
        cm = m.get("confusion_matrix") or {}
        tot = sum(cm.values()) if cm else 0
        row["n_test"] = tot
        row["attack_prevalence_test"] = _num(m.get("attack_prevalence"))
        row["pred_attack_rate"] = (cm.get("tp", 0) + cm.get("fp", 0)) / tot if tot else float("nan")
        final_rows.append(row)
        for rl in res.get("round_logs", []) or []:
            round_rows.append({
                "dataset": d["dataset"], "variant": d["variant"], "seed": d["seed"], "round": rl["round_num"],
                "f1": _num(rl.get("tier1_global_test_f1")),
                "balanced_accuracy": _num(rl.get("tier1_global_test_bal_acc")),
                "mcc": _num(rl.get("tier1_global_test_mcc")),
            })
    return pd.DataFrame(final_rows), pd.DataFrame(round_rows)


def t_ci95(std: float, n: int) -> float:
    """Half-width of the two-sided 95% Student-t confidence interval of a mean."""
    if n < 2 or not np.isfinite(std):
        return float("nan")
    return float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))


def summarize(df: pd.DataFrame, metrics: Sequence[str], by: Sequence[str] = ("dataset", "variant")) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key, g in df.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row: Dict[str, object] = dict(zip(by, key))
        row["n_seeds"] = int(g["seed"].nunique())
        for m in metrics:
            v = g[m].dropna().to_numpy(dtype=float)
            n = len(v)
            std = float(v.std(ddof=1)) if n > 1 else float("nan")
            row[f"{m}_mean"] = float(v.mean()) if n else float("nan")
            row[f"{m}_std"] = std
            row[f"{m}_ci95"] = t_ci95(std, n)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_compare(df: pd.DataFrame, a: str, b: str, metric: str, by: str = "dataset") -> pd.DataFrame:
    """Paired (over seeds) difference  metric(a) - metric(b)  per dataset, with 95% CI and paired t-test."""
    out: List[Dict[str, object]] = []
    for key, g in df.groupby(by, sort=False):
        pa = g[g["variant"] == a].set_index("seed")[metric]
        pb = g[g["variant"] == b].set_index("seed")[metric]
        common = pa.index.intersection(pb.index)
        if len(common) < 2:
            continue
        va, vb = pa.loc[common].to_numpy(float), pb.loc[common].to_numpy(float)
        d = va - vb
        std = float(d.std(ddof=1))
        p = float("nan") if std == 0.0 else float(stats.ttest_rel(va, vb).pvalue)
        out.append({by: key, "a": a, "b": b, "metric": metric, "n_seeds": len(d), "mean_diff": float(d.mean()),
                    "ci95": t_ci95(std, len(d)), "p_value": p})
    return pd.DataFrame(out)


def beats_majority(df: pd.DataFrame, metric: str = "balanced_accuracy", margin: float = 0.0) -> pd.DataFrame:
    """Per (dataset, variant): is  metric - majority_metric  > margin with 95% confidence (over seeds)?"""
    out: List[Dict[str, object]] = []
    maj_col = f"maj_{metric}"
    for (ds, var), g in df.groupby(["dataset", "variant"], sort=False):
        d = (g[metric] - g[maj_col]).dropna().to_numpy(float)
        n = len(d)
        if n == 0:
            continue
        std = float(d.std(ddof=1)) if n > 1 else float("nan")
        ci = t_ci95(std, n)
        mean = float(d.mean())
        out.append({"dataset": ds, "variant": var, "metric": metric, "n_seeds": n, "mean_gain_over_majority": mean,
                    "ci95": ci, "beats_majority": bool(np.isfinite(ci) and (mean - ci) > margin)})
    return pd.DataFrame(out)


def convergence(rounds_df: pd.DataFrame, metric: str = "balanced_accuracy") -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (ds, var, r), g in rounds_df.groupby(["dataset", "variant", "round"], sort=True):
        v = g[metric].dropna().to_numpy(float)
        std = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        rows.append({"dataset": ds, "variant": var, "round": int(r), "n_seeds": len(v),
                     "mean": float(v.mean()) if len(v) else float("nan"), "ci95": t_ci95(std, len(v))})
    return pd.DataFrame(rows)


def fmt_pm(mean: float, ci: float, digits: int = 3) -> str:
    """'0.834±.012' style cell (leading zero dropped on the CI to save width)."""
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(ci):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{ci:.{digits}f}".replace("±0.", "±.")


# ----------------------------------------------------------------------------------------------- transfer study
TRANSFER_COLUMNS: List[str] = [
    "balanced_accuracy", "mcc", "auroc", "f1", "attack_recall", "benign_recall", "false_positive_rate",
    "false_negative_rate", "pred_attack_rate",
]


def load_transfer(run_dir: str) -> pd.DataFrame:
    """Long frame: one row per (source, target, seed, mode) from the ``transfer__*.json`` files of ``transfer.py``."""
    rows: List[Dict[str, object]] = []
    for fn in sorted(glob.glob(os.path.join(run_dir, "*__s*.json"))):
        if os.path.basename(fn).startswith("_"):
            continue
        with open(fn, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if "targets" not in d:
            continue
        for tgt, cell in d["targets"].items():
            for mode in ("strict", "target_scaler", "strict_shuffled"):
                if mode not in cell:
                    continue
                row: Dict[str, object] = {
                    "source": d["source"], "target": tgt, "seed": d["seed"], "mode": mode,
                    "is_diagonal": d["source"] == tgt, "n_test": cell.get("n_test"),
                    "attack_prevalence_test": _num(cell.get("attack_prevalence_test")),
                }
                for k in TRANSFER_COLUMNS:
                    row[k] = _num(cell[mode].get(k))
                rows.append(row)
    return pd.DataFrame(rows)


def transfer_summary(df: pd.DataFrame, metric: str = "balanced_accuracy", mode: str = "strict") -> pd.DataFrame:
    """(source, target): mean, std, 95% CI over seeds, n_seeds, plus the target's mean test prevalence."""
    sub = df[df["mode"] == mode]
    rows: List[Dict[str, object]] = []
    for (src, tgt), g in sub.groupby(["source", "target"], sort=False):
        v = g[metric].dropna().to_numpy(float)
        std = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        rows.append({"source": src, "target": tgt, "n_seeds": len(v), "mean": float(v.mean()) if len(v) else float("nan"),
                     "std": std, "ci95": t_ci95(std, len(v)), "is_diagonal": src == tgt,
                     "attack_prevalence_test": float(g["attack_prevalence_test"].mean())})
    return pd.DataFrame(rows)


def transfer_matrix(summary: pd.DataFrame, value: str = "mean") -> pd.DataFrame:
    """Pivot a ``transfer_summary`` frame to a source x target matrix of ``value``."""
    return summary.pivot(index="source", columns="target", values=value)


def transfer_diagnostics(df: pd.DataFrame, chance: float = 0.5, metric: str = "balanced_accuracy") -> pd.DataFrame:
    """Per (source, target != source): does zero-shot transfer beat chance with 95% confidence, how large is the
    drop from the in-domain cell of the SAME source model, and does the column-shuffled control collapse to chance?"""
    rows: List[Dict[str, object]] = []
    strict = df[df["mode"] == "strict"]
    shuf = df[df["mode"] == "strict_shuffled"]
    for src in strict["source"].unique():
        diag = strict[(strict["source"] == src) & (strict["target"] == src)].set_index("seed")[metric]
        for tgt in strict["target"].unique():
            if tgt == src:
                continue
            cell = strict[(strict["source"] == src) & (strict["target"] == tgt)].set_index("seed")[metric].dropna()
            if len(cell) == 0:
                continue
            ctl = shuf[(shuf["source"] == src) & (shuf["target"] == tgt)].set_index("seed")[metric].dropna()
            std = float(cell.std(ddof=1)) if len(cell) > 1 else float("nan")
            ci = t_ci95(std, len(cell))
            drop = (diag.reindex(cell.index) - cell).dropna()
            rows.append({
                "source": src, "target": tgt, "n_seeds": len(cell), "mean": float(cell.mean()), "ci95": ci,
                "beats_chance": bool(np.isfinite(ci) and (cell.mean() - ci) > chance),
                "mean_drop_from_in_domain": float(drop.mean()) if len(drop) else float("nan"),
                "shuffled_control_mean": float(ctl.mean()) if len(ctl) else float("nan"),
                "control_gap": float(cell.mean() - ctl.mean()) if len(ctl) else float("nan"),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- partition heterogeneity
def _kl_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise KL(a || b) in bits; terms with a == 0 contribute 0 (b > 0 wherever a > 0 for the mixtures used here)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(a > 0, a * np.log2(a / b), 0.0)
    return t.sum(axis=1)


def partition_stats(manifest: Dict[str, object]) -> Dict[str, float]:
    """How unequal a partition is, from the manifest stored in a run file (active, non-empty clients only).

    * ``js_mean``  mean over clients of the Jensen-Shannon divergence (base 2, range 0-1) between the client's label
      distribution and the pooled one (0 = every client looks like the whole corpus);
    * ``tv_mean``  the same with total-variation distance;
    * ``single_class_frac`` / ``no_attack_frac``  share of clients that hold one class only / no attack flow at all;
    * ``attack_share_min/max/std``  spread of the per-client attack share (class 0 is benign, every other class an attack);
    * ``size_ratio`` (largest / smallest client), ``size_cv`` (coefficient of variation of client sizes).
    """
    clients = [c for c in (manifest.get("clients") or {}).values()
               if not c.get("is_heldout") and sum(float(v) for v in (c.get("class_distribution") or {}).values()) > 0]
    if not clients:
        return {}
    classes = sorted({int(k) for c in clients for k in (c.get("class_distribution") or {})})
    counts = np.array([[float({int(k): v for k, v in (c.get("class_distribution") or {}).items()}.get(cl, 0.0)) for cl in classes]
                       for c in clients], dtype=float)
    n = counts.sum(axis=1)
    pooled = counts.sum(axis=0) / counts.sum()
    q = counts / n[:, None]
    mix = 0.5 * (q + pooled[None, :])
    jsd = 0.5 * _kl_bits(q, mix) + 0.5 * _kl_bits(np.broadcast_to(pooled, q.shape).copy(), mix)
    tv = 0.5 * np.abs(q - pooled[None, :]).sum(axis=1)
    benign = counts[:, classes.index(0)] if 0 in classes else np.zeros(len(n))
    attack = n - benign
    share = attack / n
    return {
        "n_clients": float(len(n)), "size_min": float(n.min()), "size_max": float(n.max()),
        "size_ratio": float(n.max() / n.min()), "size_cv": float(n.std() / n.mean()),
        "js_mean": float(jsd.mean()), "tv_mean": float(tv.mean()),
        "single_class_frac": float(((counts > 0).sum(axis=1) == 1).mean()), "no_attack_frac": float((attack == 0).mean()),
        "attack_share_min": float(share.min()), "attack_share_max": float(share.max()), "attack_share_std": float(share.std()),
    }


def load_partition_stats(run_dir: str) -> pd.DataFrame:
    """One row per distinct (corpus, partition, seed) with the ``partition_stats`` of the split the runs actually used.

    Runs that differ only in the aggregation weights share one partition, so they are counted once.  Run files without a
    stored manifest are skipped (nothing is imputed)."""
    rows: List[Dict[str, object]] = []
    for fn in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        if os.path.basename(fn).startswith("_"):
            continue
        with open(fn, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        man = (d.get("result") or {}).get("partition_manifest")
        if not man:
            continue
        st = partition_stats(man)
        if not st:
            continue
        row: Dict[str, object] = {"dataset": d["dataset"], "variant": d["variant"], "seed": d["seed"], "strategy": man.get("strategy")}
        for fname, level in (d.get("factors") or {}).items():
            row[f"f_{fname}"] = level
        row.update(st)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    key = ["dataset", "f_partition", "seed"] if "f_partition" in df.columns else ["dataset", "variant", "seed"]
    return df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
