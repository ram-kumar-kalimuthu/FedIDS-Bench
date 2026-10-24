"""
Machine-written statements of what the run files show, for ``results/report/claims.md``.

The manuscript must quote these (or the CSV rows behind them) instead of typing numbers; negative findings are listed first on purpose.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from fedids_bench.experiments.aggregate import summarize
from fedids_bench.reporting.paper import ALGORITHM_LABELS, ATTACK_LABELS, DEFENSE_LABELS, PARTITION_LABELS, ds_label, summarize_macro
from fedids_bench.reporting.paper_tables import _gain_flags


def _plain(s: str) -> str:
    return s.replace("$", "").replace("\\gamma", "γ").replace("\\sigma", "σ").replace("{", "").replace("}", "")


def main_claims(final: pd.DataFrame, paired: pd.DataFrame | None, metric: str = "balanced_accuracy") -> List[str]:
    out: List[str] = []
    fl = final[final["f_algorithm"] != "centralized"]
    flags = _gain_flags(final, metric, ["dataset", "f_algorithm", "f_partition"])
    fails = [(k, v) for k, v in flags.items() if not v]
    n_ok, n_tot = sum(flags.values()), len(flags)
    out.append(f"Cells (corpus x method x partition) whose balanced accuracy is above the majority baseline with 95% confidence: {n_ok}/{n_tot}.")
    for (ds, alg, part), _ in fails:
        out.append(f"  NOT above majority: {ds_label(ds)} / {ALGORITHM_LABELS.get(alg, alg)} / {PARTITION_LABELS.get(part, part)}")
    s = summarize(final, [metric], by=["dataset", "f_algorithm", "f_partition"])
    for alg in ["centralized", "fedavg", "fedprox", "fedadam"]:
        for part in (["none"] if alg == "centralized" else ["iid", "dir0.5", "dir0.1"]):
            g = s[(s["f_algorithm"] == alg) & (s["f_partition"] == part)]
            if len(g):
                lo, hi = g[f"{metric}_mean"].min(), g[f"{metric}_mean"].max()
                out.append(f"{ALGORITHM_LABELS.get(alg, alg)} / {PARTITION_LABELS.get(part, part)}: balanced accuracy {lo:.3f}-{hi:.3f} across {len(g)} corpora.")
    if paired is not None and len(paired):
        sig = paired[paired["p_holm"] < 0.05]
        out.append(f"Paired differences with Holm-adjusted p < 0.05: {len(sig)}/{len(paired)} comparisons.")
        for r in sig.itertuples():
            out.append(f"  {r.comparison} / {PARTITION_LABELS.get(getattr(r, 'f_partition', '--'), '--')} / {ds_label(r.dataset)}: "
                       f"{r.mean_diff:+.3f} ± {r.ci95:.3f} (p_holm = {r.p_holm:.3g})")
    return out


def partition_claims(final: pd.DataFrame, delta: pd.DataFrame | None) -> List[str]:
    out: List[str] = []
    if delta is not None and len(delta):
        m = delta[delta["dataset"] == "macro"]
        for r in m.itertuples():
            out.append(f"Equal-weight minus count-weighted FedAvg, {PARTITION_LABELS.get(r.f_partition, r.f_partition)} (macro over corpora): "
                       f"{r.mean_diff:+.3f} ± {r.ci95:.3f}.")
    return out


def heterogeneity_claims(het: pd.DataFrame | None) -> List[str]:
    out: List[str] = []
    if het is None or het.empty:
        return out
    for r in het.itertuples():
        out.append(f"{PARTITION_LABELS.get(r.partition, r.partition)} partition (macro over corpora): JS divergence {r.js_mean:.3f}, "
                   f"{r.single_class_pct:.0f}% of clients hold a single class, {r.no_attack_pct:.0f}% hold no attack flow, "
                   f"largest/smallest client {r.size_ratio:.1f}x, per-client attack share {r.attack_share_min:.2f}-{r.attack_share_max:.2f}.")
    return out


def robustness_claims(final: pd.DataFrame, metric: str = "balanced_accuracy") -> List[str]:
    out: List[str] = []
    if final.empty:
        return out
    s = summarize_macro(final, metric, ["f_attack", "f_defense"])
    for atk in s["f_attack"].unique():
        g = s[s["f_attack"] == atk].sort_values(f"{metric}_mean", ascending=False)
        best, worst = g.iloc[0], g.iloc[-1]
        out.append(f"{_plain(ATTACK_LABELS.get(atk, atk))}: best defense {DEFENSE_LABELS.get(best['f_defense'], best['f_defense'])} "
                   f"({best[f'{metric}_mean']:.3f}), worst {DEFENSE_LABELS.get(worst['f_defense'], worst['f_defense'])} ({worst[f'{metric}_mean']:.3f}) "
                   f"(macro balanced accuracy).")
    return out


def inflation_claims(final: pd.DataFrame, metric: str = "balanced_accuracy") -> List[str]:
    out: List[str] = []
    if final.empty:
        return out
    s = summarize_macro(final, metric, ["f_payload", "f_partition", "f_weights"])
    for (pl, pa), g in s.groupby(["f_payload", "f_partition"]):
        w = g[g["f_weights"] == "weighted"]
        u = g[g["f_weights"] == "uniform"]
        if len(w) and len(u):
            out.append(f"Inflation / {pl} / {PARTITION_LABELS.get(pa, pa)}: count-weighted {w.iloc[0][f'{metric}_mean']:.3f}, "
                       f"equal-weight {u.iloc[0][f'{metric}_mean']:.3f}.")
    return out


def transfer_claims(diag: pd.DataFrame | None) -> List[str]:
    out: List[str] = []
    if diag is None or diag.empty:
        return out
    n_ok = int(diag["beats_chance"].sum())
    out.append(f"Off-diagonal zero-shot cells above chance (0.5) with 95% confidence: {n_ok}/{len(diag)}.")
    out.append(f"Mean drop from the in-domain score of the same model: {diag['mean_drop_from_in_domain'].mean():+.3f}.")
    if diag["control_gap"].notna().any():
        out.append(f"Mean gap to the column-shuffled control (should be > 0 if column semantics carry the transfer): {diag['control_gap'].mean():+.3f}.")
    return out


def render(sections: Dict[str, List[str]], banner: str | None) -> str:
    lines = ["# Machine-generated findings", ""]
    if banner:
        lines += [f"> **{banner}**", ""]
    lines += ["Every number below is computed from the run files by `scripts/make_report.py`; quote these, do not retype them.", ""]
    for title, items in sections.items():
        if not items:
            continue
        lines += [f"## {title}", ""]
        for it in items:
            lines.append(it if it.startswith("  ") else f"- {it}")
        lines.append("")
    return "\n".join(lines)
