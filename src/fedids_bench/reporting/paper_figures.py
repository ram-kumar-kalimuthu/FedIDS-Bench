"""
Vector (PDF) figures for the paper, drawn ONLY from the run files.  Sizes follow the LNCS text block (12.2 cm = 4.8 in); text is
>= 7 pt at final size; colors are the Okabe-Ito color-blind-safe palette and every series also has its own marker / line style so the
figures survive grayscale printing.  Fonts are embedded (Type 42), nothing is rasterized.
"""
from __future__ import annotations

import io
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from fedids_bench.experiments.aggregate import convergence, summarize
from fedids_bench.reporting.paper import (
    ALGORITHM_LABELS, DEFENSE_LABELS, PARTITION_LABELS, ds_label, present_order, summarize_macro,
)

TEXT_W_IN = 4.8
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
DATASET_ORDER = ["unsw_nb15", "cicids2017", "cse_cic_ids2018", "iot23", "ton_iot"]
_PLAIN = {"clean": "None", "labelflip": "Label flip", "poison5": "Poison (γ=5)", "byz1": "Byzantine (σ=1)", "backdoor": "Backdoor",
          "inflate": "Inflate+poison"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def setup_style() -> str:
    """Embed fonts as Type 42 and pick the first serif family that really embeds (Termes = LNCS body font)."""
    plt = _plt()
    import matplotlib as mpl
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "stix", "axes.unicode_minus": False,
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "grid.linewidth": 0.4, "grid.alpha": 0.4, "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })
    chosen = "DejaVu Serif"
    for fam in ("TeX Gyre Termes", "Liberation Serif", "STIXGeneral", "DejaVu Serif"):
        try:
            mpl.rcParams["font.family"] = "serif"
            mpl.rcParams["font.serif"] = [fam]
            fig = plt.figure(figsize=(1, 1))
            fig.text(0.1, 0.5, "Test 0.5 ±")
            buf = io.BytesIO()
            fig.savefig(buf, format="pdf")
            plt.close(fig)
            if b"/Type3" in buf.getvalue()[:200000] and fam != "DejaVu Serif":
                continue
            chosen = fam
            break
        except Exception:
            continue
    mpl.rcParams["font.serif"] = [chosen]
    return chosen


def _finish(fig, path: str, banner: Optional[str]) -> str:
    plt = _plt()
    if banner:
        # on TOP of the figure (below, it overprinted the x-labels and the legend) and wrapped, so it never widens the figure
        import textwrap
        # a long stamp is shortened to its headline (the details live in completeness.json and the table captions); never cut mid-word
        text = banner if len(banner) <= 110 else banner.split(":")[0].strip() + " (incomplete grids or rehearsal settings; see completeness.json)"
        fig.text(0.5, 1.0, textwrap.fill(text, 100), ha="center", va="bottom", fontsize=5.5, color="#B00020", linespacing=1.1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def _grid_axes(n: int, ncols: int, height_per_row: float):
    plt = _plt()
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(TEXT_W_IN, height_per_row * nrows), squeeze=False)
    return fig, axes.ravel()


# ------------------------------------------------------------------------------------------------ convergence
def fig_convergence(final: pd.DataFrame, rounds: pd.DataFrame, partition: str, path: str, metric: str = "balanced_accuracy",
                    banner: Optional[str] = None) -> Optional[str]:
    plt = _plt()
    meta = final[["variant", "f_algorithm", "f_partition"]].drop_duplicates()
    sel = meta[(meta["f_partition"] == partition) & (meta["f_algorithm"] != "centralized")]
    if sel.empty:
        return None
    conv = convergence(rounds[rounds["variant"].isin(sel["variant"])], metric).merge(sel, on="variant")
    datasets = present_order(conv["dataset"].unique(), DATASET_ORDER)
    algs = [a for a in ["fedavg", "fedprox", "fedadam"] if a in set(conv["f_algorithm"])]
    fig, axes = _grid_axes(len(datasets) + 1, 3, 1.75)
    cen = summarize(final[final["f_algorithm"] == "centralized"], [metric], by=["dataset"]) if "centralized" in set(final["f_algorithm"]) else None
    for ax, ds in zip(axes, datasets):
        for i, a in enumerate(algs):
            c = conv[(conv["dataset"] == ds) & (conv["f_algorithm"] == a)].sort_values("round")
            ci = c["ci95"].fillna(0.0).to_numpy()
            ax.plot(c["round"], c["mean"], color=OKABE[i], marker=MARKERS[i], markevery=max(1, len(c) // 5), label=ALGORITHM_LABELS[a])
            ax.fill_between(c["round"], c["mean"] - ci, c["mean"] + ci, color=OKABE[i], alpha=0.18, linewidth=0)
        ax.axhline(0.5, color="0.4", ls=":", lw=0.8)
        if cen is not None and (cen["dataset"] == ds).any():
            ax.axhline(float(cen[cen["dataset"] == ds][f"{metric}_mean"].iloc[0]), color="0.2", ls="--", lw=0.8)
        ax.set_title(ds_label(ds), pad=2)
        ax.set_ylim(0.45, 1.0)
        ax.grid(True)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    for ax in axes[:len(datasets)]:
        ax.set_xlabel("round")
    for k in range(0, len(datasets), 3):
        axes[k].set_ylabel("balanced accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], color="0.4", ls=":", lw=0.8), Line2D([], [], color="0.2", ls="--", lw=0.8)]
    labels += ["majority", "centralized"]
    axes[len(datasets)].legend(handles, labels, loc="center", frameon=False, title=f"{PARTITION_LABELS.get(partition, partition)} (mean, 95% CI)".replace("$", ""))
    fig.tight_layout(pad=0.4)
    return _finish(fig, path, banner)


# ------------------------------------------------------------------------------------------------ heatmaps
def _heatmap(ax, M: np.ndarray, xt: Sequence[str], yt: Sequence[str], vmin: float, vmax: float, fmt: str = "{:.2f}", diag: bool = False,
             cmap: str = "viridis", fs: float = 7.0):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(xt)), xt, rotation=30, ha="right")
    ax.set_yticks(range(len(yt)), yt)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    thr = vmin + 0.55 * (vmax - vmin)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color="black" if v > thr else "white")
    if diag:
        for i in range(min(M.shape)):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="white", lw=1.2))
    return im


def fig_robustness(final: pd.DataFrame, path: str, metric: str = "balanced_accuracy", banner: Optional[str] = None) -> Optional[str]:
    plt = _plt()
    if final.empty:
        return None
    attacks = present_order(final["f_attack"].unique(), list(_PLAIN))
    defenses = present_order(final["f_defense"].unique(), list(DEFENSE_LABELS))
    s = summarize_macro(final, metric, ["f_attack", "f_defense"])
    M = np.full((len(attacks), len(defenses)), np.nan)
    for i, a in enumerate(attacks):
        for j, d in enumerate(defenses):
            r = s[(s["f_attack"] == a) & (s["f_defense"] == d)]
            if len(r):
                M[i, j] = r.iloc[0][f"{metric}_mean"]
    fig, ax = plt.subplots(figsize=(TEXT_W_IN, 0.42 * len(attacks) + 0.9))
    im = _heatmap(ax, M, [DEFENSE_LABELS.get(d, d) for d in defenses], [_PLAIN.get(a, a) for a in attacks], 0.5, 1.0, fs=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("balanced accuracy (macro over corpora)", fontsize=7.5)
    cb.outline.set_visible(False)
    ax.set_title("Attack × defense (20% malicious clients; 0.5 = chance)", pad=4)
    return _finish(fig, path, banner)


def fig_robustness_by_dataset(final: pd.DataFrame, path: str, metric: str = "balanced_accuracy", banner: Optional[str] = None) -> Optional[str]:
    if final.empty:
        return None
    datasets = present_order(final["dataset"].unique(), DATASET_ORDER)
    attacks = present_order(final["f_attack"].unique(), list(_PLAIN))
    defenses = present_order(final["f_defense"].unique(), list(DEFENSE_LABELS))
    fig, axes = _grid_axes(len(datasets), 3, 2.3)
    im = None
    for ax, ds in zip(axes, datasets):
        s = summarize(final[final["dataset"] == ds], [metric], by=["f_attack", "f_defense"])
        M = np.full((len(attacks), len(defenses)), np.nan)
        for i, a in enumerate(attacks):
            for j, d in enumerate(defenses):
                r = s[(s["f_attack"] == a) & (s["f_defense"] == d)]
                if len(r):
                    M[i, j] = r.iloc[0][f"{metric}_mean"]
        im = _heatmap(ax, M, [DEFENSE_LABELS.get(d, d) for d in defenses], [_PLAIN.get(a, a) for a in attacks], 0.5, 1.0, fmt="{:.2f}", fs=5.5)
        ax.set_title(ds_label(ds), pad=2)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    return _finish(fig, path, banner)


# ------------------------------------------------------------------------------------------------ sweeps
def fig_sweeps(sweep_df: pd.DataFrame, path: str, metric: str = "balanced_accuracy", banner: Optional[str] = None) -> Optional[str]:
    plt = _plt()
    if sweep_df.empty:
        return None
    from fedids_bench.reporting.paper_tables import SWEEP_ORDER
    titles = {"gamma": "poisoning strength γ", "sigma": "Byzantine scale σ", "fraction": "malicious share"}
    xl = {"g1": "1", "g5": "5", "g10": "10", "s0.5": "0.5", "s1": "1", "s5": "5", "m10": "10%", "m20": "20%", "m30": "30%", "m40": "40%"}
    sweeps = [s for s in ("gamma", "sigma", "fraction") if (sweep_df["sweep"] == s).any()]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(TEXT_W_IN, 1.9), squeeze=False, sharey=True)
    defenses = present_order(sweep_df["defense"].unique(), list(DEFENSE_LABELS))
    for ax, sw in zip(axes[0], sweeps):
        g = sweep_df[sweep_df["sweep"] == sw]
        levels = present_order(g["level"], SWEEP_ORDER[sw])
        for i, d in enumerate(defenses):
            gd = g[g["defense"] == d].set_index("level").reindex(levels)
            if gd[f"{metric}_mean"].notna().any():
                ax.errorbar(range(len(levels)), gd[f"{metric}_mean"], yerr=gd[f"{metric}_ci95"].fillna(0), color=OKABE[i % len(OKABE)],
                            marker=MARKERS[i % len(MARKERS)], capsize=1.5, label=DEFENSE_LABELS.get(d, d))
        ax.axhline(0.5, color="0.4", ls=":", lw=0.8)
        ax.set_xticks(range(len(levels)), [xl.get(x, x) for x in levels])
        ax.set_xlabel(titles[sw])
        ax.set_ylim(0.45, 1.0)
        ax.grid(True, axis="y")
    axes[0][0].set_ylabel("balanced accuracy")
    h, l = [], []
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(pad=0.4)
    return _finish(fig, path, banner)


# ------------------------------------------------------------------------------------------------ partitioning / inflation (grouped bars)
def _bars(ax, groups: Sequence[str], series: Dict[str, pd.DataFrame], metric: str, colors: Sequence[str], hatches: Sequence[str]):
    w = 0.8 / max(1, len(series))
    for k, (name, s) in enumerate(series.items()):
        x = np.arange(len(groups)) + (k - (len(series) - 1) / 2) * w
        m = s.reindex(groups)
        ax.bar(x, m[f"{metric}_mean"], w * 0.92, yerr=m[f"{metric}_ci95"].fillna(0), color=colors[k], hatch=hatches[k], edgecolor="white",
               error_kw={"elinewidth": 0.7, "capsize": 1.2}, label=name)


def _weight_handles(series: Dict[str, pd.DataFrame], colors: Sequence[str], hatches: Sequence[str]):
    """Legend handles built explicitly.  ``ax.legend(labels)`` without handles pairs the labels with the first artists it finds
    (here: the dotted majority line), which printed "Count-weighted" next to the majority line in an earlier version."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    names = {"weighted": "Count-weighted", "uniform": "Equal-weight"}
    handles = [Patch(facecolor=colors[k], hatch=hatches[k], edgecolor="white", label=names.get(n, n)) for k, n in enumerate(series)]
    handles.append(Line2D([0], [0], color="0.4", ls=":", lw=0.8, label="majority baseline (0.5)"))
    return handles


def fig_partition(final: pd.DataFrame, path: str, metric: str = "balanced_accuracy", banner: Optional[str] = None) -> Optional[str]:
    plt = _plt()
    if final.empty:
        return None
    parts = present_order(final["f_partition"].unique(), list(PARTITION_LABELS))
    s = summarize_macro(final, metric, ["f_partition", "f_weights"])
    fig, ax = plt.subplots(figsize=(TEXT_W_IN, 2.0))
    series = {w: s[s["f_weights"] == w].set_index("f_partition") for w in ("weighted", "uniform") if (s["f_weights"] == w).any()}
    _bars(ax, parts, series, metric, [OKABE[0], OKABE[1]], ["", "//"])
    ax.axhline(0.5, color="0.4", ls=":", lw=0.8)
    ax.set_xticks(range(len(parts)), [PARTITION_LABELS.get(p, p).replace("$", "") for p in parts])
    ax.set_ylabel("balanced accuracy"); ax.set_ylim(0.45, 1.0); ax.grid(True, axis="y")
    hs = _weight_handles(series, [OKABE[0], OKABE[1]], ["", "//"])
    ax.legend(handles=hs, frameon=False, ncol=len(hs), loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2,
              handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(pad=0.4)
    return _finish(fig, path, banner)


def fig_inflation(final: pd.DataFrame, path: str, metric: str = "balanced_accuracy", banner: Optional[str] = None) -> Optional[str]:
    plt = _plt()
    if final.empty:
        return None
    parts = present_order(final["f_partition"].unique(), list(PARTITION_LABELS))
    payloads = present_order(final["f_payload"].unique(), ["honest", "labelflip", "poison5"])
    lab = {"honest": "honest update", "labelflip": "+ label flip", "poison5": "+ poison (γ=5)"}
    fig, axes = plt.subplots(1, len(parts), figsize=(TEXT_W_IN, 2.0), squeeze=False, sharey=True)
    for ax, pa in zip(axes[0], parts):
        s = summarize_macro(final[final["f_partition"] == pa], metric, ["f_payload", "f_weights"])
        series = {w: s[s["f_weights"] == w].set_index("f_payload") for w in ("weighted", "uniform") if (s["f_weights"] == w).any()}
        _bars(ax, payloads, series, metric, [OKABE[0], OKABE[1]], ["", "//"])
        ax.axhline(0.5, color="0.4", ls=":", lw=0.8)
        ax.set_xticks(range(len(payloads)), [lab[p] for p in payloads], rotation=20, ha="right")
        ax.set_title(PARTITION_LABELS.get(pa, pa).replace("$", ""), pad=2)
        ax.set_ylim(0.45, 1.0); ax.grid(True, axis="y")
    axes[0][0].set_ylabel("balanced accuracy")
    hs = _weight_handles(series, [OKABE[0], OKABE[1]], ["", "//"])
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.88))
    fig.legend(handles=hs, frameon=False, ncol=len(hs), loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.2)
    return _finish(fig, path, banner)


# ------------------------------------------------------------------------------------------------ transfer
def fig_transfer(tdf: pd.DataFrame, path: str, modes: Sequence[str] = ("strict", "strict_shuffled"), metric: str = "balanced_accuracy",
                 banner: Optional[str] = None) -> Optional[str]:
    plt = _plt()
    from fedids_bench.experiments.aggregate import transfer_matrix, transfer_summary
    if tdf.empty:
        return None
    order = present_order(set(tdf["source"]) | set(tdf["target"]), DATASET_ORDER)
    titles = {"strict": "Strict zero-shot", "target_scaler": "Legacy (target scaler)", "strict_shuffled": "Control: columns shuffled"}
    fig, axes = plt.subplots(1, len(modes), figsize=(TEXT_W_IN if len(modes) > 1 else TEXT_W_IN * 0.62, 2.35), squeeze=False)
    im = None
    for ax, mode in zip(axes[0], modes):
        sm = transfer_summary(tdf, metric, mode)
        M = transfer_matrix(sm).reindex(index=order, columns=order).to_numpy(float)
        im = _heatmap(ax, M, [ds_label(o, short=True) for o in order], [ds_label(o, short=True) for o in order], 0.5, 1.0, diag=True, fs=6.5)
        ax.set_title(titles[mode], pad=3)
        ax.set_xlabel("target")
    axes[0][0].set_ylabel("source")
    cb = fig.colorbar(im, ax=list(axes[0]), fraction=0.03, pad=0.02)
    cb.set_label("balanced accuracy", fontsize=7.5)
    cb.outline.set_visible(False)
    return _finish(fig, path, banner)


# ------------------------------------------------------------------------------------------------ edge
def fig_edge(edge_csv: str, path: str, banner: Optional[str] = None) -> Optional[str]:
    plt = _plt()
    if not os.path.exists(edge_csv):
        return None
    df = pd.read_csv(edge_csv)
    an = df[df["kind"] == "analytical"]
    if an.empty:
        return None
    names = [str(n).split("(")[0].strip() for n in an["device"]]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W_IN, 1.7))
    for ax, (best, worst, title, unit) in zip(axes, [("latency_ms_per_sample_best", "latency_ms_per_sample_worst", "Latency per sample", "ms"),
                                                      ("energy_mj_per_batch_best", "energy_mj_per_batch_worst", "Energy per batch (upper bound)", "mJ")]):
        y = np.arange(len(an))
        lo, hi = an[best].to_numpy(float), an[worst].to_numpy(float)
        ax.hlines(y, lo, hi, color=OKABE[0], lw=3)
        ax.plot(lo, y, "|", color="black", ms=8)
        ax.plot(hi, y, "|", color="black", ms=8)
        ax.set_xscale("log")
        ax.set_yticks(y, names if ax is axes[0] else [""] * len(y))
        # short labels on purpose: the two long x-labels used to overlap in the middle of the figure; what the bar ends mean
        # (efficiency eta = 0.1 at the left end, 0.01 at the right end) is stated in the caption of the manuscript
        ax.set_xlabel(f"{unit} (log scale)")
        ax.set_title(title, pad=2)
        ax.grid(True, axis="x")
        ax.invert_yaxis()
    fig.tight_layout(pad=0.4)
    return _finish(fig, path, banner)
