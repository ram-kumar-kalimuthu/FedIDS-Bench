"""
Table builders for the paper report.  Every function takes the run frames written by ``experiments.grid`` /
``experiments.transfer`` (never typed numbers) and returns ``Table`` objects: a tidy CSV frame with ALL numbers
(mean, std, 95% CI, n) and, where the paper needs one, the LaTeX float.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from fedids_bench.experiments.aggregate import (
    summarize, t_ci95, transfer_diagnostics, transfer_matrix, transfer_summary,
)
from fedids_bench.reporting.paper import (
    ALGORITHM_LABELS, ATTACK_LABELS, DEFENSE_LABELS, PARTITION_LABELS, PAYLOAD_LABELS, WEIGHT_LABELS,
    ds_label, macro_by_seed, paired_diff, present_order, summarize_macro, tex_escape, tex_num, tex_table,
)

NOT_ABOVE = r"$^{\ast}$"          # marks a cell whose 95% CI does not clear the majority-class baseline
SIG = r"$^{\dagger}$"             # marks a paired difference with Holm-adjusted p < 0.05
METRIC_NAMES = {"balanced_accuracy": "balanced accuracy", "mcc": "MCC", "auroc": "AUROC", "f1": "attack-class F1",
                "accuracy": "accuracy"}
DATASET_ORDER = ["unsw_nb15", "cicids2017", "cse_cic_ids2018", "iot23", "ton_iot"]


@dataclass
class Table:
    name: str
    data: pd.DataFrame
    tex: Optional[str] = None


# ------------------------------------------------------------------------------------------------ small utilities
def _gain_flags(df: pd.DataFrame, metric: str, by: Sequence[str]) -> Dict[tuple, bool]:
    """(key) -> True when  metric - majority_metric  clears 0 with 95% confidence over seeds."""
    flags: Dict[tuple, bool] = {}
    maj = f"maj_{metric}"
    for key, g in df.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        d = (g[metric] - g[maj]).dropna().to_numpy(float)
        n = len(d)
        std = float(d.std(ddof=1)) if n > 1 else float("nan")
        ci = t_ci95(std, n)
        flags[key] = bool(np.isfinite(ci) and d.mean() - ci > 0.0)
    return flags


def _row(s: pd.DataFrame, **where) -> Optional[pd.Series]:
    m = np.ones(len(s), dtype=bool)
    for k, v in where.items():
        m &= (s[k] == v).to_numpy()
    sub = s[m]
    return None if sub.empty else sub.iloc[0]


def _get(s: pd.DataFrame, col: str, **where) -> float:
    r = _row(s, **where)
    return float("nan") if r is None else float(r[col])


def _cell(s: Optional[pd.Series], metric: str, flag: str = "", bold: bool = False) -> str:
    if s is None:
        return "--"
    return tex_num(s[f"{metric}_mean"], s[f"{metric}_ci95"], 3, bold=bold, flag=flag)


# ------------------------------------------------------------------------------------------------ FL algorithms (main grid)
def build_main(final: pd.DataFrame, banner: Optional[str] = None, metrics: Sequence[str] = ("balanced_accuracy", "mcc", "auroc", "f1"),
               ) -> List[Table]:
    """Centralized reference + FedAvg / FedProx / FedAdam x {IID, Dir(0.5), Dir(0.1)}: one table per metric, plus paired differences."""
    out: List[Table] = []
    datasets = present_order(final["dataset"].unique(), DATASET_ORDER)
    parts = [p for p in ["iid", "dir0.5", "dir0.1"] if p in set(final["f_partition"])]
    algs = [a for a in ["fedavg", "fedprox", "fedadam"] if a in set(final["f_algorithm"])]
    s = summarize(final, list(metrics), by=["dataset", "f_algorithm", "f_partition"])
    csv_rows: List[Dict[str, Any]] = []
    for metric in metrics:
        flags = _gain_flags(final, metric, ["dataset", "f_algorithm", "f_partition"])
        rows: List[List[str]] = []
        for ds in datasets:
            for pi, part in enumerate(parts):
                cells = [ds_label(ds) if pi == 0 else "", PARTITION_LABELS.get(part, part)]
                if "centralized" in set(final["f_algorithm"]):
                    c = _row(s, dataset=ds, f_algorithm="centralized", f_partition="none")
                    cells.append(_cell(c, metric, "" if (c is None or flags.get((ds, "centralized", "none"), False)) else NOT_ABOVE)
                                 if pi == 0 else "")
                means = {a: _get(s, f"{metric}_mean", dataset=ds, f_algorithm=a, f_partition=part) for a in algs}
                best = max(means, key=lambda a: means[a] if np.isfinite(means[a]) else -np.inf) if means else None
                for a in algs:
                    r = _row(s, dataset=ds, f_algorithm=a, f_partition=part)
                    ok = flags.get((ds, a, part), False)
                    cells.append(_cell(r, metric, "" if (r is None or ok) else NOT_ABOVE, bold=(a == best and r is not None)))
                    if r is not None:
                        csv_rows.append({"metric": metric, "dataset": ds, "algorithm": a, "partition": part, "n_seeds": int(r["n_seeds"]),
                                         "mean": r[f"{metric}_mean"], "std": r[f"{metric}_std"], "ci95": r[f"{metric}_ci95"],
                                         "beats_majority_95": ok})
                rows.append(cells)
        if "centralized" in set(final["f_algorithm"]):
            for ds in datasets:
                r = _row(s, dataset=ds, f_algorithm="centralized", f_partition="none")
                if r is not None:
                    csv_rows.append({"metric": metric, "dataset": ds, "algorithm": "centralized", "partition": "none",
                                     "n_seeds": int(r["n_seeds"]), "mean": r[f"{metric}_mean"], "std": r[f"{metric}_std"],
                                     "ci95": r[f"{metric}_ci95"], "beats_majority_95": flags.get((ds, "centralized", "none"), False)})
        header = ["Corpus", "Partition"] + (["Central."] if "centralized" in set(final["f_algorithm"]) else []) + \
                 [ALGORITHM_LABELS[a] for a in algs]
        colspec = "ll" + "c" * (len(header) - 2)
        maj_txt = {"balanced_accuracy": "0.5", "auroc": "0.5", "mcc": "0",
                   # F1 of the attack class depends on which class is the majority: a constant "benign" predictor scores 0,
                   # a constant "attack" predictor scores 2p/(1+p) with p the attack share; which class is the majority is a property of
                   # the sampled data (read from the run files), not assumed here
                   "f1": r"0 where benign flows are the majority and $2p/(1{+}p)$ where attack flows are ($p$ = attack share; the star test uses the value of each run)",
                   }.get(metric, "the majority value")
        note = (rf"Mean $\pm$ 95\% Student-$t$ CI over {int(s['n_seeds'].max())} seeds, "
                rf"{METRIC_NAMES.get(metric, metric)} on the held-out test split. The majority-class baseline scores {maj_txt}. "
                rf"{NOT_ABOVE} = not above that baseline with 95\% confidence. Bold = highest mean among the federated algorithms "
                r"(differences are tested in the paired table). The centralized model does not depend on the partition.")
        tex = tex_table(header, rows, colspec, caption=rf"Detection quality ({METRIC_NAMES.get(metric, metric)}) of the centralized reference and "
                        rf"the three federated algorithms.", label=f"tab:main_{metric}", note=note, banner_text=banner,
                        midrules_after=[i * len(parts) + len(parts) - 1 for i in range(len(datasets) - 1)])
        out.append(Table(f"main_{metric}", pd.DataFrame([r for r in csv_rows if r["metric"] == metric]), tex))

    # paired differences against FedAvg (same seed = same subsample, split, partition and initial weights)
    pm = metrics[0]
    fl = final[final["f_algorithm"].isin(algs)]
    diffs: List[pd.DataFrame] = []
    for other in [a for a in algs if a != "fedavg"]:
        d = paired_diff(fl, "f_algorithm", other, "fedavg", pm, ["dataset", "f_partition", "seed"], ["dataset", "f_partition"])
        if len(d):
            d.insert(0, "comparison", f"{other} - fedavg")
            diffs.append(d)
    if "iid" in parts and "dir0.1" in parts:
        sub = final[(final["f_algorithm"] == "fedavg") & final["f_partition"].isin(["iid", "dir0.1"])]
        d = paired_diff(sub, "f_partition", "iid", "dir0.1", pm, ["dataset", "seed"], ["dataset"])
        if len(d):
            d.insert(0, "comparison", "fedavg: iid - dir0.1")
            d["f_partition"] = "--"
            diffs.append(d)
    if diffs:
        dd = pd.concat(diffs, ignore_index=True)
        from fedids_bench.reporting.paper import holm
        dd["p_holm"] = holm(dd["p_value"].to_numpy())          # one family over every comparison in this table
        rows = []
        for comp, g in dd.groupby("comparison", sort=False):
            for part, gp in g.groupby("f_partition", sort=False):
                cells = [comp.replace("fedavg", "FedAvg").replace("fedprox", "FedProx").replace("fedadam", "FedAdam").replace("iid", "IID")
                         .replace("dir0.1", "Dir(0.1)").replace(" - ", r" $-$ "), PARTITION_LABELS.get(part, part)]
                for ds in datasets:
                    r = gp[gp["dataset"] == ds]
                    cells.append("--" if r.empty else tex_num(r.iloc[0]["mean_diff"], r.iloc[0]["ci95"], 3,
                                                              flag=SIG if r.iloc[0]["p_holm"] < 0.05 else ""))
                rows.append(cells)
        tex = tex_table(["Comparison", "Partition"] + [ds_label(d, short=True) for d in datasets], rows, "ll" + "c" * len(datasets),
                        caption=rf"Paired differences in {METRIC_NAMES.get(pm, pm)} (same seed $\Rightarrow$ same data, split, partition, initial weights).",
                        label="tab:main_paired", tabcolsep_pt=2.0,
                        note=rf"Mean difference $\pm$ 95\% CI over seeds. {SIG} = Holm-adjusted paired $t$-test $p<0.05$ "
                             r"(family: every comparison in this table). Short corpus names: UNSW = UNSW-NB15, CIC17 = CICIDS2017, "
                             r"CSE18 = CSE-CIC-IDS2018, IoT23 = IoT-23, ToN = ToN-IoT.", banner_text=banner)
        out.append(Table("main_paired", dd, tex))
    return out


# ------------------------------------------------------------------------------------------------ partitioning ablation
def build_partitioning(final: pd.DataFrame, banner: Optional[str] = None, metric: str = "balanced_accuracy") -> List[Table]:
    out: List[Table] = []
    datasets = present_order(final["dataset"].unique(), DATASET_ORDER)
    parts = present_order(final["f_partition"].unique(), list(PARTITION_LABELS))
    s = summarize(final, [metric, "mcc", "f1", "auroc"], by=["dataset", "f_partition", "f_weights"])
    full = s.rename(columns={"f_partition": "partition", "f_weights": "weights"})
    out.append(Table("partition_full", full, None))
    flags = _gain_flags(final, metric, ["dataset", "f_partition", "f_weights"])

    # (a) absolute level: count-weighted FedAvg per partition x corpus
    rows = []
    for part in parts:
        cells = [PARTITION_LABELS.get(part, part)]
        for ds in datasets:
            r = _row(s, dataset=ds, f_partition=part, f_weights="weighted")
            cells.append(_cell(r, metric, "" if (r is None or flags.get((ds, part, "weighted"), False)) else NOT_ABOVE))
        rows.append(cells)
    tex = tex_table(["Partition"] + [ds_label(d, short=True) for d in datasets], rows, "l" + "c" * len(datasets),
                    caption=rf"FedAvg ({METRIC_NAMES[metric]}) under {len(parts)} ways of splitting the data over 10 clients (count-weighted aggregation).",
                    label="tab:partition_level",
                    note=rf"Mean $\pm$ 95\% CI over seeds. {NOT_ABOVE} = not above the majority baseline (0.5) with 95\% confidence. "
                         r"Dir = Dirichlet label skew; Qty-skew = Dirichlet quantity skew ($\alpha{=}0.5$); Pareto(1.5) = Pareto quantity skew; "
                         r"Domain-$L$ = every client holds benign traffic plus $L{-}1$ attack categories (Domain-1: strictly one category).",
                    banner_text=banner)
    out.append(Table("partition_level", pd.DataFrame(rows, columns=["partition"] + datasets), tex))

    # (b) clean-case cost of equal weights: uniform - weighted, paired over seeds
    d = paired_diff(final, "f_weights", "uniform", "weighted", metric, ["dataset", "f_partition", "seed"], ["dataset", "f_partition"])
    if len(d):
        macro = final.groupby(["f_partition", "f_weights", "seed"], as_index=False)[metric].mean()
        wide = macro.pivot_table(index=["f_partition", "seed"], columns="f_weights", values=metric)
        if {"uniform", "weighted"} <= set(wide.columns):
            md = (wide["uniform"] - wide["weighted"]).dropna().reset_index(name="diff")
            mrows = []
            for part, g in md.groupby("f_partition"):
                v = g["diff"].to_numpy(float)
                std = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
                mrows.append({"f_partition": part, "dataset": "macro", "n": len(v), "mean_diff": float(v.mean()), "ci95": t_ci95(std, len(v)),
                              "p_value": np.nan, "p_holm": np.nan})
            d = pd.concat([d, pd.DataFrame(mrows)], ignore_index=True)
        rows = []
        for part in parts:
            cells = [PARTITION_LABELS.get(part, part)]
            for ds in datasets + ["macro"]:
                r = d[(d["dataset"] == ds) & (d["f_partition"] == part)]
                cells.append("--" if r.empty else tex_num(r.iloc[0]["mean_diff"], r.iloc[0]["ci95"], 3,
                                                          flag=SIG if (np.isfinite(r.iloc[0]["p_holm"]) and r.iloc[0]["p_holm"] < 0.05) else ""))
            rows.append(cells)
        tex = tex_table(["Partition"] + [ds_label(x, short=True) for x in datasets] + ["Macro"], rows, "l" + "c" * (len(datasets) + 1),
                        caption=rf"Clean-case cost of equal aggregation weights: {METRIC_NAMES[metric]} of equal-weight minus count-weighted FedAvg.",
                        label="tab:partition_delta",
                        note=rf"Paired over seeds, mean difference $\pm$ 95\% CI. {SIG} = Holm-adjusted $p<0.05$ across corpora and partitions. "
                             r"Macro = per-seed average over the five corpora.", banner_text=banner)
        out.append(Table("partition_delta", d, tex))
    return out


# ------------------------------------------------------------------------------------------------ partition heterogeneity
HETEROGENEITY_COLUMNS = ("js_mean", "single_class_pct", "no_attack_pct", "size_ratio", "attack_share_min", "attack_share_max")


def _pm_plain(mean: float, ci: float, digits: int) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(ci):
        return f"{mean:.{digits}f}"
    return rf"{mean:.{digits}f} $\pm$ {ci:.{digits}f}"


def build_partition_heterogeneity(stats: pd.DataFrame, banner: Optional[str] = None) -> List[Table]:
    """How unequal the splits really are (answers 'how were alpha and the other partition parameters chosen / justified').

    ``stats`` comes from ``aggregate.load_partition_stats``: one row per (corpus, partition, seed), measured on the
    partitions the FedAvg runs actually used."""
    if stats is None or stats.empty or "f_partition" not in stats.columns:
        return []
    st = stats.copy()
    st["single_class_pct"] = 100.0 * st["single_class_frac"]
    st["no_attack_pct"] = 100.0 * st["no_attack_frac"]
    parts = present_order(st["f_partition"].unique(), list(PARTITION_LABELS))
    by_ds = summarize(st, list(HETEROGENEITY_COLUMNS) + ["tv_mean", "size_cv"], by=["dataset", "f_partition"])
    macro = {c: summarize_macro(st, c, ["f_partition"]) for c in HETEROGENEITY_COLUMNS}
    rows: List[List[str]] = []
    frame: List[Dict[str, Any]] = []
    n_seeds = int(st.groupby(["dataset", "f_partition"])["seed"].nunique().min())
    for part in parts:
        rec: Dict[str, Any] = {"partition": part}
        cells = [PARTITION_LABELS.get(part, part)]
        for col, digits in (("js_mean", 3), ("single_class_pct", 0), ("no_attack_pct", 0), ("size_ratio", 1)):
            r = _row(macro[col], f_partition=part)
            m, c = (float("nan"), float("nan")) if r is None else (float(r[f"{col}_mean"]), float(r[f"{col}_ci95"]))
            rec[col], rec[f"{col}_ci95"] = m, c
            cells.append(_pm_plain(m, c, digits))
        lo = _row(macro["attack_share_min"], f_partition=part)
        hi = _row(macro["attack_share_max"], f_partition=part)
        lo_m = float("nan") if lo is None else float(lo["attack_share_min_mean"])
        hi_m = float("nan") if hi is None else float(hi["attack_share_max_mean"])
        rec["attack_share_min"], rec["attack_share_max"] = lo_m, hi_m
        cells.append("--" if not (np.isfinite(lo_m) and np.isfinite(hi_m)) else f"{lo_m:.2f}--{hi_m:.2f}")
        rows.append(cells)
        frame.append(rec)
    tex = tex_table(
        ["Partition", "JS div.", r"Single-class (\%)", r"No attack (\%)", "Size max/min", "Attack share"], rows, "lccccc",
        caption="Measured heterogeneity of the client partitions (10 clients) used by the FedAvg runs.",
        label="tab:partition_heterogeneity",
        note=rf"Macro-average over the corpora per seed, then mean $\pm$ 95\% CI over {n_seeds} seeds. JS div. = Jensen--Shannon divergence "
             r"(base 2, range 0--1) between a client's label distribution and the pooled one, averaged over clients. Single-class / No attack = "
             r"share of clients that hold only one class / no attack flow at all. Size max/min = largest over smallest client. "
             r"Attack share = fraction of a client's flows that are attacks (mean of the smallest and of the largest client share). "
             r"Dir = Dirichlet label skew; Qty-skew = Dirichlet quantity skew; Domain-$L$ = benign plus $L{-}1$ attack categories per client. "
             r"All divergences use the binary training label, so a split that only separates attack \emph{categories} (Domain-2) looks homogeneous here although its clients see different traffic.",
        banner_text=banner)
    by_ds = by_ds.rename(columns={"f_partition": "partition"})
    return [Table("partition_heterogeneity_by_corpus", by_ds, None), Table("partition_heterogeneity", pd.DataFrame(frame), tex)]


# ------------------------------------------------------------------------------------------------ robustness
def _attack_defense_table(df: pd.DataFrame, metric: str, name: str, caption: str, label: str, banner: Optional[str],
                          macro: bool, note_extra: str = "") -> Table:
    attacks = present_order(df["f_attack"].unique(), list(ATTACK_LABELS))
    defenses = present_order(df["f_defense"].unique(), list(DEFENSE_LABELS))
    if macro:
        s = summarize_macro(df, metric, ["f_attack", "f_defense"])
    else:
        s = summarize(df, [metric], by=["f_attack", "f_defense"])
    rows = []
    for atk in attacks:
        cells = [ATTACK_LABELS.get(atk, tex_escape(atk))]
        means = {d: _get(s, f"{metric}_mean", f_attack=atk, f_defense=d) for d in defenses}
        best = max(means, key=lambda d: means[d] if np.isfinite(means[d]) else -np.inf)
        for d in defenses:
            cells.append(_cell(_row(s, f_attack=atk, f_defense=d), metric, bold=(d == best)))
        rows.append(cells)
    note = (rf"{'Macro-average over the corpora per seed, then mean' if macro else 'Mean'} $\pm$ 95\% CI over {int(s['n_seeds'].max())} seeds; "
            r"20\% of the clients are malicious (Krum: $f{=}2$). Bold = best defense for that attack. " + note_extra)
    tex = tex_table(["Attack"] + [DEFENSE_LABELS.get(d, d) for d in defenses], rows, "l" + "c" * len(defenses),
                    caption=caption, label=label, note=note, banner_text=banner)
    return Table(name, s.rename(columns={"f_attack": "attack", "f_defense": "defense"}), tex)


def build_robustness(final: pd.DataFrame, banner: Optional[str] = None, metric: str = "balanced_accuracy") -> List[Table]:
    out: List[Table] = []
    datasets = present_order(final["dataset"].unique(), DATASET_ORDER)
    out.append(_attack_defense_table(
        final, metric, "robustness_macro", rf"Robustness: {METRIC_NAMES[metric]} for every attack and defense (macro-average over corpora).",
        "tab:robust_macro", banner, macro=True))
    for ds in datasets:
        out.append(_attack_defense_table(
            final[final["dataset"] == ds], metric, f"robustness_{ds}", rf"Robustness on {ds_label(ds)}: {METRIC_NAMES[metric]}.",
            f"tab:robust_{ds}", banner, macro=False))
    # full numbers for the supplement
    out.append(Table("robustness_all_metrics", summarize(
        final, ["balanced_accuracy", "mcc", "f1", "auroc", "false_positive_rate", "attack_recall", "bd_asr"],
        by=["dataset", "f_attack", "f_defense"]), None))
    # drop caused by each attack (paired with the clean run of the same defense) and the clean-case cost of each defense
    if "clean" in set(final["f_attack"]):
        clean = final[final["f_attack"] == "clean"].set_index(["dataset", "f_defense", "seed"])[metric]
        att = final[final["f_attack"] != "clean"].copy()
        att["clean_value"] = [clean.get((r.dataset, r.f_defense, r.seed), np.nan) for r in att.itertuples()]
        att["drop"] = att["clean_value"] - att[metric]
        out.append(Table("robustness_drop", summarize(att.dropna(subset=["drop"]), ["drop"], by=["dataset", "f_attack", "f_defense"]), None))
        cl = final[final["f_attack"] == "clean"]
        costs = []
        for dfn in [d for d in present_order(cl["f_defense"].unique(), list(DEFENSE_LABELS)) if d != "fedavg"]:
            c = paired_diff(cl, "f_defense", dfn, "fedavg", metric, ["dataset", "seed"], ["dataset"])
            if len(c):
                c.insert(0, "defense", dfn)
                costs.append(c)
        if costs:
            out.append(Table("robustness_clean_cost", pd.concat(costs, ignore_index=True), None))
    # backdoor success rate
    bd = final[final["f_attack"] == "backdoor"]
    if len(bd) and bd["bd_asr"].notna().any():
        defenses = present_order(bd["f_defense"].unique(), list(DEFENSE_LABELS))
        s = summarize_macro(bd.dropna(subset=["bd_asr"]), "bd_asr", ["f_defense"])
        cells = []
        for d in defenses:
            r = _row(s, f_defense=d)
            cells.append(_cell(r, "bd_asr"))
        tex = tex_table([DEFENSE_LABELS.get(d, d) for d in defenses], [cells], "c" * len(defenses),
                        caption="Backdoor attack success rate (share of triggered attack flows classified benign), macro-average over corpora.",
                        label="tab:backdoor_asr", note=r"Mean $\pm$ 95\% CI over seeds; lower is better.", banner_text=banner)
        out.append(Table("backdoor_asr", s.rename(columns={"f_defense": "defense"}), tex))
    return out


# ------------------------------------------------------------------------------------------------ sample-count inflation
def build_inflation(final: pd.DataFrame, banner: Optional[str] = None, metric: str = "balanced_accuracy") -> List[Table]:
    out: List[Table] = []
    datasets = present_order(final["dataset"].unique(), DATASET_ORDER)
    out.append(Table("inflation_full", summarize(final, [metric, "mcc", "f1", "auroc"], by=["dataset", "f_payload", "f_partition", "f_weights"]), None))
    payloads = present_order(final["f_payload"].unique(), list(PAYLOAD_LABELS))
    parts = present_order(final["f_partition"].unique(), list(PARTITION_LABELS))
    s = summarize_macro(final, metric, ["f_payload", "f_partition", "f_weights"])
    d = paired_diff(final, "f_weights", "uniform", "weighted", metric, ["dataset", "f_payload", "f_partition", "seed"],
                    ["dataset", "f_payload", "f_partition"])
    rows = []
    for pl in payloads:
        for pa in parts:
            w, u = _row(s, f_payload=pl, f_partition=pa, f_weights="weighted"), _row(s, f_payload=pl, f_partition=pa, f_weights="uniform")
            macro_pairs = final.groupby(["f_payload", "f_partition", "f_weights", "seed"], as_index=False)[metric].mean()
            wide = macro_pairs[(macro_pairs["f_payload"] == pl) & (macro_pairs["f_partition"] == pa)].pivot_table(
                index="seed", columns="f_weights", values=metric)
            if {"uniform", "weighted"} <= set(wide.columns):
                v = (wide["uniform"] - wide["weighted"]).dropna().to_numpy(float)
                std = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
                dcell = tex_num(float(v.mean()), t_ci95(std, len(v)), 3) if len(v) else "--"
            else:
                dcell = "--"
            rows.append([PAYLOAD_LABELS.get(pl, pl) if pa == parts[0] else "", PARTITION_LABELS.get(pa, pa), _cell(w, metric), _cell(u, metric), dcell])
    tex = tex_table(["Malicious update", "Partition", "Count-weighted", "Equal-weight", r"Equal $-$ count"], rows, "llccc",
                    caption=rf"Sample-count spoofing: 20\% of the clients report $10^6$ samples. {METRIC_NAMES[metric].capitalize()}, macro-average over corpora.",
                    label="tab:inflation",
                    note=r"Equal-weight FedAvg ignores reported counts, so it is unaffected by the lie by construction; its column shows the "
                         r"cost of equal weights when the same malicious updates are present. Mean $\pm$ 95\% CI over seeds.", banner_text=banner)
    out.append(Table("inflation", pd.concat([s, d], ignore_index=True, sort=False), tex))
    return out


# ------------------------------------------------------------------------------------------------ sensitivity sweeps
def sweep_frame(final: pd.DataFrame, metric: str = "balanced_accuracy") -> pd.DataFrame:
    """Long frame: sweep, level, defense, macro mean / CI over seeds (rows come from ``f_gamma`` / ``f_sigma`` / ``f_fraction``)."""
    rows: List[pd.DataFrame] = []
    for sweep, col in (("gamma", "f_gamma"), ("sigma", "f_sigma"), ("fraction", "f_fraction")):
        if col not in final.columns:
            continue
        sub = final[final[col].notna()]
        if sub.empty:
            continue
        s = summarize_macro(sub, metric, [col, "f_defense"]).rename(columns={col: "level", "f_defense": "defense"})
        s.insert(0, "sweep", sweep)
        rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


SWEEP_ORDER = {"gamma": ["g1", "g5", "g10"], "sigma": ["s0.5", "s1", "s5"], "fraction": ["m10", "m20", "m30", "m40"]}
SWEEP_LEVEL_LABELS = {"g1": r"$\gamma{=}1$", "g5": r"$\gamma{=}5$", "g10": r"$\gamma{=}10$", "s0.5": r"$\sigma{=}0.5$", "s1": r"$\sigma{=}1$",
                      "s5": r"$\sigma{=}5$", "m10": "10\\%", "m20": "20\\%", "m30": "30\\%", "m40": "40\\%"}


def build_sweeps(final: pd.DataFrame, banner: Optional[str] = None, metric: str = "balanced_accuracy") -> List[Table]:
    sf = sweep_frame(final, metric)
    if sf.empty:
        return []
    out = [Table("sweeps_macro", sf, None)]
    for sweep, title in (("gamma", r"model-poisoning strength $\gamma$ (20\% attackers)"),
                         ("sigma", r"Byzantine noise scale $\sigma$ (20\% attackers)"),
                         ("fraction", r"share of malicious clients (model poisoning, $\gamma{=}5$)")):
        g = sf[sf["sweep"] == sweep]
        if g.empty:
            continue
        levels = present_order(g["level"], SWEEP_ORDER[sweep])
        defenses = present_order(g["defense"].unique(), list(DEFENSE_LABELS))
        rows = []
        for lv in levels:
            cells = [SWEEP_LEVEL_LABELS.get(lv, tex_escape(lv))]
            for d in defenses:
                r = _row(g, level=lv, defense=d)
                cells.append(_cell(r, metric))
            rows.append(cells)
        tex = tex_table(["Level"] + [DEFENSE_LABELS.get(d, d) for d in defenses], rows, "l" + "c" * len(defenses),
                        caption=rf"Sensitivity to {title}: {METRIC_NAMES[metric]}, macro-average over corpora.", label=f"tab:sweep_{sweep}",
                        note=r"Mean $\pm$ 95\% CI over seeds. Krum uses $f$ = the true number of attackers; (40\%, Krum) violates $n>2f+2$ "
                             r"and is not run." if sweep == "fraction" else r"Mean $\pm$ 95\% CI over seeds.", banner_text=banner)
        out.append(Table(f"sweep_{sweep}", g, tex))
    return out


# ------------------------------------------------------------------------------------------------ transfer
def build_transfer(tdf: pd.DataFrame, banner: Optional[str] = None, metric: str = "balanced_accuracy") -> List[Table]:
    out: List[Table] = []
    if tdf.empty:
        return out
    order = present_order(set(tdf["source"]) | set(tdf["target"]), DATASET_ORDER)
    titles = {"strict": "strict zero-shot (source-train standardization)", "target_scaler": "legacy convention (target-train standardization)",
              "strict_shuffled": "control: strict, feature columns randomly permuted"}
    for mode in ("strict", "target_scaler", "strict_shuffled"):
        sm = transfer_summary(tdf, metric, mode)
        if sm.empty:
            continue
        rows = []
        for src in order:
            cells = [ds_label(src, short=True)]
            for tgt in order:
                r = sm[(sm["source"] == src) & (sm["target"] == tgt)]
                if r.empty:
                    cells.append("--")
                else:
                    r0 = r.iloc[0]
                    cells.append(tex_num(r0["mean"], r0["ci95"], 3, bold=bool(r0["is_diagonal"])))
            rows.append(cells)
        tex = tex_table([r"Source $\downarrow$ / target"] + [ds_label(t, short=True) for t in order], rows, "l" + "c" * len(order),
                        caption=rf"Cross-corpus transfer ({METRIC_NAMES[metric]}), {titles[mode]}.", label=f"tab:transfer_{mode}",
                        note=r"One FedAvg model per source corpus and seed; bold diagonal = in-domain test split of the same model. "
                             r"Mean $\pm$ 95\% CI over seeds; chance = 0.5. Short corpus names as in the paired-difference table.",
                        banner_text=banner)
        out.append(Table(f"transfer_{mode}", sm, tex))
    diag = transfer_diagnostics(tdf, 0.5, metric)
    if len(diag):
        out.append(Table("transfer_diagnostics", diag, None))
    prev = tdf[tdf["mode"] == "strict"].groupby("target", as_index=False)["attack_prevalence_test"].mean()
    out.append(Table("transfer_target_prevalence", prev, None))
    return out


# ------------------------------------------------------------------------------------------------ edge, tuning, datasets
def build_edge(path: str, banner: Optional[str] = None) -> List[Table]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    an = df[df["kind"] == "analytical"]
    if an.empty:
        return [Table("edge_profile", df, None)]
    r0 = an.iloc[0]
    rows = []
    details = []
    for _, r in an.iterrows():
        full = str(r["device"])
        short, _, rest = full.partition(" (")
        name = tex_escape(short)
        if "replace with" in full.lower():          # the default laptop row: TDP/peak are placeholders until the author names the CPU
            name += r"\,\placeholdertext{CPU model?}"
        elif rest:
            details.append(f"{tex_escape(short)}: {tex_escape(rest.rstrip(')'))}")
        rows.append([name, f"{r['tdp_w']:g}", f"{r['peak_gflops']:g}",
                     f"{r['latency_ms_per_sample_best']:.4f}--{r['latency_ms_per_sample_worst']:.4f}",
                     f"{r['energy_mj_per_batch_best']:.3f}--{r['energy_mj_per_batch_worst']:.3f}"])
    meas = df[df["kind"] == "measured"]
    note = (rf"Model {tex_escape(r0['model_dims'])}: {int(r0['n_params']):,} parameters ({int(r0['param_bytes']):,} bytes), "
            rf"{int(r0['flops_per_sample']):,} FLOPs per sample, {int(r0['traffic_bytes_per_client_round']):,} bytes per client per round. "
            rf"Latency = FLOPs / (peak $\times\,\eta$), $\eta\in[{r0['eta_worst']:g},{r0['eta_best']:g}]$; energy = batch {int(r0['batch_size'])} $\times$ "
            r"latency $\times$ TDP (an upper bound). These are analytical ESTIMATES, not measurements; TDP and peak rates marked as assumptions "
            r"in the CSV are not manufacturer ratings.")
    if details:
        note += " " + "; ".join(details) + "."
    if len(meas):
        m = meas.iloc[0]
        note += rf" Measured on the build host: {m['latency_ms_per_sample_measured']:.5f} ms/sample (NumPy)."
    tex = tex_table(["Device", "TDP (W)", "Peak (GFLOP/s)", "Latency (ms/sample)", "Energy (mJ/batch)"], rows, "lccrr",
                    caption="Edge cost of one local inference pass, computed from the trained model.", label="tab:edge", note=note, banner_text=banner)
    return [Table("edge_profile", df, tex)]


def build_tuning(selected_csv: str, trials_csv: Optional[str], banner: Optional[str] = None) -> List[Table]:
    if not os.path.exists(selected_csv):
        return []
    sel = pd.read_csv(selected_csv)
    out = [Table("tuning_selected", sel, None)]
    if trials_csv and os.path.exists(trials_csv):
        out.append(Table("tuning_trials", pd.read_csv(trials_csv), None))
    datasets = present_order(sel["dataset"].unique(), DATASET_ORDER)
    groups = present_order(sel["group"].unique(), ["centralized", "fedavg", "fedprox", "fedadam"])
    rows = []
    for ds in datasets:
        for gi, g in enumerate(groups):
            r = sel[(sel["dataset"] == ds) & (sel["group"] == g)]
            if r.empty:
                continue
            r0 = r.iloc[0]
            rows.append([ds_label(ds) if gi == 0 else "", ALGORITHM_LABELS.get(g, g), f"{r0['val_balanced_accuracy_default']:.3f}",
                         f"{r0['val_balanced_accuracy_best']:.3f}", f"{r0['gain_over_default']:+.3f}"])
    tex = tex_table(["Corpus", "Method", "Default", "Tuned", "Gain"], rows, "llccc",
                    caption="Hyper-parameter tuning: validation balanced accuracy of the repository defaults and of the selected configuration.",
                    label="tab:tuning", note=r"Selected on the validation split of tuning rows only (rows that never occur in a reported test split); "
                    r"the same number of random trials for every method; every trial is in the supplementary CSV.", banner_text=banner)
    out.append(Table("tuning", sel, tex))
    return out


def build_datasets(desc_json: str, main_dir: Optional[str], banner: Optional[str] = None) -> List[Table]:
    if not os.path.exists(desc_json):
        return []
    with open(desc_json, "r", encoding="utf-8") as fh:
        desc = json.load(fh)
    used: Dict[str, Dict[str, Any]] = {}
    if main_dir and os.path.isdir(main_dir):
        for fn in sorted(glob.glob(os.path.join(main_dir, "*.json"))):
            if os.path.basename(fn).startswith("_"):
                continue
            with open(fn, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            used.setdefault(d["dataset"], d["result"].get("dataset_summary", {}))
    datasets = present_order(desc.keys(), DATASET_ORDER)
    rows, recs = [], []
    for ds in datasets:
        d, u = desc[ds], used.get(ds, {})
        sc = u.get("split_counts") or {}
        rows.append([ds_label(ds), f"{d['n_rows']:,}", str(d["n_native_columns"]), str(d["native_label_classes"]), f"{d['attack_share']:.3f}",
                     f"{u.get('n_samples', 0):,}" if u else "--",
                     "--" if "canonical_duplicate_rate" not in d or d["canonical_duplicate_rate"] is None else f"{float(d['canonical_duplicate_rate']):.3f}"])
        recs.append({"dataset": ds, **{k: v for k, v in d.items() if not isinstance(v, (dict, list))}, "used_n_samples": u.get("n_samples"),
                     "used_split_counts": json.dumps(sc), "used_n_features": u.get("n_features"),
                     "used_n_duplicates_dropped": u.get("n_duplicates_dropped")})
    tex = tex_table(["Corpus", "Flows", "Cols", "Classes", "Attack share", "Sampled", "Dup. rate"], rows, "lrccccc",
                    caption="The five corpora as read from the files (all days/files pooled), and the number of flows sampled per run.",
                    label="tab:datasets",
                    note=r"Counts come from \texttt{scripts/describe\_datasets.py}. Dup. rate = share of rows whose eight harmonized fields "
                         r"duplicate an earlier row (exact duplicates are removed BEFORE splitting). Sampled = flows drawn per run "
                         r"(stratified by attack category, then split 70/15/15).", banner_text=banner)
    return [Table("datasets", pd.DataFrame(recs), tex)]
