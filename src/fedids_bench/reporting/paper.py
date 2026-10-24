"""
Shared helpers for the paper report (``scripts/make_report.py``): labels, completeness / provenance checks,
statistics on top of ``experiments.aggregate`` and a small LaTeX table writer.

Rules this module enforces on purpose:

* a table or figure is only built from run files that exist on disk (nothing is typed by hand);
* a grid whose run directory does not contain EVERY run its spec declares is refused (``--allow-partial`` overrides, and the
  report then says so on every artifact);
* runs made with command-line overrides (``--set``, narrowed ``--datasets`` / ``--seeds``) are labeled REHEARSAL.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from fedids_bench.experiments import grid as gridlib
from fedids_bench.experiments.aggregate import fmt_pm, summarize, t_ci95

DATASET_LABELS = {
    "unsw_nb15": "UNSW-NB15", "cicids2017": "CICIDS2017", "cse_cic_ids2018": "CSE-CIC-IDS2018",
    "iot23": "IoT-23", "ton_iot": "ToN-IoT",
}
DATASET_SHORT = {
    "unsw_nb15": "UNSW", "cicids2017": "CIC17", "cse_cic_ids2018": "CSE18", "iot23": "IoT23", "ton_iot": "ToN",
}
ALGORITHM_LABELS = {"centralized": "Central.", "fedavg": "FedAvg", "fedprox": "FedProx", "fedadam": "FedAdam"}
PARTITION_LABELS = {
    "iid": "IID", "dir0.5": r"Dir(0.5)", "dir0.1": r"Dir(0.1)", "quantity": "Qty-skew", "pareto1.5": r"Pareto(1.5)",
    "domain2": "Domain-2", "domain1": "Domain-1", "none": "--",
}
ATTACK_LABELS = {
    "clean": "None", "labelflip": "Label flip", "poison5": r"Poison ($\gamma{=}5$)", "byz1": r"Byzantine ($\sigma{=}1$)",
    "backdoor": "Backdoor", "inflate": "Inflate+poison",
}
DEFENSE_LABELS = {
    "fedavg": "FedAvg", "uniform": "Equal-w.", "clipping": "Clip", "trimmed": "Trim", "median": "Median", "krum": "Krum",
}
PAYLOAD_LABELS = {"honest": "Honest update", "labelflip": "+ label flip", "poison5": r"+ poison ($\gamma{=}5$)"}
WEIGHT_LABELS = {"weighted": "Count-weighted", "uniform": "Equal-weight"}

PAPER_GRIDS = ("main", "partitioning", "robustness", "inflation", "sweeps", "transfer")


# ------------------------------------------------------------------------------------------------ completeness / provenance
@dataclass
class GridStatus:
    name: str
    run_dir: str
    n_expected: int = 0
    n_found: int = 0
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)       # reasons the grid is not paper-grade
    warnings: List[str] = field(default_factory=list)    # worth knowing, does not stamp the tables (e.g. uncommitted code)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.missing and not self.errors

    @property
    def paper_grade(self) -> bool:
        return self.complete and not self.flags

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "run_dir": self.run_dir, "n_expected": self.n_expected, "n_found": self.n_found,
                "n_missing": len(self.missing), "missing_examples": self.missing[:10], "n_extra": len(self.extra),
                "errors": self.errors[:10], "flags": self.flags, "warnings": self.warnings, "paper_grade": self.paper_grade,
                "git_revision": self.provenance.get("git_revision"), "written_utc": self.provenance.get("written_utc"),
                "tuned_sha256": self.provenance.get("tuned_sha256")}


def _is_transfer_spec(raw: Dict[str, Any]) -> bool:
    return "variants" not in raw and "matrix" not in raw and "matrices" not in raw


def expected_run_ids(spec_path: str) -> List[str]:
    """Run ids the canonical spec declares (full datasets x variants x seeds), independent of tuned values."""
    import yaml
    with open(spec_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if _is_transfer_spec(raw):
        return [f"{raw['name']}__{ds}__s{int(s)}" for ds in raw["datasets"] for s in raw["seeds"]]
    spec = gridlib.load_spec(spec_path)
    return [j["run_id"] for j in gridlib.expand_spec(spec, None)]


def spec_needs_tuned(spec_path: str) -> bool:
    import yaml
    with open(spec_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if _is_transfer_spec(raw):
        return bool(raw.get("group"))
    return any(v.get("group") for v in gridlib.load_spec(spec_path)["variants"])


def expected_settings(spec_path: str, tuned: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """run id -> the full settings the canonical spec (plus the tuned file) asks for."""
    import yaml
    with open(spec_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if _is_transfer_spec(raw):
        from fedids_bench.experiments.transfer import build_transfer_config
        out: Dict[str, Dict[str, Any]] = {}
        for ds in raw["datasets"]:
            for s in raw["seeds"]:
                cfg = build_transfer_config(raw, ds, int(s), tuned)
                out[cfg["experiment_id"]] = cfg
        return out
    return {j["run_id"]: j["overrides"] for j in gridlib.expand_spec(gridlib.load_spec(spec_path), tuned)}


def read_provenance(run_dir: str, name: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, f"_provenance_{name}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_tuned_file(*candidates: Optional[str]) -> Optional[Dict[str, Any]]:
    for c in candidates:
        if c and os.path.exists(c):
            with open(c, "r", encoding="utf-8") as fh:
                return json.load(fh)
    return None


def grid_status(name: str, run_dir: str, spec_path: str, tuned_path: Optional[str] = None) -> GridStatus:
    """Completeness AND integrity of one run folder.

    Every run file carries the exact settings it was produced with; they are compared with the settings the canonical spec
    (plus the tuned-hyper-parameter file) asks for.  A run made with ``--set`` overrides, left over from a rehearsal, or
    made for another seed list therefore cannot pass as a paper run, whatever the last invocation's provenance says."""
    st = GridStatus(name=name, run_dir=run_dir)
    prov = read_provenance(run_dir, name)
    st.provenance = prov
    expected_ids = set(expected_run_ids(spec_path))
    present = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(run_dir, "*.json"))
               if not os.path.basename(p).startswith("_")}
    st.n_expected, st.n_found = len(expected_ids), len(expected_ids & present)
    st.missing = sorted(expected_ids - present)
    st.extra = sorted(present - expected_ids)
    st.errors = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*.error.txt")))
    if not prov:
        st.warnings.append("no provenance file next to the runs (folder copied without _provenance_*.json?)")

    tuned = _load_tuned_file(tuned_path, prov.get("tuned_path"))
    ignore: Tuple[Tuple[str, ...], ...] = ()
    if spec_needs_tuned(spec_path) and tuned is None:
        from fedids_bench.experiments.tuning import TUNABLE_PATHS
        ignore = TUNABLE_PATHS
        st.flags.append("tuned hyper-parameters not verified: results/tuned_hparams.json not found (grid run with --no-tuned, or file missing)")
    expected = expected_settings(spec_path, tuned)

    mismatched: List[Tuple[str, str]] = []
    revisions: Dict[str, int] = {}
    for rid in sorted(expected_ids & present):
        try:
            with open(os.path.join(run_dir, rid + ".json"), "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            mismatched.append((rid, "unreadable run file"))
            continue
        why = gridlib.stored_settings_differ(payload, expected[rid], ignore)
        if why:
            mismatched.append((rid, why))
        if payload.get("code_revision"):
            revisions[payload["code_revision"]] = revisions.get(payload["code_revision"], 0) + 1
    if mismatched:
        rid, why = mismatched[0]
        st.flags.append(f"settings differ from the spec in {len(mismatched)} run file(s), e.g. {rid}: {why}")
    if st.extra:
        st.flags.append(f"{len(st.extra)} run file(s) not declared by the spec would enter the statistics, e.g. {st.extra[0]}")
    if len(revisions) > 1:
        st.warnings.append(f"runs were produced by {len(revisions)} different code revisions: "
                           + ", ".join(f"{r[:10]}x{n}" for r, n in sorted(revisions.items())))
    if any(r.endswith("+dirty") for r in revisions):
        st.warnings.append("some runs were made from a working tree with uncommitted code changes "
                           "(commit before the paper runs so every run records a clean revision)")
    return st


def banner(statuses: Iterable[GridStatus], allow_partial: bool, allow_rehearsal: bool) -> Optional[str]:
    """Short text stamped on every artifact when it is NOT paper-grade; None when everything is.

    The per-grid details (which runs are missing, which overrides were used) are in ``completeness.json`` and on the console."""
    statuses = list(statuses)
    parts: List[str] = []
    incomplete = [f"{s.name} {s.n_found}/{s.n_expected}" for s in statuses if not s.complete]
    if incomplete:
        parts.append("incomplete grids (" + ", ".join(incomplete) + ")")
    rehearsal = sorted({s.name for s in statuses for f in s.flags if f.startswith(("settings differ", "tuned hyper-parameters not verified"))})
    if rehearsal:
        parts.append("settings differ from the spec (rehearsal or stale runs) in " + ", ".join(rehearsal))
    extra = sorted({s.name for s in statuses for f in s.flags if "not declared by the spec" in f})
    if extra:
        parts.append("undeclared extra runs in " + ", ".join(extra))
    return None if not parts else "NOT PAPER-GRADE: " + "; ".join(parts)


# ------------------------------------------------------------------------------------------------ statistics
def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs stay NaN and are ignored in the family size)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        running = max(running, adj)
        out[idx] = running
    return out


def numbers_tex(statuses: Sequence[GridStatus], frames: Dict[str, pd.DataFrame]) -> str:
    """LaTeX macros for the few counts the manuscript quotes (``\\nRunsTotal`` ...), so no count is ever typed by hand.

    Written next to the tables as ``numbers.tex``; ``paper/macros.tex`` inputs it when it exists.  Macro names contain letters only.
    """
    def camel(name: str) -> str:
        return "".join(w.capitalize() for w in re.split(r"[^A-Za-z]+", name) if w)

    lines = ["% generated by scripts/make_report.py -- do not edit"]
    total = 0
    for st in statuses:
        lines.append("\\newcommand{\\nRuns%s}{%d}" % (camel(st.name), st.n_found))
        total += st.n_found
    lines.append("\\newcommand{\\nRunsTotal}{%d}" % total)
    main = frames.get("main")
    if main is not None and len(main) and {"dataset", "seed"} <= set(main.columns):
        lines.append("\\newcommand{\\nCorpora}{%d}" % main["dataset"].nunique())
        lines.append("\\newcommand{\\nSeeds}{%d}" % main["seed"].nunique())
    return "\n".join(lines) + "\n"


def macro_by_seed(df: pd.DataFrame, metric: str, group_cols: Sequence[str]) -> pd.DataFrame:
    """Per (group, seed) mean over datasets; groups missing any dataset for a seed are dropped (never silently imputed)."""
    n_ds = df["dataset"].nunique()
    g = df.groupby(list(group_cols) + ["seed"], as_index=False).agg(value=(metric, "mean"), n=("dataset", "nunique"))
    g = g[g["n"] == n_ds].drop(columns="n").rename(columns={"value": metric})
    return g


def summarize_macro(df: pd.DataFrame, metric: str, group_cols: Sequence[str]) -> pd.DataFrame:
    """Macro-average over datasets per seed, then mean +/- 95% CI over seeds."""
    m = macro_by_seed(df, metric, group_cols)
    m["dataset"] = "macro"
    return summarize(m, [metric], by=list(group_cols))


def paired_diff(df: pd.DataFrame, factor: str, a: str, b: str, metric: str, index_cols: Sequence[str],
                group_cols: Sequence[str]) -> pd.DataFrame:
    """Paired (over seeds) difference  metric[a] - metric[b]  for the two levels of ``factor``.

    ``index_cols`` identify a pair (e.g. dataset, seed + the other factors); ``group_cols`` are what the result is reported by.
    """
    from scipy import stats
    wide = df.pivot_table(index=list(index_cols), columns=factor, values=metric, aggfunc="mean")
    if a not in wide.columns or b not in wide.columns:
        return pd.DataFrame()
    d = (wide[a] - wide[b]).dropna().rename("diff").reset_index()
    rows: List[Dict[str, Any]] = []
    for key, g in d.groupby(list(group_cols), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        v = g["diff"].to_numpy(float)
        n = len(v)
        std = float(v.std(ddof=1)) if n > 1 else float("nan")
        p = float("nan")
        if n > 1 and std > 0:
            p = float(stats.ttest_1samp(v, 0.0).pvalue)
        rows.append({**dict(zip(group_cols, key)), "n": n, "mean_diff": float(v.mean()), "ci95": t_ci95(std, n), "p_value": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_value"].to_numpy())
    return out


# ------------------------------------------------------------------------------------------------ LaTeX
_TEX_ESC = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}


def tex_escape(s: Any) -> str:
    return "".join(_TEX_ESC.get(c, c) for c in str(s))


def tex_num(mean: float, ci: float, digits: int = 3, bold: bool = False, flag: str = "", strip_zero: bool = True) -> str:
    """``.912$\\pm$.006``-style cell (leading zeros dropped, as is usual for bounded scores)."""
    s = fmt_pm(mean, ci, digits)
    if s == "--":
        return s
    if strip_zero:
        s = re.sub(r"^(-?)0\.", r"\1.", s)
    s = s.replace("±", r"$\pm$")
    if bold:
        s = r"\textbf{" + s + "}"
    return s + flag


def tex_table(header: Sequence[str], rows: Sequence[Sequence[str]], colspec: str, caption: str, label: str, note: str = "",
              banner_text: Optional[str] = None, font: str = r"\small", tabcolsep_pt: float = 3.0,
              style: str = "booktabs", midrules_after: Iterable[int] = ()) -> str:
    """One LNCS-style floating table: caption ABOVE the table, ``\\small`` body text; wrapped in ``\\fittable`` (paper/macros.tex),
    which shrinks a table only if it would overflow the text block and fails the FINAL build if that costs more than 10 %.

    ``rows`` hold already-escaped cell strings; a row equal to ``["\\midrule"]`` inserts a rule.  ``midrules_after`` lists row
    indexes after which a thin rule is drawn.
    """
    top, mid, bot = (r"\toprule", r"\midrule", r"\bottomrule") if style == "booktabs" else (r"\hline", r"\hline", r"\hline")
    after = set(midrules_after)
    lines = [r"\begin{table}[t]", r"\caption{" + caption + r"}", r"\label{" + label + "}", r"\centering", font,
             rf"\setlength{{\tabcolsep}}{{{tabcolsep_pt:g}pt}}", r"\fittable{\begin{tabular}{" + colspec + "}", top,
             " & ".join(header) + r" \\", mid]
    for i, r in enumerate(rows):
        lines.append(" & ".join(r) + r" \\")
        if i in after:
            lines.append(mid if style == "booktabs" else r"\hline")
    lines += [bot, r"\end{tabular}}"]
    if note:
        lines += [r"\par\smallskip", r"{\footnotesize " + note + "}"]
    if banner_text:
        lines += [r"\par{\footnotesize\textbf{" + tex_escape(banner_text) + "}}"]
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_artifact(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def ds_label(name: str, short: bool = False) -> str:
    return (DATASET_SHORT if short else DATASET_LABELS).get(name, name)


def present_order(values: Iterable[str], preferred: Sequence[str]) -> List[str]:
    vals = list(dict.fromkeys(values))
    return [p for p in preferred if p in vals] + [v for v in vals if v not in preferred]
