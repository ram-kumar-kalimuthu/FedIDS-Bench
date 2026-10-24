"""
Declarative experiment grids: datasets x variants x seeds.

* every run is cached as one compact JSON (``<out_dir>/<run_id>.json``) and skipped if it already exists,
* runs may be spread over processes (``n_jobs``); results do not depend on execution order,
* hyper-parameters selected on the VALIDATION split can be merged in per (dataset, group) from a tuning file,
* tables and figures are produced from these JSON files only (see ``aggregate.py``).
"""
from __future__ import annotations

import itertools
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fedids_bench.config import deep_merge

# Result keys kept in the compact run file.
RESULT_KEYS = (
    "mode", "algorithm", "dataset", "metrics", "baseline_majority", "backdoor", "eval_split", "class_weights",
    "round_logs", "communication", "resource_cost", "dataset_summary", "partition_manifest",
)


# Settings that only say WHERE things live on this machine; they never change a result and are ignored when a stored run is
# compared with the settings a job asks for.
LOCATION_KEYS = (("dataset", "data_dir"), ("dataset", "cache_dir"), ("dataset", "use_cache"), ("output_dir",))
_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def make_run_id(grid: str, dataset: str, variant: str, seed: int) -> str:
    return f"{grid}__{dataset}__{variant}__s{seed}"


def comparable_overrides(overrides: Dict[str, Any], ignore: Tuple[Tuple[str, ...], ...] = ()) -> Dict[str, Any]:
    """JSON-normalized copy of a run's settings without the machine-specific location keys (and any ``ignore`` paths)."""
    d = json.loads(json.dumps(overrides, default=lambda o: _json_default(o)))
    for path in tuple(LOCATION_KEYS) + tuple(ignore):
        cur: Any = d
        for k in path[:-1]:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, dict):
            cur.pop(path[-1], None)
    return d


def first_difference(stored: Any, wanted: Any, prefix: str = "") -> Optional[str]:
    """Dotted key of the first setting on which two settings dicts differ (None when they agree)."""
    if isinstance(stored, dict) and isinstance(wanted, dict):
        for k in sorted(set(stored) | set(wanted)):
            if k not in stored:
                return f"{prefix}{k}: missing in the stored run (spec wants {wanted[k]!r})"
            if k not in wanted:
                return f"{prefix}{k}: stored run has {stored[k]!r}, spec has no such setting"
            diff = first_difference(stored[k], wanted[k], f"{prefix}{k}.")
            if diff:
                return diff
        return None
    if stored != wanted:
        return f"{prefix.rstrip('.')}: stored {stored!r} != spec {wanted!r}"
    return None


def stored_settings_differ(payload: Dict[str, Any], wanted_overrides: Dict[str, Any],
                           ignore: Tuple[Tuple[str, ...], ...] = ()) -> Optional[str]:
    """Why a stored run file does NOT correspond to the settings a job asks for (None when it does)."""
    stored = payload.get("overrides")
    if not isinstance(stored, dict):
        return "stored run has no settings record"
    return first_difference(comparable_overrides(stored, ignore), comparable_overrides(wanted_overrides, ignore))


def is_current(path: str, wanted_overrides: Dict[str, Any]) -> bool:
    """A cached result file counts only if it was produced with exactly the requested settings.

    This is what stops a 3,000-row rehearsal from being silently reused by a 50,000-row paper run (same run id)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return False
    return stored_settings_differ(payload, wanted_overrides) is None


def expand_matrix(matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cartesian product of named factor levels -> variants.

    ``matrix = {group, prefix, factors: {factor: [{name, overrides, group?}, ...], ...}, exclude: [{factor: level}, ...]}``.
    The variant name is the level names joined by ``__`` (factor order as written, optionally preceded by ``prefix``);
    ``exclude`` drops every combination that matches ALL entries of one exclusion.  A level may carry its own ``group``
    (the tuned-hyper-parameter group, e.g. the algorithm); it overrides the matrix-level ``group``.
    """
    factors = list(matrix["factors"].items())
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*[levels for _, levels in factors]):
        chosen = {fname: lvl["name"] for (fname, _), lvl in zip(factors, combo)}
        if any(all(chosen.get(k) == v for k, v in ex.items()) for ex in matrix.get("exclude", [])):
            continue
        ov: Dict[str, Any] = {}
        group = matrix.get("group")
        for lvl in combo:
            ov = deep_merge(ov, lvl.get("overrides", {}))
            if lvl.get("group"):
                group = lvl["group"]
        name = "__".join(chosen.values())
        if matrix.get("prefix"):
            name = f"{matrix['prefix']}__{name}"
        out.append({"name": name, "group": group, "overrides": ov, "factors": chosen})
    return out


def expand_spec(spec: Dict[str, Any], tuned: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Cartesian product -> list of jobs.  Merge order: base < tuned[dataset][group] < variant < (dataset, seed)."""
    jobs: List[Dict[str, Any]] = []
    for ds in spec["datasets"]:
        for var in spec["variants"]:
            for seed in spec["seeds"]:
                ov = deep_merge({}, spec.get("base", {}))
                group = var.get("group")
                if tuned and group:
                    ov = deep_merge(ov, tuned.get(ds, {}).get(group, {}))
                ov = deep_merge(ov, var.get("overrides", {}))
                ov = deep_merge(ov, {"dataset": {"name": ds}, "seed": int(seed)})
                run_id = make_run_id(spec["name"], ds, var["name"], int(seed))
                ov["experiment_id"] = run_id
                ov.setdefault("output_dir", os.path.join("results", "_scratch", run_id))
                jobs.append({
                    "run_id": run_id, "grid": spec["name"], "dataset": ds, "variant": var["name"],
                    "group": group, "seed": int(seed), "overrides": ov, "factors": dict(var.get("factors", {})),
                })
    return jobs


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_job(job: Dict[str, Any], out_dir: str, force: bool = False) -> Tuple[str, str]:
    """Run one job (cached).  Returns (run_id, status) with status in {'ok', 'cached'}."""
    path = os.path.join(out_dir, job["run_id"] + ".json")
    if os.path.exists(path) and not force and is_current(path, job["overrides"]):
        return job["run_id"], "cached"
    from fedids_bench.config import ExperimentConfig            # local imports keep workers light
    from fedids_bench.experiments.runner import run_experiment

    cfg = ExperimentConfig(**job["overrides"])
    t0 = time.perf_counter()
    res = run_experiment(cfg, save=False)
    payload = {
        "run_id": job["run_id"], "grid": job["grid"], "dataset": job["dataset"], "variant": job["variant"],
        "group": job["group"], "seed": job["seed"], "overrides": job["overrides"], "factors": job.get("factors", {}),
        "wall_time_sec": round(time.perf_counter() - t0, 3), "code_revision": job.get("code_revision"),
        "result": {k: res[k] for k in RESULT_KEYS if k in res},
    }
    os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, default=_json_default)
    os.replace(tmp, path)
    return job["run_id"], "ok"


def _init_worker() -> None:
    import torch
    torch.set_num_threads(1)


def _safe_run(job: Dict[str, Any], out_dir: str, force: bool) -> Tuple[str, str, Optional[str]]:
    try:
        rid, status = run_job(job, out_dir, force)
        return rid, status, None
    except Exception:                                           # keep the grid going; log the traceback
        err = traceback.format_exc()
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, job["run_id"] + ".error.txt"), "w", encoding="utf-8") as fh:
            fh.write(err)
        return job["run_id"], "error", err.strip().splitlines()[-1]


def execute(
    jobs: List[Dict[str, Any]], out_dir: str, n_jobs: int = 1, force: bool = False, limit: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, int]:
    """Run all jobs; returns counts {'ok','cached','error'}."""
    if limit is not None:
        jobs = jobs[:limit]
    from fedids_bench.experiments.cli_utils import git_revision
    revision = git_revision(_REPO_DIR)                          # once, in the parent: every worker stamps the same value
    jobs = [{**j, "code_revision": revision} for j in jobs]
    stale = [j["run_id"] for j in jobs if not force and os.path.exists(os.path.join(out_dir, j["run_id"] + ".json"))
             and not is_current(os.path.join(out_dir, j["run_id"] + ".json"), j["overrides"])]
    if stale and verbose:
        print(f"note: {len(stale)} existing result file(s) in {out_dir} were produced with DIFFERENT settings (e.g. {stale[0]}) "
              f"and will be re-run and replaced.", flush=True)
    counts = {"ok": 0, "cached": 0, "error": 0}
    total = len(jobs)
    done = 0
    t0 = time.perf_counter()

    def _report(rid: str, status: str, err: Optional[str]) -> None:
        nonlocal done
        done += 1
        counts[status] += 1
        if verbose and (status != "cached" or done % 50 == 0 or done == total):
            msg = f"[{done}/{total}] {status:6s} {rid}"
            if err:
                msg += f"  <- {err}"
            print(msg, flush=True)

    if n_jobs <= 1:
        for job in jobs:
            _report(*_safe_run(job, out_dir, force))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker) as ex:
            futs = [ex.submit(_safe_run, job, out_dir, force) for job in jobs]
            for fut in as_completed(futs):
                _report(*fut.result())
    if verbose:
        print(f"finished {total} jobs in {time.perf_counter() - t0:.1f}s -> {counts}", flush=True)
    return counts


def load_spec(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    matrices = ([spec["matrix"]] if "matrix" in spec else []) + list(spec.get("matrices", []))
    for m in matrices:
        spec["variants"] = list(spec.get("variants", [])) + expand_matrix(m)
    for key in ("name", "datasets", "variants", "seeds"):
        if key not in spec:
            raise ValueError(f"Grid spec {path} is missing '{key}'")
    names = [v["name"] for v in spec["variants"]]
    if len(set(names)) != len(names):
        raise ValueError(f"Grid spec {path} has duplicate variant names")
    return spec


def load_tuned(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tuned hyper-parameter file not found: {path} (run scripts/tune_hparams.py first)")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
