"""
Strict zero-shot cross-dataset transfer.

For every (source, seed): train ONE federated model on the SOURCE corpus, then evaluate it on the held-out test
split of EVERY corpus (the diagonal is the in-domain score of the very same model).  Two rescalings are stored:

* ``strict``  - the target's raw harmonized features are standardized with the SOURCE train statistics
                (no target statistic is touched: genuine zero-shot),
* ``target``  - the target keeps its own train-split scaler (the legacy convention, kept only for comparison).

One compact JSON per (source, seed) is written to ``<out_dir>/transfer__<source>__s<seed>.json`` and skipped on
re-runs.  ``aggregate.load_transfer`` turns the files into the paper's transfer matrix.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fedids_bench.config import ExperimentConfig, deep_merge

TRANSFER_METRICS = (
    "balanced_accuracy", "mcc", "auroc", "f1", "attack_recall", "benign_recall",
    "false_positive_rate", "false_negative_rate",
)
MODES = ("strict", "target_scaler", "strict_shuffled")       # the three evaluations stored for every (source, target)
SHUFFLE_SEED_OFFSET = 100_003                                   # control permutation = SHUFFLE_SEED_OFFSET + seed


def transfer_run_id(source: str, seed: int, grid: str = "transfer") -> str:
    return f"{grid}__{source}__s{seed}"


def _pick(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in TRANSFER_METRICS:
        v = metrics.get(k)
        out[k] = None if v is None else float(v)
    cm = metrics.get("confusion_matrix") or {}
    total = sum(cm.values()) if cm else 0
    # share of test flows the model calls 'attack': next to the true prevalence it shows a calibration shift
    out["pred_attack_rate"] = (cm["tp"] + cm["fp"]) / total if total else None
    return out


def build_transfer_config(spec: Dict[str, Any], source: str, seed: int, tuned: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Config (as a dict) of the source model: base < tuned[source][group] < variant overrides < (dataset, seed)."""
    ov = deep_merge({}, spec.get("base", {}))
    group = spec.get("group")
    if tuned and group:
        ov = deep_merge(ov, tuned.get(source, {}).get(group, {}))
    ov = deep_merge(ov, spec.get("overrides", {}))
    ov = deep_merge(ov, {"dataset": {"name": source}, "seed": int(seed)})
    ov["experiment_id"] = transfer_run_id(source, seed, spec.get("name", "transfer"))
    ov.setdefault("output_dir", os.path.join("results", "_scratch", ov["experiment_id"]))
    return ov


def run_transfer_seed(spec: Dict[str, Any], seed: int, out_dir: str, tuned: Optional[Dict[str, Any]] = None,
                      force: bool = False, revision: Optional[str] = None) -> List[Tuple[str, str]]:
    """All sources for one seed.  Every corpus is loaded once and reused as source and as target.

    A stored result is reused only when it was produced with the settings requested now (``grid.is_current``)."""
    from fedids_bench.data.loaders import get_dataset_loader
    from fedids_bench.evaluation.cross_dataset import evaluate_cross_dataset
    from fedids_bench.experiments.grid import is_current
    from fedids_bench.experiments.runner import run_experiment

    grid = spec.get("name", "transfer")
    datasets: Sequence[str] = spec["datasets"]
    os.makedirs(out_dir, exist_ok=True)

    todo = [s for s in datasets if force or not is_current(os.path.join(out_dir, transfer_run_id(s, seed, grid) + ".json"),
                                                          build_transfer_config(spec, s, seed, tuned))]
    status: List[Tuple[str, str]] = [(transfer_run_id(s, seed, grid), "cached") for s in datasets if s not in todo]
    if not todo:
        return status

    loaded: Dict[str, Any] = {}
    for name in datasets:                                    # every corpus is a potential target
        cfg = ExperimentConfig(**build_transfer_config(spec, name, seed, tuned))
        loaded[name] = get_dataset_loader(name).load(cfg.dataset, seed=cfg.seed)

    for source in todo:
        t0 = time.perf_counter()
        cfg_dict = build_transfer_config(spec, source, seed, tuned)
        cfg = ExperimentConfig(**cfg_dict)
        res, art = run_experiment(cfg, save=False, return_artifacts=True, dataset=loaded[source])
        model, src_ds = art["model"], art["dataset"]
        targets: Dict[str, Any] = {}
        for tgt in datasets:
            tds = loaded[tgt]
            strict = evaluate_cross_dataset(model, tds, source_manifest=src_ds.manifest, scaler="source", split=2)
            legacy = evaluate_cross_dataset(model, tds, scaler="target", split=2)
            shuffled = evaluate_cross_dataset(model, tds, source_manifest=src_ds.manifest, scaler="source", split=2,
                                              permute_columns_seed=SHUFFLE_SEED_OFFSET + int(seed))
            targets[tgt] = {
                "n_test": strict["n_samples"],
                "attack_prevalence_test": (tds.manifest.get("attack_prevalence") or {}).get("test"),
                "strict": _pick(strict["metrics"]),
                "target_scaler": _pick(legacy["metrics"]),
                "strict_shuffled": _pick(shuffled["metrics"]),
            }
        payload = {
            "run_id": transfer_run_id(source, seed, grid), "grid": grid, "source": source, "seed": int(seed),
            "overrides": cfg_dict, "wall_time_sec": round(time.perf_counter() - t0, 3), "code_revision": revision,
            "in_domain_federated": _pick(res.get("metrics", {})),
            "feature_names": src_ds.manifest.get("feature_names"),
            "targets": targets,
        }
        path = os.path.join(out_dir, payload["run_id"] + ".json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
        status.append((payload["run_id"], "ok"))
    return status


def _safe_seed(spec: Dict[str, Any], seed: int, out_dir: str, tuned: Optional[Dict[str, Any]], force: bool,
               revision: Optional[str] = None):
    try:
        return run_transfer_seed(spec, seed, out_dir, tuned, force, revision), None
    except Exception:
        err = traceback.format_exc()
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{spec.get('name', 'transfer')}__s{seed}.error.txt"), "w", encoding="utf-8") as fh:
            fh.write(err)
        return [(f"{spec.get('name', 'transfer')}__s{seed}", "error")], err.strip().splitlines()[-1]


def _init_worker() -> None:
    import torch
    torch.set_num_threads(1)


def execute_transfer(spec: Dict[str, Any], out_dir: str, tuned: Optional[Dict[str, Any]] = None, n_jobs: int = 1,
                     force: bool = False, verbose: bool = True) -> Dict[str, int]:
    from fedids_bench.experiments.cli_utils import git_revision
    from fedids_bench.experiments.grid import _REPO_DIR
    revision = git_revision(_REPO_DIR)                            # once, in the parent process
    counts = {"ok": 0, "cached": 0, "error": 0}
    seeds = [int(s) for s in spec["seeds"]]

    def _report(items: List[Tuple[str, str]], err: Optional[str]) -> None:
        for rid, st in items:
            counts[st] += 1
            if verbose and st != "cached":
                print(f"{st:6s} {rid}" + (f"  <- {err}" if err else ""), flush=True)

    if n_jobs <= 1 or len(seeds) == 1:
        for s in seeds:
            _report(*_safe_seed(spec, s, out_dir, tuned, force, revision))
    else:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(seeds)), initializer=_init_worker) as ex:
            futs = [ex.submit(_safe_seed, spec, s, out_dir, tuned, force, revision) for s in seeds]
            for fut in as_completed(futs):
                _report(*fut.result())
    if verbose:
        print(f"transfer finished -> {counts}", flush=True)
    return counts
