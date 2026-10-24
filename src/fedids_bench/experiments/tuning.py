"""
Hyper-parameter selection protocol (used by ``scripts/tune_hparams.py``).

* Search: the SAME number of seeded random trials for every (dataset, algorithm); trial 0 is always the repository
  default so the gain from tuning is visible.  Every algorithm searches the shared knobs (client learning rate, local
  epochs, batch size, class weighting, client momentum); FedProx also searches ``mu`` and FedAdam its server step size
  and ``tau``.  The model architecture is FIXED (it is not an FL-algorithm knob; see ``--search-hidden``).
* Data: tuning runs use the ``tune`` row role (rows that can never appear in a reported test split) and are scored on
  their VALIDATION split; the test split is never read.  Seeds are disjoint from the reporting seeds.
* Selection: highest mean validation balanced accuracy over the tuning seeds (ties: higher MCC).
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

GROUPS = ("fedavg", "fedprox", "fedadam", "centralized")

# trial 0 = the values the repository shipped with (what the first submission effectively used)
DEFAULT_TRIAL: Dict[str, Any] = {
    "model": {"learning_rate": 0.01},
    "federated": {"local_epochs": 1, "batch_size": 32, "class_weight": "none", "client_momentum": 0.0},
}
DEFAULT_EXTRA = {
    "fedprox": {"federated": {"mu": 0.01}},
    "fedadam": {"federated": {"tau": 1e-3}},
}

SPACE = {
    "lr": ("loguniform", 0.003, 0.5),
    "local_epochs": ("choice", [1, 2, 3, 5]),
    "batch_size": ("choice", [32, 64, 128]),
    "class_weight": ("choice", ["none", "balanced"]),
    "client_momentum": ("choice", [0.0, 0.9]),
    "mu": ("choice", [0.001, 0.01, 0.1, 1.0]),
    "server_lr": ("loguniform", 0.003, 0.3),
    "tau": ("choice", [1e-3, 1e-2, 1e-1]),
    "hidden_dims": ("choice", [[32, 16], [64, 32], [128, 64]]),
}


# every setting a tuned-hyper-parameter file can supply (so a report can still verify all OTHER settings without the file)
TUNABLE_PATHS = (
    ("model", "learning_rate"), ("model", "hidden_dims"),
    ("federated", "local_epochs"), ("federated", "batch_size"), ("federated", "class_weight"),
    ("federated", "client_momentum"), ("federated", "mu"), ("federated", "server_lr"), ("federated", "tau"),
)


def _rng(group: str, seed: int) -> random.Random:
    return random.Random(int(hashlib.sha1(f"{group}:{seed}".encode()).hexdigest()[:8], 16))   # stable across processes


def _draw(rng: random.Random, key: str) -> Any:
    kind, *args = SPACE[key]
    if kind == "loguniform":
        lo, hi = args
        return float(round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 5))
    return rng.choice(args[0])


def sample_trials(group: str, n_trials: int, seed: int = 2026, search_hidden: bool = False) -> List[Dict[str, Any]]:
    """Deterministic list of ``n_trials`` nested override dicts; index 0 is the repository default."""
    if group not in GROUPS:
        raise ValueError(f"unknown group '{group}' (expected one of {GROUPS})")
    base = {"model": dict(DEFAULT_TRIAL["model"]), "federated": dict(DEFAULT_TRIAL["federated"])}
    for k, v in DEFAULT_EXTRA.get(group, {}).get("federated", {}).items():
        base["federated"][k] = v
    trials = [base]
    rng = _rng(group, seed)
    for _ in range(max(0, n_trials - 1)):
        t: Dict[str, Any] = {"model": {"learning_rate": _draw(rng, "lr")},
                             "federated": {"local_epochs": _draw(rng, "local_epochs"), "batch_size": _draw(rng, "batch_size"),
                                           "class_weight": _draw(rng, "class_weight"), "client_momentum": _draw(rng, "client_momentum")}}
        if group == "fedprox":
            t["federated"]["mu"] = _draw(rng, "mu")
        if group == "fedadam":
            t["federated"]["server_lr"] = _draw(rng, "server_lr")
            t["federated"]["tau"] = _draw(rng, "tau")
        if search_hidden:
            t["model"]["hidden_dims"] = _draw(rng, "hidden_dims")
        trials.append(t)
    return trials


def flatten(overrides: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for sec, d in overrides.items():
        for k, v in d.items():
            out[f"{sec}.{k}"] = v if not isinstance(v, list) else "-".join(map(str, v))
    return out


def variant_name(group: str, idx: int) -> str:
    return f"{group}__t{idx:02d}"


def parse_variant(name: str) -> Dict[str, Any]:
    group, t = name.split("__t")
    return {"group": group, "trial": int(t)}


def build_spec(datasets: Sequence[str], groups: Sequence[str], seeds: Sequence[int], n_trials: int, n_samples: int, rounds: int,
               hidden: Sequence[int], data_dir: str, search_hidden: bool = False, space_seed: int = 2026,
               pool: str = "all", num_clients: int = 10) -> Dict[str, Any]:
    """Grid spec (``grid.expand_spec`` format) of every tuning trial."""
    variants: List[Dict[str, Any]] = []
    for g in groups:
        for i, trial in enumerate(sample_trials(g, n_trials, space_seed, search_hidden)):
            ov: Dict[str, Any] = {"model": dict(trial["model"]), "federated": dict(trial["federated"])}
            if g == "centralized":
                ov["mode"] = "centralized"
            else:
                ov["federated"]["algorithm"] = g
            variants.append({"name": variant_name(g, i), "group": None, "overrides": ov})
    return {
        "name": "tuning", "datasets": list(datasets), "seeds": [int(s) for s in seeds], "variants": variants,
        "base": {
            "dataset": {"n_samples": n_samples, "test_fraction": 0.15, "val_fraction": 0.15, "eval_on": "val", "row_role": "tune",
                        "pool": pool, "feature_set": "core", "data_dir": data_dir},
            "partitioning": {"strategy": "dirichlet", "alpha": 0.5, "num_clients": num_clients, "heldout_client_fraction": 0.0},
            "model": {"architecture": "mlp", "hidden_dims": list(hidden)},
            "federated": {"rounds": rounds, "clients_per_round": num_clients},
        },
    }


def select_best(final: pd.DataFrame, n_trials_by_group: Dict[str, int]) -> pd.DataFrame:
    """Mean validation score per (dataset, group, trial) + selection flag.  ``final`` comes from ``aggregate.load_runs``."""
    f = final.copy()
    parsed = f["variant"].map(parse_variant)
    f["group"] = parsed.map(lambda d: d["group"])
    f["trial"] = parsed.map(lambda d: d["trial"])
    g = f.groupby(["dataset", "group", "trial"], as_index=False).agg(
        n_seeds=("seed", "nunique"), val_balanced_accuracy=("balanced_accuracy", "mean"),
        val_balanced_accuracy_std=("balanced_accuracy", "std"), val_mcc=("mcc", "mean"), val_auroc=("auroc", "mean"),
        val_f1=("f1", "mean"))
    g["selected"] = False
    for (ds, grp), sub in g.groupby(["dataset", "group"]):
        order = sub.sort_values(["val_balanced_accuracy", "val_mcc", "trial"], ascending=[False, False, True])
        g.loc[order.index[0], "selected"] = True
    return g.sort_values(["dataset", "group", "trial"]).reset_index(drop=True)
