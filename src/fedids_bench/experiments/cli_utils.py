"""Small helpers shared by the command-line scripts."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import yaml


def csv_list(value: Optional[str], cast=str) -> Optional[List[Any]]:
    """'a,b,c' -> ['a','b','c'] (None stays None)."""
    if value is None:
        return None
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def parse_overrides(items: Sequence[str]) -> Dict[str, Any]:
    """['dataset.n_samples=2000', 'federated.rounds=5'] -> {'dataset': {'n_samples': 2000}, 'federated': {'rounds': 5}}.

    Values are parsed as YAML, so ``model.hidden_dims=[64,32]`` and ``attack.type=none`` work."""
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got '{item}'")
        key, raw = item.split("=", 1)
        cur = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(raw)
    return out


def default_n_jobs() -> int:
    """Half of the logical CPUs (each worker is single-threaded torch), at least 1."""
    return max(1, (os.cpu_count() or 2) // 2)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


CODE_PATHS = ("src", "scripts", "configs")      # a change here can change a result; results/, data/, paper/, docs/ cannot


def git_revision(repo_dir: str, code_paths: Sequence[str] = CODE_PATHS) -> Optional[str]:
    """``<commit>`` or ``<commit>+dirty`` when code that can influence a result has uncommitted changes.

    Result files written by a running grid do not make the tree dirty (only ``code_paths`` are inspected), so every run of one
    invocation records the same revision."""
    try:
        rev = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain", "--", *code_paths],
                               capture_output=True, text=True, timeout=30)
        if rev.returncode != 0:
            return None
        return rev.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def write_provenance(out_dir: str, name: str, payload: Dict[str, Any], repo_dir: Optional[str] = None) -> str:
    """Write ``<out_dir>/_provenance_<name>.json`` (what was run, with which code, on which machine)."""
    import numpy
    import pandas
    import sklearn
    import torch

    os.makedirs(out_dir, exist_ok=True)
    info = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": git_revision(repo_dir) if repo_dir else None,
        "python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "torch": torch.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__,
        "argv": sys.argv,
    }
    info.update(payload)
    path = os.path.join(out_dir, f"_provenance_{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=str)
    return path
