import os
import sys
import random
import platform
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)


def derive_seed(*parts: int) -> int:
    """Independent 32-bit seed for a tuple of integers, e.g. ``derive_seed(run_seed, client_id, round_num)``.

    The former ``seed + 10000 * client + round`` arithmetic collides across runs: (seed 0, round 1) and (seed 1, round 0) of the
    same client shared one random stream, which silently correlates the "independent" seeds behind every confidence interval.
    ``SeedSequence`` mixes the integers, so two different tuples give unrelated streams.
    """
    entropy = [int(p) & 0xFFFFFFFFFFFFFFFF for p in parts]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def get_environment_info() -> Dict[str, Any]:
    """Capture runtime environment metadata for reproducibility."""
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        pass

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "CPU",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "git_commit_hash": git_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }
