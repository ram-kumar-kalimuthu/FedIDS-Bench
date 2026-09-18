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
