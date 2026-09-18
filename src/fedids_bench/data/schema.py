from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class PreparedDataset:
    X: np.ndarray             # float32 [n_samples, n_features]
    y: np.ndarray             # int64 [n_samples] (0=benign, >0=attack)
    attack_type: np.ndarray   # int64 [n_samples] (0=benign, 1=DoS, 2=Probe, etc.)
    split: np.ndarray         # int8 [n_samples] (0=train, 1=val, 2=test)
    manifest: Dict[str, Any]  # Metadata sidecar
    protocol: Optional[np.ndarray] = None  # int64 [n_samples] where available

    def __post_init__(self):
        assert self.X.dtype == np.float32, f"X must be float32, got {self.X.dtype}"
        assert self.y.dtype == np.int64, f"y must be int64, got {self.y.dtype}"
        assert self.attack_type.dtype == np.int64, f"attack_type must be int64, got {self.attack_type.dtype}"
        assert self.split.dtype == np.int8, f"split must be int8, got {self.split.dtype}"
        assert len(self.X) == len(self.y) == len(self.attack_type) == len(self.split), "Length mismatch across dataset components"
