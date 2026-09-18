from typing import Tuple, Optional
import numpy as np
from fedids_bench.attacks.base import BaseAttack

class BackdoorAttack(BaseAttack):
    """
    Backdoor (Watermark Trigger) attack.
    Injects a watermark trigger into a fraction of client samples and changes their target label to target_label.
    """
    def __init__(
        self,
        target_label: int = 0,
        trigger_fraction: float = 0.2,
        trigger_val: float = 5.0,
        trigger_dims: int = 2
    ):
        self.target_label = target_label
        self.trigger_fraction = trigger_fraction
        self.trigger_val = trigger_val
        self.trigger_dims = trigger_dims

    def inject_trigger(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_triggered = X.copy()
        y_triggered = y.copy()
        n = len(X)
        num_poisoned = int(n * self.trigger_fraction)
        if num_poisoned > 0:
            dims = min(self.trigger_dims, X.shape[1])
            X_triggered[:num_poisoned, :dims] = self.trigger_val
            y_triggered[:num_poisoned] = self.target_label
        return X_triggered, y_triggered

    def apply_data_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
        seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.inject_trigger(X, y)
