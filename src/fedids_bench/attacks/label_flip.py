from typing import Tuple, Optional
import numpy as np
from fedids_bench.attacks.base import BaseAttack

class LabelFlipAttack(BaseAttack):
    """
    Label flipping attack.
    Flips labels of malicious clients to confuse the global model.
    """
    def __init__(self, target_label: Optional[int] = None, n_classes: int = 4):
        self.target_label = target_label
        self.n_classes = n_classes

    def apply_data_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
        seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_poisoned = y.copy()
        if self.target_label is not None:
            # Map all labels to target_label
            y_poisoned[:] = self.target_label
        else:
            # Shift labels cyclically: (label + 1) % n_classes
            y_poisoned = (y_poisoned + 1) % self.n_classes
        return X.copy(), y_poisoned
