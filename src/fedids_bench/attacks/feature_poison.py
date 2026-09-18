from typing import Tuple
import numpy as np
from fedids_bench.attacks.base import BaseAttack

class FeaturePoisonAttack(BaseAttack):
    """
    Feature poisoning attack.
    Injects Gaussian noise or extreme perturbations into client input features.
    """
    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def apply_data_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
        seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed + client_id)
        noise = rng.normal(loc=0.0, scale=self.strength, size=X.shape)
        X_poisoned = X + noise
        return X_poisoned.astype(np.float32), y.copy()
