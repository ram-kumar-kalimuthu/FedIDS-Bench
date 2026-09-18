from abc import ABC, abstractmethod
from typing import Tuple, Dict
import numpy as np
import torch

class BaseAttack(ABC):
    """Abstract base class for all client attacks in FedIDS-Bench."""

    def apply_data_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
        seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data poisoning (label flip, feature noise, backdoor injection) to client data.
        By default returns unchanged data.
        """
        return X, y

    def apply_model_attack(
        self,
        local_weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        client_id: int,
        seed: int
    ) -> Dict[str, torch.Tensor]:
        """
        Apply model poisoning (scaling, sign-flip, byzantine noise) to client update weights.
        By default returns unchanged local weights.
        """
        return local_weights
