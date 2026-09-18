from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import torch

class BaseDefense(ABC):
    """Abstract base class for robust aggregation defenses in FedIDS-Bench."""

    @abstractmethod
    def aggregate(
        self,
        global_weights: Dict[str, torch.Tensor],
        client_updates: List[Dict[str, torch.Tensor]],
        sample_counts: List[int],
        trust_client_sample_counts: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates securely under potential attacks.
        Returns the new global model state_dict.
        """
        pass
