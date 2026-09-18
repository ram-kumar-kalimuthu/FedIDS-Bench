from typing import List, Dict
import torch
from fedids_bench.defenses.base import BaseDefense
from fedids_bench.federated.algorithms.fedavg import fedavg_aggregate

class NormClippingDefense(BaseDefense):
    """
    Gradient / Update Norm Clipping defense.
    Clips local update vectors delta = w_k - w_global to a maximum L2 norm threshold.
    """
    def __init__(self, clip_threshold: float = 5.0):
        self.clip_threshold = clip_threshold

    def aggregate(
        self,
        global_weights: Dict[str, torch.Tensor],
        client_updates: List[Dict[str, torch.Tensor]],
        sample_counts: List[int],
        trust_client_sample_counts: bool = True
    ) -> Dict[str, torch.Tensor]:
        clipped_updates = []
        for update in client_updates:
            # Compute total L2 norm of update delta across all parameters
            total_norm_sq = 0.0
            deltas = {}
            for name, tensor in update.items():
                g_tensor = global_weights[name]
                delta = tensor - g_tensor
                deltas[name] = delta
                total_norm_sq += torch.sum(delta ** 2).item()
            
            l2_norm = total_norm_sq ** 0.5
            clip_coef = min(1.0, self.clip_threshold / (l2_norm + 1e-8))

            clipped_update = {}
            for name, delta in deltas.items():
                clipped_update[name] = global_weights[name] + delta * clip_coef
            clipped_updates.append(clipped_update)

        return fedavg_aggregate(clipped_updates, sample_counts, trust_client_sample_counts)
