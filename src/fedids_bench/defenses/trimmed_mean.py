from typing import List, Dict
import torch
from fedids_bench.defenses.base import BaseDefense

class TrimmedMeanDefense(BaseDefense):
    """
    Coordinate-wise Trimmed Mean defense.
    Sorts update coordinates across clients and trims the top and bottom beta fraction.
    """
    def __init__(self, trim_ratio: float = 0.2):
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        global_weights: Dict[str, torch.Tensor],
        client_updates: List[Dict[str, torch.Tensor]],
        sample_counts: List[int],
        trust_client_sample_counts: bool = True
    ) -> Dict[str, torch.Tensor]:
        num_clients = len(client_updates)
        if num_clients == 0:
            return global_weights

        beta = int(num_clients * self.trim_ratio)
        # Ensure we keep at least 1 client
        if 2 * beta >= num_clients:
            beta = max(0, (num_clients - 1) // 2)

        aggregated = {}
        first_keys = client_updates[0].keys()

        for name in first_keys:
            stacked = torch.stack([update[name] for update in client_updates], dim=0)
            sorted_tensors, _ = torch.sort(stacked, dim=0)

            if beta > 0:
                trimmed = sorted_tensors[beta : num_clients - beta]
            else:
                trimmed = sorted_tensors

            aggregated[name] = torch.mean(trimmed, dim=0)

        return aggregated
