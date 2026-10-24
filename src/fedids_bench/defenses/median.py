from typing import List, Dict
import torch
from fedids_bench.defenses.base import BaseDefense

class MedianDefense(BaseDefense):
    """
    Coordinate-wise Median defense.
    Computes coordinate-wise median across client parameter tensors.
    """
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

        aggregated = {}
        first_keys = client_updates[0].keys()

        for name in first_keys:
            stacked = torch.stack([update[name] for update in client_updates], dim=0)
            # torch.median returns the LOWER of the two middle values for an even count, a systematic downward bias with the
            # usual 10 clients; the textbook coordinate-wise median averages them.
            ordered, _ = torch.sort(stacked, dim=0)
            mid = num_clients // 2
            aggregated[name] = ordered[mid] if num_clients % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])

        return aggregated
