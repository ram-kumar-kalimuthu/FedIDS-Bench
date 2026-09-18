from typing import List, Dict
import torch

def fedavg_aggregate(
    client_updates: List[Dict[str, torch.Tensor]],
    sample_counts: List[int],
    trust_client_sample_counts: bool = True
) -> Dict[str, torch.Tensor]:
    """
    McMahan et al. (2017) weighted FedAvg aggregation.
    
    Formula:
        w_{t+1} = sum_{k=1}^K (w_k * update_k)
        where w_k = n_k / N_total if trust_client_sample_counts=True,
        else w_k = 1 / K (equal weighting defense against sample-count reporting attacks).
    """
    if not client_updates:
        raise ValueError("client_updates list cannot be empty.")
    if len(client_updates) != len(sample_counts):
        raise ValueError(
            f"Length mismatch: {len(client_updates)} updates vs {len(sample_counts)} sample counts."
        )

    num_clients = len(client_updates)
    total_samples = sum(sample_counts)

    if trust_client_sample_counts and total_samples <= 0:
        raise ValueError(f"Total sample count across clients must be > 0, got {total_samples}.")

    aggregated_state_dict: Dict[str, torch.Tensor] = {}
    first_update = client_updates[0]

    for key in first_update.keys():
        weighted_tensor = torch.zeros_like(first_update[key], dtype=torch.float32)
        for update, n_k in zip(client_updates, sample_counts):
            if trust_client_sample_counts:
                weight = float(n_k) / float(total_samples)
            else:
                weight = 1.0 / float(num_clients)
            weighted_tensor += update[key].to(torch.float32) * weight

        aggregated_state_dict[key] = weighted_tensor.to(first_update[key].dtype)

    return aggregated_state_dict
