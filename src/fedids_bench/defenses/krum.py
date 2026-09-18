from typing import List, Dict
import torch
from fedids_bench.defenses.base import BaseDefense
from fedids_bench.federated.algorithms.fedavg import fedavg_aggregate

class KrumDefense(BaseDefense):
    """
    Krum / Multi-Krum Byzantine Robust Aggregation (Blanchard et al., 2017).
    Selects updates closest to their N - f - 2 neighbors in Euclidean space.
    """
    def __init__(self, f: int = 1, multi: bool = False, m: int = 1):
        self.f = f
        self.multi = multi
        self.m = m

    def _flatten_update(self, update: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([tensor.flatten() for tensor in update.values()])

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

        if num_clients <= 2:
            return fedavg_aggregate(client_updates, sample_counts, trust_client_sample_counts)

        # Flatten updates into vectors
        flat_vectors = [self._flatten_update(u) for u in client_updates]
        
        # Distance matrix
        dist_matrix = torch.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                d = torch.sum((flat_vectors[i] - flat_vectors[j]) ** 2).item()
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Calculate Krum score for each client
        # Keep N - f - 2 smallest distances
        k_neighbors = max(1, num_clients - self.f - 2)
        scores = []
        for i in range(num_clients):
            sorted_dists, _ = torch.sort(dist_matrix[i])
            # Exclude distance to self (which is 0 at index 0)
            score = torch.sum(sorted_dists[1 : 1 + k_neighbors]).item()
            scores.append(score)

        if not self.multi or self.m <= 1:
            best_idx = int(torch.argmin(torch.tensor(scores)).item())
            return client_updates[best_idx]
        else:
            # Multi-Krum: take average of top m clients with lowest scores
            m_select = min(self.m, num_clients)
            top_m_indices = torch.topk(torch.tensor(scores), k=m_select, largest=False).indices.tolist()
            selected_updates = [client_updates[idx] for idx in top_m_indices]
            selected_counts = [sample_counts[idx] for idx in top_m_indices]
            return fedavg_aggregate(selected_updates, selected_counts, trust_client_sample_counts)
