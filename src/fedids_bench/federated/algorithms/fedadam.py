from typing import List, Dict
import torch

class FedAdamServerOptimizer:
    """
    FedAdam Server Optimizer (Reddi et al., 2021).
    Applies adaptive server-side momentum and velocity to client update pseudo-gradients.
    """
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, tau: float = 1e-3, lr: float = 0.01):
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.lr = lr
        self.m: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}

    def aggregate(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        client_updates: List[Dict[str, torch.Tensor]],
        sample_counts: List[int],
        trust_client_sample_counts: bool = True
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            raise ValueError("client_updates list cannot be empty.")
            
        num_clients = len(client_updates)
        total_samples = sum(sample_counts)
        first_key = list(global_state_dict.keys())[0]

        # Initialize momentum m and velocity v if empty
        if not self.m:
            for k, v in global_state_dict.items():
                self.m[k] = torch.zeros_like(v, dtype=torch.float32)
                self.v[k] = torch.zeros_like(v, dtype=torch.float32)

        # 1. Compute weighted average pseudo-gradient Delta_t
        delta_t: Dict[str, torch.Tensor] = {}
        for key in global_state_dict.keys():
            delta_key = torch.zeros_like(global_state_dict[key], dtype=torch.float32)
            for update, n_k in zip(client_updates, sample_counts):
                if trust_client_sample_counts and total_samples > 0:
                    weight = float(n_k) / float(total_samples)
                else:
                    weight = 1.0 / float(num_clients)
                
                # Client update delta = w_client - w_global
                client_delta = update[key].to(torch.float32) - global_state_dict[key].to(torch.float32)
                delta_key += client_delta * weight
            delta_t[key] = delta_key

        # 2. Update server momentum m_t and velocity v_t
        new_global_state: Dict[str, torch.Tensor] = {}
        for key, g_val in global_state_dict.items():
            d = delta_t[key]
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * d
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (d ** 2)

            # 3. Apply FedAdam step to global model
            step = self.m[key] / (torch.sqrt(self.v[key]) + self.tau)
            updated_tensor = g_val.to(torch.float32) + self.lr * step
            new_global_state[key] = updated_tensor.to(g_val.dtype)

        return new_global_state
