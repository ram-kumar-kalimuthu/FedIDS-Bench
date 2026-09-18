from typing import Dict
import torch
from fedids_bench.attacks.base import BaseAttack

class ByzantineAttack(BaseAttack):
    """
    Byzantine attack.
    Replaces model updates with random noise scaled to match magnitude of normal updates.
    """
    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def apply_model_attack(
        self,
        local_weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        client_id: int,
        seed: int
    ) -> Dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed + client_id)
        poisoned_weights = {}
        for name, local_tensor in local_weights.items():
            global_tensor = global_weights[name]
            delta = local_tensor - global_tensor
            norm = torch.norm(delta)
            noise = torch.randn(delta.shape, generator=gen, dtype=delta.dtype, device=delta.device)
            noise_norm = torch.norm(noise)
            if noise_norm > 0:
                noise = noise * (norm / noise_norm) * self.strength
            poisoned_weights[name] = global_tensor + noise
        return poisoned_weights
