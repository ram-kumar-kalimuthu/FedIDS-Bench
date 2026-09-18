from typing import Dict
import torch
from fedids_bench.attacks.base import BaseAttack

class ModelPoisonAttack(BaseAttack):
    """
    Model poisoning attack.
    Scales or flips the sign of local parameter update vectors delta = w_local - w_global.
    """
    def __init__(self, strength: float = -1.0):
        # Default strength = -1.0 means sign-inversion attack
        self.strength = strength

    def apply_model_attack(
        self,
        local_weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        client_id: int,
        seed: int
    ) -> Dict[str, torch.Tensor]:
        poisoned_weights = {}
        for name, local_tensor in local_weights.items():
            global_tensor = global_weights[name]
            delta = local_tensor - global_tensor
            poisoned_weights[name] = global_tensor + self.strength * delta
        return poisoned_weights
