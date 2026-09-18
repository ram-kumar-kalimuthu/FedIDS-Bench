from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import torch
import torch.nn as nn


class IDSModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return state dict with detached cloned tensors."""
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def set_parameters(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load state dict into model."""
        self.load_state_dict(state_dict)

    def num_parameters(self) -> int:
        """Total trainable and non-trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def parameter_bytes(self) -> int:
        """
        Exact raw parameter tensor byte size: SUM(numel * element_size_in_bytes)
        over all trainable and non-trainable parameters.
        """
        total_bytes = 0
        for p in self.parameters():
            total_bytes += p.numel() * p.element_size()
        return total_bytes

    def total_state_bytes(self) -> int:
        """
        HIGH-3 FIX: Measure total wire transfer byte size including persistent state buffers
        (e.g., BatchNorm running_mean, running_var) from state_dict().
        """
        total_bytes = 0
        for v in self.state_dict().values():
            total_bytes += v.numel() * v.element_size()
        return total_bytes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities for numpy input array."""
        self.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X).float()
            logits = self.forward(tensor_x)
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute class predictions for numpy input array."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)

