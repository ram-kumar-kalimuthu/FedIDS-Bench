from typing import List, Optional
import torch
import torch.nn as nn
from fedids_bench.models.base import IDSModel

class SmallMLP(IDSModel):
    def __init__(self, n_features: int, n_classes: int, hidden_dims: Optional[List[int]] = None, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            
        if hidden_dims is None:
            hidden_dims = [32, 16]
            
        layers: List[nn.Module] = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
