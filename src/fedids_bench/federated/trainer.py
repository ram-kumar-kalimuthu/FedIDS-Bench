import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
from fedids_bench.models.base import IDSModel

class NumPyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def train_local(
    model: IDSModel,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    momentum: float = 0.0,
    class_weights: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Perform local training for a client.
    `class_weights` (optional) are inverse-frequency cross-entropy weights shared by the server;
    `momentum` is plain SGD momentum (0.0 = the original behavior).
    Creates a FRESH optimizer instance for every local training pass.
    Uses strict seed setting for DataLoader shuffle determinism.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset = NumPyDataset(X, y)
    g = torch.Generator()
    g.manual_seed(seed)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )

    # Fresh optimizer per client local training pass
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    weight = None if class_weights is None else torch.tensor(np.asarray(class_weights), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return model.get_parameters(), len(X)
