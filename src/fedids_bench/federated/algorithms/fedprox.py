import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np
from torch.utils.data import DataLoader
from fedids_bench.models.base import IDSModel
from fedids_bench.federated.trainer import NumPyDataset

def train_local_fedprox(
    model: IDSModel,
    global_model: IDSModel,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    mu: float,
    seed: int
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Perform local training for a client using FedProx (Li et al. 2020).
    Adds proximal regularization term: (mu / 2) * ||w - w_global||^2
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = NumPyDataset(X, y)
    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Freeze reference global parameters for proximal term
    global_params = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            # Proximal penalty: (mu / 2) * sum(||w - w_global||^2)
            if mu > 0.0:
                prox_loss = 0.0
                for name, param in model.named_parameters():
                    if name in global_params:
                        prox_loss += torch.sum((param - global_params[name]) ** 2)
                loss += (mu / 2.0) * prox_loss

            loss.backward()
            optimizer.step()

    return model.get_parameters(), len(X)
