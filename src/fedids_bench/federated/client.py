import time
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from fedids_bench.models.base import IDSModel
from fedids_bench.federated.trainer import train_local
from fedids_bench.federated.algorithms.fedprox import train_local_fedprox
from fedids_bench.attacks.base import BaseAttack

class FederatedClient:
    def __init__(self, client_id: int, X: np.ndarray, y: np.ndarray, is_heldout: bool = False):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.is_heldout = is_heldout

    @property
    def num_samples(self) -> int:
        return len(self.X)

    def train(
        self,
        global_model: IDSModel,
        epochs: int,
        batch_size: int,
        lr: float,
        base_seed: int,
        round_num: int,
        algorithm: str = "fedavg",
        mu: float = 0.01,
        attack: Optional[BaseAttack] = None,
        is_malicious: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """
        Perform client local training.
        Supports FedAvg, FedProx, data attacks, and model attacks.
        """
        if self.is_heldout or self.num_samples == 0:
            raise ValueError(f"Client {self.client_id} is heldout or has no training data.")

        client_round_seed = base_seed + (self.client_id * 10000) + round_num

        X_train, y_train = self.X, self.y
        if is_malicious and attack is not None:
            X_train, y_train = attack.apply_data_attack(self.X, self.y, self.client_id, client_round_seed)

        import copy
        client_model = copy.deepcopy(global_model)

        start_time = time.perf_counter()

        if algorithm == "fedprox":
            updated_weights, n_k = train_local_fedprox(
                model=client_model,
                global_model=global_model,
                X=X_train,
                y=y_train,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                mu=mu,
                seed=client_round_seed
            )
        else:
            updated_weights, n_k = train_local(
                model=client_model,
                X=X_train,
                y=y_train,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                seed=client_round_seed
            )

        if is_malicious and attack is not None:
            updated_weights = attack.apply_model_attack(
                local_weights=updated_weights,
                global_weights=global_model.get_parameters(),
                client_id=self.client_id,
                seed=client_round_seed
            )

        duration = time.perf_counter() - start_time
        return updated_weights, n_k, duration

