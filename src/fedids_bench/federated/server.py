import time
from typing import List, Dict, Tuple, Any
import numpy as np
import torch

from fedids_bench.models.base import IDSModel
from fedids_bench.federated.client import FederatedClient
from fedids_bench.federated.algorithms.fedavg import fedavg_aggregate
from fedids_bench.federated.algorithms.fedadam import FedAdamServerOptimizer
from fedids_bench.attacks.factory import get_attack
from fedids_bench.defenses.factory import get_defense
from fedids_bench.evaluation.communication import CommunicationTracker
from fedids_bench.evaluation.resource_cost import ResourceTracker
from fedids_bench.config import ExperimentConfig

class FederatedServer:
    def __init__(self, global_model: IDSModel, config: ExperimentConfig):
        self.global_model = global_model
        self.config = config
        self.comm_tracker = CommunicationTracker(global_model)
        self.resource_tracker = ResourceTracker(global_model)
        
        self.attack = get_attack(config.attack, n_classes=config.dataset.n_classes)
        self.defense = get_defense(config.defense)

        if config.federated.algorithm == "fedadam":
            self.fedadam_optimizer = FedAdamServerOptimizer(
                beta1=config.federated.beta1,
                beta2=config.federated.beta2,
                tau=config.federated.tau,
                lr=config.model.learning_rate
            )
        else:
            self.fedadam_optimizer = None

    def select_clients(self, active_clients: List[FederatedClient], round_num: int, seed: int) -> List[FederatedClient]:
        """Select a subset of active clients for this round."""
        rng = np.random.default_rng(seed + round_num)
        k = min(self.config.federated.clients_per_round, len(active_clients))
        selected_idx = rng.choice(len(active_clients), size=k, replace=False)
        return [active_clients[i] for i in selected_idx]

    def run_round(
        self,
        active_clients: List[FederatedClient],
        round_num: int
    ) -> Dict[str, Any]:
        """Orchestrate a single FL round."""
        selected_clients = self.select_clients(active_clients, round_num, self.config.seed)
        num_participating = len(selected_clients)

        self.comm_tracker.record_round(round_num, num_participating)

        # Determine total malicious count across active clients
        num_malicious = int(len(active_clients) * self.config.attack.malicious_fraction)
        malicious_client_ids = set(range(num_malicious))

        attack_active = (self.config.attack.type != "none") and (round_num >= self.config.attack.start_round)

        client_updates: List[Dict[str, torch.Tensor]] = []
        sample_counts: List[int] = []

        for client in selected_clients:
            is_mal = attack_active and (client.client_id in malicious_client_ids)
            update_weights, n_k, train_duration = client.train(
                global_model=self.global_model,
                epochs=self.config.federated.local_epochs,
                batch_size=self.config.federated.batch_size,
                lr=self.config.model.learning_rate,
                base_seed=self.config.seed,
                round_num=round_num,
                algorithm=self.config.federated.algorithm,
                mu=self.config.federated.mu,
                attack=self.attack,
                is_malicious=is_mal
            )
            client_updates.append(update_weights)
            sample_counts.append(n_k)
            self.resource_tracker.add_client_training_time(train_duration)

        agg_start = time.perf_counter()
        
        trust_nk = self.config.federated.trust_client_sample_counts

        # If a robust defense is explicitly configured, use defense aggregation
        if self.defense is not None:
            new_global_weights = self.defense.aggregate(
                global_weights=self.global_model.get_parameters(),
                client_updates=client_updates,
                sample_counts=sample_counts,
                trust_client_sample_counts=trust_nk
            )
        elif self.config.federated.algorithm in ["fedavg", "fedprox"]:
            new_global_weights = fedavg_aggregate(client_updates, sample_counts, trust_client_sample_counts=trust_nk)
        elif self.config.federated.algorithm == "fedadam":
            new_global_weights = self.fedadam_optimizer.aggregate(
                global_state_dict=self.global_model.get_parameters(),
                client_updates=client_updates,
                sample_counts=sample_counts,
                trust_client_sample_counts=trust_nk
            )
        else:
            raise NotImplementedError(f"Algorithm '{self.config.federated.algorithm}' not recognized.")
        
        agg_duration = time.perf_counter() - agg_start
        self.resource_tracker.add_server_aggregation_time(agg_duration)

        self.global_model.set_parameters(new_global_weights)

        return {
            "round_num": round_num,
            "participating_clients": [c.client_id for c in selected_clients],
            "sample_counts": sample_counts,
            "aggregation_time_sec": round(agg_duration, 4)
        }

