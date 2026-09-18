from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Type
import logging
import numpy as np
from fedids_bench.data.schema import PreparedDataset
from fedids_bench.config import PartitioningConfig

logger = logging.getLogger("fedids_bench")

class PartitionResult:
    def __init__(
        self,
        client_indices: Dict[int, np.ndarray],
        heldout_client_ids: List[int],
        manifest: Dict[str, Any]
    ):
        self.client_indices = client_indices
        self.heldout_client_ids = heldout_client_ids
        self.manifest = manifest


class BasePartitioner(ABC):
    @abstractmethod
    def partition(self, dataset: PreparedDataset, config: PartitioningConfig, seed: int = 42) -> PartitionResult:
        pass


def _calculate_heldout_count(num_clients: int, requested_fraction: float) -> int:
    """MEDIUM-3 FIX: Enforce minimum floor for heldout clients when fraction > 0."""
    if requested_fraction <= 0.0 or num_clients < 2:
        if num_clients < 2 and requested_fraction > 0.0:
            logger.warning(f"num_clients={num_clients} is too small for heldout client splits. Setting n_heldout=0.")
        return 0
    n_heldout = max(1, int(np.ceil(num_clients * requested_fraction)))
    if n_heldout >= num_clients:
        n_heldout = num_clients - 1
    return n_heldout


class IIDPartitioner(BasePartitioner):
    def partition(self, dataset: PreparedDataset, config: PartitioningConfig, seed: int = 42) -> PartitionResult:
        rng = np.random.default_rng(seed)
        
        train_indices = np.where(dataset.split == 0)[0]
        rng.shuffle(train_indices)
        
        num_clients = config.num_clients
        n_heldout = _calculate_heldout_count(num_clients, config.heldout_client_fraction)
        
        all_client_ids = list(range(num_clients))
        heldout_client_ids = list(all_client_ids[:n_heldout])
        active_client_ids = list(all_client_ids[n_heldout:])
        n_active = len(active_client_ids)
        
        client_indices: Dict[int, np.ndarray] = {}
        
        if n_active > 0:
            splits = np.array_split(train_indices, n_active)
            for idx, cid in enumerate(active_client_ids):
                client_indices[cid] = splits[idx]
                
        for cid in heldout_client_ids:
            client_indices[cid] = np.array([], dtype=np.int64)
            
        manifest = _build_partition_manifest("iid", dataset, client_indices, heldout_client_ids, config)
        return PartitionResult(client_indices, heldout_client_ids, manifest)


class DirichletPartitioner(BasePartitioner):
    def partition(self, dataset: PreparedDataset, config: PartitioningConfig, seed: int = 42) -> PartitionResult:
        rng = np.random.default_rng(seed)
        
        train_indices = np.where(dataset.split == 0)[0]
        y_train = dataset.y[train_indices]
        
        num_clients = config.num_clients
        n_heldout = _calculate_heldout_count(num_clients, config.heldout_client_fraction)
        
        all_client_ids = list(range(num_clients))
        heldout_client_ids = list(all_client_ids[:n_heldout])
        active_client_ids = list(all_client_ids[n_heldout:])
        n_active = len(active_client_ids)
        
        client_indices: Dict[int, np.ndarray] = {cid: np.array([], dtype=np.int64) for cid in all_client_ids}
        
        if n_active > 0:
            n_classes = len(np.unique(dataset.y))
            client_sample_idx: List[List[int]] = [[] for _ in range(n_active)]
            
            for c in range(n_classes):
                c_indices = train_indices[y_train == c]
                rng.shuffle(c_indices)
                
                proportions = rng.dirichlet(np.repeat(config.alpha, n_active))
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]
                
                c_splits = np.split(c_indices, proportions)
                for i in range(n_active):
                    client_sample_idx[i].extend(c_splits[i])
                    
            for idx, cid in enumerate(active_client_ids):
                arr = np.array(client_sample_idx[idx], dtype=np.int64)
                rng.shuffle(arr)
                client_indices[cid] = arr
                
        manifest = _build_partition_manifest("dirichlet", dataset, client_indices, heldout_client_ids, config)
        return PartitionResult(client_indices, heldout_client_ids, manifest)


class LabelSkewPartitioner(BasePartitioner):
    """Partition dataset such that each active client receives a subset of attack classes."""
    def partition(self, dataset: PreparedDataset, config: PartitioningConfig, seed: int = 42) -> PartitionResult:
        rng = np.random.default_rng(seed)
        
        train_indices = np.where(dataset.split == 0)[0]
        y_train = dataset.y[train_indices]
        
        num_clients = config.num_clients
        n_heldout = _calculate_heldout_count(num_clients, config.heldout_client_fraction)
        
        all_client_ids = list(range(num_clients))
        heldout_client_ids = list(all_client_ids[:n_heldout])
        active_client_ids = list(all_client_ids[n_heldout:])
        n_active = len(active_client_ids)
        
        client_indices: Dict[int, np.ndarray] = {cid: np.array([], dtype=np.int64) for cid in all_client_ids}
        
        if n_active > 0:
            n_classes = len(np.unique(dataset.y))
            # Assign classes to clients round-robin
            for c in range(n_classes):
                c_indices = train_indices[y_train == c]
                rng.shuffle(c_indices)
                c_splits = np.array_split(c_indices, n_active)
                for i, cid in enumerate(active_client_ids):
                    client_indices[cid] = np.concatenate([client_indices[cid], c_splits[i]])

        manifest = _build_partition_manifest("label_skew", dataset, client_indices, heldout_client_ids, config)
        return PartitionResult(client_indices, heldout_client_ids, manifest)


class QuantitySkewPartitioner(BasePartitioner):
    """Partition dataset with Dirichlet quantity skew (unequal sample sizes per client)."""
    def partition(self, dataset: PreparedDataset, config: PartitioningConfig, seed: int = 42) -> PartitionResult:
        rng = np.random.default_rng(seed)
        
        train_indices = np.where(dataset.split == 0)[0]
        rng.shuffle(train_indices)
        
        num_clients = config.num_clients
        n_heldout = _calculate_heldout_count(num_clients, config.heldout_client_fraction)
        
        all_client_ids = list(range(num_clients))
        heldout_client_ids = list(all_client_ids[:n_heldout])
        active_client_ids = list(all_client_ids[n_heldout:])
        n_active = len(active_client_ids)
        
        client_indices: Dict[int, np.ndarray] = {cid: np.array([], dtype=np.int64) for cid in all_client_ids}
        
        if n_active > 0:
            proportions = rng.dirichlet(np.repeat(config.alpha, n_active))
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(train_indices)).astype(int)[:-1]
            
            c_splits = np.split(train_indices, split_points)
            for i, cid in enumerate(active_client_ids):
                client_indices[cid] = c_splits[i]

        manifest = _build_partition_manifest("quantity_skew", dataset, client_indices, heldout_client_ids, config)
        return PartitionResult(client_indices, heldout_client_ids, manifest)


def _build_partition_manifest(
    strategy: str,
    dataset: PreparedDataset,
    client_indices: Dict[int, np.ndarray],
    heldout_client_ids: List[int],
    config: PartitioningConfig
) -> Dict[str, Any]:
    client_specs = {}
    for cid, idxs in client_indices.items():
        is_heldout = cid in heldout_client_ids
        if len(idxs) > 0:
            class_dist = {int(c): int(np.sum(dataset.y[idxs] == c)) for c in np.unique(dataset.y)}
        else:
            class_dist = {}
        client_specs[int(cid)] = {
            "client_id": int(cid),
            "is_heldout": is_heldout,
            "n_samples": len(idxs),
            "class_distribution": class_dist
        }
        
    return {
        "strategy": strategy,
        "num_clients": config.num_clients,
        "heldout_clients_count": len(heldout_client_ids),
        "heldout_client_ids": [int(x) for x in heldout_client_ids],
        "clients": client_specs
    }


PARTITIONER_REGISTRY: Dict[str, Type[BasePartitioner]] = {
    "iid": IIDPartitioner,
    "dirichlet": DirichletPartitioner,
    "label_skew": LabelSkewPartitioner,
    "quantity_skew": QuantitySkewPartitioner
}

def get_partitioner(strategy: str) -> BasePartitioner:
    if strategy not in PARTITIONER_REGISTRY:
        raise ValueError(f"Unknown partition strategy: '{strategy}'. Available: {list(PARTITIONER_REGISTRY.keys())}")
    return PARTITIONER_REGISTRY[strategy]()
