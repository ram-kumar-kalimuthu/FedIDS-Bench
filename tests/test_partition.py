import numpy as np
from fedids_bench.config import DatasetConfig, PartitioningConfig
from fedids_bench.data.synthetic import generate_synthetic_ids
from fedids_bench.data.partition import IIDPartitioner, DirichletPartitioner

def test_iid_partitioner_heldout():
    ds_cfg = DatasetConfig(n_samples=500, test_fraction=0.2)
    ds = generate_synthetic_ids(ds_cfg, seed=42)
    
    p_cfg = PartitioningConfig(strategy="iid", num_clients=5, heldout_client_fraction=0.2)
    partitioner = IIDPartitioner()
    res = partitioner.partition(ds, p_cfg, seed=42)
    
    assert len(res.heldout_client_ids) == 1
    assert len(res.client_indices[res.heldout_client_ids[0]]) == 0
    
    # Active clients check
    active_samples = sum(len(res.client_indices[cid]) for cid in res.client_indices if cid not in res.heldout_client_ids)
    train_samples = np.sum(ds.split == 0)
    assert active_samples == train_samples

def test_dirichlet_partitioner():
    ds_cfg = DatasetConfig(n_samples=600, n_classes=4, test_fraction=0.2)
    ds = generate_synthetic_ids(ds_cfg, seed=42)
    
    p_cfg = PartitioningConfig(strategy="dirichlet", num_clients=5, heldout_client_fraction=0.0, alpha=0.5)
    partitioner = DirichletPartitioner()
    res = partitioner.partition(ds, p_cfg, seed=42)
    
    total_assigned = sum(len(idxs) for idxs in res.client_indices.values())
    assert total_assigned == np.sum(ds.split == 0)
