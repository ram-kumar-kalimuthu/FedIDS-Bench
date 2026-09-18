import numpy as np
import torch
import pytest

from fedids_bench.config import DatasetConfig, PartitioningConfig, ExperimentConfig
from fedids_bench.data.synthetic import generate_synthetic_ids
from fedids_bench.data.partition import LabelSkewPartitioner, QuantitySkewPartitioner
from fedids_bench.federated.algorithms.fedavg import fedavg_aggregate
from fedids_bench.federated.algorithms.fedadam import FedAdamServerOptimizer
from fedids_bench.evaluation.classification import compute_classification_metrics
from fedids_bench.models.mlp import SmallMLP
from fedids_bench.experiments.runner import run_experiment

def test_scaler_non_leakage():
    """CRITICAL-1: Test that scaler standardizes strictly on training split."""
    cfg = DatasetConfig(n_samples=500, test_fraction=0.2)
    ds = generate_synthetic_ids(cfg, seed=42)

    train_mask = (ds.split == 0)
    # Means of training features after transformation should be ~0.0
    train_means = ds.X[train_mask].mean(axis=0)
    np.testing.assert_allclose(train_means, 0.0, atol=1e-5)

    assert "scaler" in ds.manifest
    assert "mean" in ds.manifest["scaler"]
    assert "std" in ds.manifest["scaler"]

def test_trust_client_sample_counts_defense():
    """CRITICAL-2: Test equal weighting defense when trust_client_sample_counts=False."""
    w1 = {"w": torch.tensor([1.0])}
    w2 = {"w": torch.tensor([5.0])}

    # Client 2 claims 999,000 samples vs Client 1's 1,000 samples
    client_updates = [w1, w2]
    sample_counts = [1000, 999000]

    # Weighted (Trusted): Client 2 dominates -> ~5.0
    agg_trusted = fedavg_aggregate(client_updates, sample_counts, trust_client_sample_counts=True)
    assert agg_trusted["w"].item() > 4.9

    # Untrusted (Equal Defense): Average of 1.0 and 5.0 -> 3.0
    agg_untrusted = fedavg_aggregate(client_updates, sample_counts, trust_client_sample_counts=False)
    assert agg_untrusted["w"].item() == 3.0

def test_single_class_metrics_none_fpr():
    """HIGH-1: Test that single-class predictions return None for FPR/FNR when denominators are 0."""
    # All attack samples (no benign samples present -> FPR denominator is 0)
    y_true_attack = np.array([1, 1, 1, 1])
    y_pred_attack = np.array([1, 1, 1, 1])
    m_attack = compute_classification_metrics(y_true_attack, y_pred_attack)
    assert m_attack["false_positive_rate"] is None
    assert m_attack["benign_recall"] is None

    # All benign samples (no attack samples present -> FNR denominator is 0)
    y_true_benign = np.array([0, 0, 0, 0])
    y_pred_benign = np.array([0, 0, 0, 0])
    m_benign = compute_classification_metrics(y_true_benign, y_pred_benign)
    assert m_benign["false_negative_rate"] is None
    assert m_benign["attack_recall"] is None

def test_total_state_bytes():
    """HIGH-3: Test total_state_bytes including state_dict tensors."""
    model = SmallMLP(n_features=10, n_classes=2, hidden_dims=[5], seed=42)
    assert model.total_state_bytes() == model.parameter_bytes()

def test_label_skew_partitioner():
    cfg_ds = DatasetConfig(n_samples=400, n_classes=4, test_fraction=0.2)
    ds = generate_synthetic_ids(cfg_ds, seed=42)

    p_cfg = PartitioningConfig(strategy="label_skew", num_clients=4, heldout_client_fraction=0.0)
    partitioner = LabelSkewPartitioner()
    res = partitioner.partition(ds, p_cfg, seed=42)

    assert len(res.client_indices) == 4
    total_samples = sum(len(idxs) for idxs in res.client_indices.values())
    assert total_samples == np.sum(ds.split == 0)

def test_quantity_skew_partitioner():
    cfg_ds = DatasetConfig(n_samples=500, test_fraction=0.2)
    ds = generate_synthetic_ids(cfg_ds, seed=42)

    p_cfg = PartitioningConfig(strategy="quantity_skew", num_clients=5, heldout_client_fraction=0.2, alpha=0.1)
    partitioner = QuantitySkewPartitioner()
    res = partitioner.partition(ds, p_cfg, seed=42)

    assert len(res.heldout_client_ids) == 1
    total_samples = sum(len(idxs) for idxs in res.client_indices.values())
    assert total_samples == np.sum(ds.split == 0)

def test_fedprox_runner_execution(tmp_path):
    cfg = ExperimentConfig(
        experiment_id="test_fedprox",
        output_dir=str(tmp_path / "fedprox"),
        federated={"algorithm": "fedprox", "mu": 0.05, "rounds": 2, "clients_per_round": 3, "local_epochs": 1},
        partitioning={"num_clients": 3, "heldout_client_fraction": 0.0}
    )
    res = run_experiment(cfg)
    assert res["status"] == "completed"
    assert res["algorithm"] == "fedprox"

def test_fedadam_runner_execution(tmp_path):
    cfg = ExperimentConfig(
        experiment_id="test_fedadam",
        output_dir=str(tmp_path / "fedadam"),
        federated={"algorithm": "fedadam", "beta1": 0.9, "beta2": 0.999, "rounds": 2, "clients_per_round": 3, "local_epochs": 1},
        partitioning={"num_clients": 3, "heldout_client_fraction": 0.0}
    )
    res = run_experiment(cfg)
    assert res["status"] == "completed"
    assert res["algorithm"] == "fedadam"
