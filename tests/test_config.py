import pytest
from fedids_bench.config import ExperimentConfig, load_config

def test_default_config_validation():
    cfg = ExperimentConfig()
    assert cfg.experiment_id == "default_exp"
    assert cfg.federated.algorithm == "fedavg"

def test_invalid_clients_per_round():
    with pytest.raises(ValueError, match="clients_per_round .* cannot exceed num_clients"):
        ExperimentConfig(
            partitioning={"num_clients": 5},
            federated={"clients_per_round": 10}
        )

def test_krum_defense_validation():
    with pytest.raises(ValueError, match="Krum defense requires"):
        ExperimentConfig(
            partitioning={"num_clients": 4},
            federated={"clients_per_round": 4},
            attack={"type": "model_poison", "malicious_fraction": 0.5},
            defense={"type": "krum"}
        )
