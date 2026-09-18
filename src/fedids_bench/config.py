import os
import yaml
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, model_validator

class DatasetConfig(BaseModel):
    name: str = "synthetic"
    n_samples: int = Field(default=1000, ge=10)
    n_features: int = Field(default=20, ge=2)
    n_classes: int = Field(default=4, ge=2)
    test_fraction: float = Field(default=0.2, ge=0.01, le=0.5)
    data_dir: str = "data/raw"

class PartitioningConfig(BaseModel):
    strategy: Literal["iid", "dirichlet", "label_skew", "quantity_skew"] = "iid"
    num_clients: int = Field(default=5, ge=1)
    heldout_client_fraction: float = Field(default=0.2, ge=0.0, le=0.5)
    alpha: float = Field(default=0.5, gt=0.0)

class ModelConfig(BaseModel):
    architecture: Literal["mlp", "classical"] = "mlp"
    hidden_dims: List[int] = Field(default_factory=lambda: [32, 16])
    learning_rate: float = Field(default=0.01, gt=0.0)

class FederatedConfig(BaseModel):
    algorithm: Literal["fedavg", "fedprox", "fedadam"] = "fedavg"
    rounds: int = Field(default=3, ge=1)
    local_epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=32, ge=1)
    clients_per_round: int = Field(default=5, ge=1)
    trust_client_sample_counts: bool = True
    mu: float = Field(default=0.01, ge=0.0)  # FedProx proximal term
    beta1: float = Field(default=0.9, ge=0.0, le=1.0)  # FedAdam momentum
    beta2: float = Field(default=0.999, ge=0.0, le=1.0)  # FedAdam velocity
    tau: float = Field(default=1e-3, gt=0.0)  # FedAdam adaptivity parameter

class AttackConfig(BaseModel):
    type: Literal["none", "label_flip", "model_poison", "byzantine", "backdoor"] = "none"
    malicious_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    start_round: int = Field(default=0, ge=0)
    strength: float = Field(default=1.0, ge=0.0)
    target_label: Optional[int] = None

class DefenseConfig(BaseModel):
    type: Literal["none", "clipping", "trimmed_mean", "median", "krum"] = "none"
    clip_threshold: float = Field(default=5.0, gt=0.0)
    trim_ratio: float = Field(default=0.2, ge=0.0, lt=0.5)
    krum_f: int = Field(default=1, ge=0)
    krum_multi: bool = False
    krum_m: int = Field(default=1, ge=1)


class ExperimentConfig(BaseModel):
    experiment_id: str = "default_exp"
    seed: int = 42
    mode: Literal["federated", "centralized"] = "federated"
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    partitioning: PartitioningConfig = Field(default_factory=PartitioningConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    attack: AttackConfig = Field(default_factory=AttackConfig)
    defense: DefenseConfig = Field(default_factory=DefenseConfig)
    output_dir: str = "results/default"

    @model_validator(mode="after")
    def validate_cross_fields(self) -> 'ExperimentConfig':
        if self.federated.clients_per_round > self.partitioning.num_clients:
            raise ValueError(
                f"clients_per_round ({self.federated.clients_per_round}) cannot exceed "
                f"num_clients ({self.partitioning.num_clients})."
            )
        
        # Defense validation check (e.g. Krum requires > 2*f + 2 clients)
        if self.defense.type == "krum" and self.attack.type != "none":
            f = int(self.partitioning.num_clients * self.attack.malicious_fraction)
            if self.partitioning.num_clients <= 2 * f + 2:
                raise ValueError(
                    f"Krum defense requires num_clients > 2*f + 2. "
                    f"Got num_clients={self.partitioning.num_clients}, f={f}."
                )
        return self


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str, overrides: Optional[dict] = None) -> ExperimentConfig:
    """Load config from YAML file, apply optional overrides, and validate via Pydantic."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        raw_dict = yaml.safe_load(f) or {}

    if overrides:
        raw_dict = deep_merge(raw_dict, overrides)

    return ExperimentConfig(**raw_dict)
