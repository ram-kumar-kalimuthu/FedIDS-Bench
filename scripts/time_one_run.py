import time
from fedids_bench.config import ExperimentConfig, DatasetConfig, PartitioningConfig, FederatedConfig
from fedids_bench.experiments.runner import run_experiment

cfg = ExperimentConfig(
    experiment_id="timing_test",
    seed=42,
    mode="federated",
    dataset=DatasetConfig(name="unsw_nb15", n_samples=2000),
    partitioning=PartitioningConfig(strategy="dirichlet", num_clients=10, alpha=0.5, heldout_client_fraction=0.0),
    federated=FederatedConfig(algorithm="fedavg", rounds=20, clients_per_round=10, batch_size=32, local_epochs=1)
)
t0 = time.time()
res = run_experiment(cfg)
t1 = time.time()
f1 = res["metrics"]["f1"]
print(f"ELAPSED SECONDS FOR 20 ROUNDS: {t1-t0:.2f}s, F1: {f1:.4f}")
