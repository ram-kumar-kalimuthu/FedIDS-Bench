import time
import numpy as np
import torch

from fedids_bench.config import ExperimentConfig
from fedids_bench.data.loaders import get_dataset_loader
from fedids_bench.data.partition import get_partitioner
from fedids_bench.models.mlp import SmallMLP
from fedids_bench.federated.client import FederatedClient
from fedids_bench.federated.server import FederatedServer
from fedids_bench.evaluation.classification import compute_classification_metrics
from fedids_bench.utils.reproducibility import set_seed, get_environment_info
from fedids_bench.utils.serialization import save_experiment_results
from fedids_bench.utils.logging import setup_logger
from fedids_bench.experiments.centralized import run_centralized_experiment

logger = setup_logger()

def evaluate_model_on_split(model: SmallMLP, X: np.ndarray, y: np.ndarray, label_map: dict) -> dict:
    """Helper to evaluate model on a given numpy dataset split."""
    if len(X) == 0:
        return {}
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        preds = torch.argmax(logits, dim=1).numpy()
    return compute_classification_metrics(y, preds, label_map)


def run_experiment(config: ExperimentConfig) -> dict:
    """Main experiment runner entry point."""
    set_seed(config.seed)
    env_info = get_environment_info()

    logger.info(f"Starting experiment: {config.experiment_id} (mode={config.mode})")

    # 1. Load Dataset
    loader = get_dataset_loader(config.dataset.name)
    dataset = loader.load(config.dataset, seed=config.seed)
    
    if config.mode == "centralized":
        results = run_centralized_experiment(dataset, config)
        results["environment"] = env_info
        save_experiment_results(results, config, config.output_dir)
        return results

    # 2. Partition Dataset across clients
    partitioner = get_partitioner(config.partitioning.strategy)
    partition_res = partitioner.partition(dataset, config.partitioning, seed=config.seed)

    # 3. Create Federated Clients
    clients: list[FederatedClient] = []
    for cid, idxs in partition_res.client_indices.items():
        is_heldout = cid in partition_res.heldout_client_ids
        c = FederatedClient(
            client_id=cid,
            X=dataset.X[idxs] if len(idxs) > 0 else np.empty((0, dataset.X.shape[1]), dtype=np.float32),
            y=dataset.y[idxs] if len(idxs) > 0 else np.empty((0,), dtype=np.int64),
            is_heldout=is_heldout
        )
        clients.append(c)

    active_clients = [c for c in clients if not c.is_heldout and c.num_samples > 0]
    heldout_clients = [c for c in clients if c.is_heldout and c.num_samples > 0]

    # 4. Initialize Global Model & Server
    n_features = dataset.X.shape[1]
    n_classes = len(np.unique(dataset.y))

    global_model = SmallMLP(
        n_features=n_features,
        n_classes=n_classes,
        hidden_dims=config.model.hidden_dims,
        seed=config.seed
    )

    server = FederatedServer(global_model, config)

    # Prepare Tier 1 Global Test Set
    test_mask = (dataset.split == 2)
    X_global_test = dataset.X[test_mask]
    y_global_test = dataset.y[test_mask]

    # Prepare Tier 2 Heldout Clients Test Data
    if len(heldout_clients) > 0:
        X_heldout_test = np.concatenate([c.X for c in heldout_clients], axis=0)
        y_heldout_test = np.concatenate([c.y for c in heldout_clients], axis=0)
    else:
        X_heldout_test = np.empty((0, n_features), dtype=np.float32)
        y_heldout_test = np.empty((0,), dtype=np.int64)

    label_map = dataset.manifest.get("label_map", {})
    round_logs = []

    # 5. Execute FL Rounds
    for r in range(1, config.federated.rounds + 1):
        round_info = server.run_round(active_clients, round_num=r)
        
        # Evaluate after round
        tier1_metrics = evaluate_model_on_split(global_model, X_global_test, y_global_test, label_map)
        
        round_info["tier1_global_test_f1"] = tier1_metrics.get("f1", 0.0)
        round_logs.append(round_info)
        
        logger.info(
            f"Round {r}/{config.federated.rounds} - "
            f"Global Test F1: {tier1_metrics.get('f1', 0.0):.4f} - "
            f"Aggregation Time: {round_info['aggregation_time_sec']}s"
        )

    # Final Evaluation (Tier 1 & Tier 2)
    eval_start = time.perf_counter()
    tier1_final = evaluate_model_on_split(global_model, X_global_test, y_global_test, label_map)
    tier2_final = evaluate_model_on_split(global_model, X_heldout_test, y_heldout_test, label_map)
    eval_duration = time.perf_counter() - eval_start
    server.resource_tracker.add_inference_time(eval_duration)

    comm_summary = server.comm_tracker.get_summary()
    resource_summary = server.resource_tracker.get_summary()

    final_result = {
        "experiment_id": config.experiment_id,
        "mode": "federated",
        "dataset": config.dataset.name,
        "algorithm": config.federated.algorithm,
        "model": config.model.architecture,
        "num_clients": config.partitioning.num_clients,
        "active_clients": len(active_clients),
        "heldout_clients": len(heldout_clients),
        "metrics": tier1_final,
        "tier2_heldout_clients_metrics": tier2_final,
        "communication": comm_summary,
        "resource_cost": resource_summary,
        "partition_manifest": partition_res.manifest,
        "round_logs": round_logs,
        "environment": env_info,
        "status": "completed"
    }

    save_experiment_results(final_result, config, config.output_dir)
    logger.info(f"Experiment finished successfully. Results saved to: {config.output_dir}")
    return final_result
