import os
import json
import csv
from typing import Dict, Any
from fedids_bench.config import ExperimentConfig

def save_experiment_results(
    result_data: Dict[str, Any],
    config: ExperimentConfig,
    output_dir: str
) -> str:
    """Save experiment results atomically to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Complete payload with config serialization
    payload = {
        "config": config.model_dump(mode="json"),
        "result": result_data
    }
    
    json_path = os.path.join(output_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Section 31 flat schema record CSV export
    csv_path = os.path.join(output_dir, "result.csv")
    
    metrics = result_data.get("metrics", {})
    comm = result_data.get("communication", {})
    resources = result_data.get("resource_cost", {})
    
    row = {
        "experiment_id": config.experiment_id,
        "dataset": config.dataset.name,
        "algorithm": config.federated.algorithm,
        "model": config.model.architecture,
        "partition_strategy": config.partitioning.strategy,
        "num_clients": config.partitioning.num_clients,
        "clients_per_round": config.federated.clients_per_round,
        "local_epochs": config.federated.local_epochs,
        "attack": config.attack.type,
        "malicious_fraction": config.attack.malicious_fraction,
        "defense": config.defense.type,
        "seed": config.seed,
        "rounds": config.federated.rounds,
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_f1": metrics.get("multiclass", {}).get("macro_f1", 0.0),
        "weighted_f1": metrics.get("multiclass", {}).get("weighted_f1", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "false_positive_rate": metrics.get("false_positive_rate", 0.0),
        "false_negative_rate": metrics.get("false_negative_rate", 0.0),
        "attack_recall": metrics.get("attack_recall", 0.0),
        "communication_upload_bytes": comm.get("total_upload_bytes", 0),
        "communication_download_bytes": comm.get("total_download_bytes", 0),
        "communication_total_bytes": comm.get("total_communication_bytes", 0),
        "client_training_time": resources.get("client_training_time_sec", 0.0),
        "server_aggregation_time": resources.get("server_aggregation_time_sec", 0.0),
        "inference_time": resources.get("inference_time_sec", 0.0),
        "num_parameters": resources.get("num_parameters", 0),
        "model_size_bytes": resources.get("model_size_bytes", 0),
        "status": result_data.get("status", "completed")
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    return json_path
