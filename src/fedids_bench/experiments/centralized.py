import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fedids_bench.data.schema import PreparedDataset
from fedids_bench.models.mlp import SmallMLP
from fedids_bench.federated.trainer import NumPyDataset
from fedids_bench.evaluation.classification import compute_classification_metrics
from fedids_bench.config import ExperimentConfig

def run_centralized_experiment(dataset: PreparedDataset, config: ExperimentConfig) -> dict:
    """Run centralized baseline training on the full training split."""
    start_time = time.perf_counter()

    train_mask = (dataset.split == 0)
    test_mask = (dataset.split == 2)

    X_train, y_train = dataset.X[train_mask], dataset.y[train_mask]
    X_test, y_test = dataset.X[test_mask], dataset.y[test_mask]

    n_features = dataset.X.shape[1]
    n_classes = len(np.unique(dataset.y))

    model = SmallMLP(
        n_features=n_features,
        n_classes=n_classes,
        hidden_dims=config.model.hidden_dims,
        seed=config.seed
    )

    train_dataset = NumPyDataset(X_train, y_train)
    g = torch.Generator()
    g.manual_seed(config.seed)
    
    loader = DataLoader(
        train_dataset,
        batch_size=config.federated.batch_size,
        shuffle=True,
        generator=g
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=config.model.learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_epochs = config.federated.rounds * config.federated.local_epochs

    model.train()
    for epoch in range(total_epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    training_duration = time.perf_counter() - start_time

    # Evaluate on Tier 1 Global Test Set
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test, dtype=torch.float32))
        test_preds = torch.argmax(test_logits, dim=1).numpy()

    metrics = compute_classification_metrics(y_test, test_preds, dataset.manifest.get("label_map"))

    return {
        "experiment_id": config.experiment_id,
        "mode": "centralized",
        "dataset": config.dataset.name,
        "metrics": metrics,
        "communication": {
            "parameter_bytes": model.parameter_bytes(),
            "total_upload_bytes": 0,
            "total_download_bytes": 0,
            "total_communication_bytes": 0
        },
        "resource_cost": {
            "num_parameters": model.num_parameters(),
            "model_size_bytes": model.parameter_bytes(),
            "client_training_time_sec": round(training_duration, 4),
            "server_aggregation_time_sec": 0.0,
            "inference_time_sec": 0.0
        },
        "status": "completed"
    }
