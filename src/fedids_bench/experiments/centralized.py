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
from fedids_bench.experiments.common import (
    balanced_class_weights, dataset_summary, eval_split_id, majority_baseline, predict_scores,
)

def run_centralized_experiment(dataset: PreparedDataset, config: ExperimentConfig, return_model: bool = False):
    """Run centralized baseline training on the full training split.

    With ``return_model=True`` the function returns ``(results, trained_model)`` (used by the transfer study).
    """
    start_time = time.perf_counter()

    train_mask = (dataset.split == 0)
    test_mask = (dataset.split == eval_split_id(config))
    if not test_mask.any():
        raise ValueError(f"No samples in evaluation split (eval_on='{config.dataset.eval_on}').")

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

    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.model.learning_rate, momentum=config.federated.client_momentum
    )
    class_weights = None
    if config.federated.class_weight == "balanced":
        class_weights = balanced_class_weights(y_train, n_classes)
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
    )

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
    label_map = dataset.manifest.get("label_map")
    test_preds, test_score = predict_scores(model, X_test)
    metrics = compute_classification_metrics(y_test, test_preds, label_map, y_score=test_score)

    results = {
        "experiment_id": config.experiment_id,
        "mode": "centralized",
        "dataset": config.dataset.name,
        "metrics": metrics,
        "eval_split": config.dataset.eval_on,
        "baseline_majority": majority_baseline(y_train, y_test, label_map),
        "class_weights": None if class_weights is None else [float(w) for w in class_weights],
        "dataset_summary": dataset_summary(dataset),
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
    return (results, model) if return_model else results
