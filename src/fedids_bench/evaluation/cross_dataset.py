from typing import Dict, Any
import numpy as np
import torch

from fedids_bench.models.base import IDSModel
from fedids_bench.data.schema import PreparedDataset
from fedids_bench.evaluation.classification import compute_classification_metrics

def align_features(X: np.ndarray, expected_features: int) -> np.ndarray:
    """
    Align feature dimensions of target dataset with expected model input dimensions.
    Pads with zeros if target features < expected_features, or truncates if target features > expected_features.
    """
    n_samples, current_features = X.shape
    if current_features == expected_features:
        return X
    elif current_features > expected_features:
        return X[:, :expected_features]
    else:
        padded = np.zeros((n_samples, expected_features), dtype=np.float32)
        padded[:, :current_features] = X
        return padded

def evaluate_cross_dataset(
    model: IDSModel,
    target_dataset: PreparedDataset
) -> Dict[str, Any]:
    """
    Evaluate trained model zero-shot on a different target dataset.
    """
    expected_dim = model.network[0].in_features
    X_aligned = align_features(target_dataset.X, expected_dim)

    preds = model.predict(X_aligned)

    metrics = compute_classification_metrics(
        y_true=target_dataset.y,
        y_pred=preds,
        label_map=target_dataset.manifest.get("label_map", {})
    )

    return {
        "target_dataset_name": target_dataset.manifest.get("dataset_name", "target"),
        "n_samples": len(target_dataset.X),
        "aligned_features": expected_dim,
        "metrics": metrics
    }
