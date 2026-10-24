"""Helpers shared by the federated and the centralized runner."""
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from fedids_bench.config import ExperimentConfig
from fedids_bench.evaluation.classification import compute_classification_metrics


def eval_split_id(config: ExperimentConfig) -> int:
    """Split used for the reported metrics: 2 = test (final numbers), 1 = validation (tuning)."""
    return 1 if config.dataset.eval_on == "val" else 2


def balanced_class_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency CE weights, w_c = N / (C * n_c); computed from the TRAIN split only."""
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    return (counts.sum() / (n_classes * np.maximum(counts, 1.0))).astype(np.float32)


def predict_scores(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hard predictions and the attack score 1 - P(benign) (used for a real AUROC)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
    return probs.argmax(axis=1), 1.0 - probs[:, 0]


def evaluate_model_on_split(model, X: np.ndarray, y: np.ndarray, label_map: dict) -> dict:
    if len(X) == 0:
        return {}
    preds, score = predict_scores(model, X)
    return compute_classification_metrics(y, preds, label_map, y_score=score)


def majority_baseline(y_train: np.ndarray, y_eval: np.ndarray, label_map: Optional[dict] = None) -> Dict:
    """Always predict the TRAIN-majority class on the evaluation split."""
    cls = int(np.bincount(y_train).argmax())
    preds = np.full_like(y_eval, cls)
    score = np.full(len(y_eval), 0.0 if cls == 0 else 1.0)
    out = compute_classification_metrics(y_eval, preds, label_map, y_score=score)
    out["majority_class"] = cls
    return out


def dataset_summary(dataset) -> Dict:
    keep = ("n_samples", "n_features", "feature_set", "split_counts", "attack_prevalence",
            "canonical_fields_found", "n_duplicates_dropped", "source_file")
    return {k: dataset.manifest[k] for k in keep if k in dataset.manifest}
