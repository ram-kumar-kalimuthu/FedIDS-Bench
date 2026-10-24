from typing import Dict, Any, Optional
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

def rescale_with_source(target_dataset: PreparedDataset, source_manifest: Dict[str, Any]) -> np.ndarray:
    """
    Zero-shot rescaling: standardize the target's UN-scaled harmonized features with the SOURCE
    corpus's train-split statistics (no target statistics are used).
    """
    if target_dataset.X_feat is None:
        raise ValueError("Target dataset carries no un-scaled features (X_feat); cannot rescale.")
    mean = np.asarray(source_manifest["scaler_mean"], dtype=np.float32)
    scale = np.asarray(source_manifest["scaler_scale"], dtype=np.float32)
    return ((target_dataset.X_feat - mean) / scale).astype(np.float32)


def evaluate_cross_dataset(
    model: IDSModel,
    target_dataset: PreparedDataset,
    source_manifest: Optional[Dict[str, Any]] = None,
    scaler: str = "target",
    split: Optional[int] = None,
    permute_columns_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained model zero-shot on a different target dataset.

    scaler="target": the target keeps its own train-split standardization (default, legacy).
    scaler="source": the target is re-standardized with the SOURCE statistics (strict zero-shot).
    split: evaluate on this split of the target (2 = its test split, as the transfer study does so that the
           diagonal and off-diagonal cells use identically sized held-out sets); None (default, legacy) = all rows.
    permute_columns_seed: CONTROL.  When set, the feature columns of the target are shuffled with this seed before
           the model sees them.  If the transfer score hardly drops, the model is not using the harmonized feature
           semantics (it is reacting to class priors or scale), and the cell says nothing about shared structure.
    """
    if scaler == "source":
        if source_manifest is None:
            raise ValueError("scaler='source' requires source_manifest.")
        X_all = rescale_with_source(target_dataset, source_manifest)
    else:
        X_all = target_dataset.X

    mask = np.ones(len(X_all), dtype=bool) if split is None else (target_dataset.split == split)
    X_t, y_t = X_all[mask], target_dataset.y[mask]

    expected_dim = model.network[0].in_features
    X_aligned = align_features(X_t, expected_dim)
    if permute_columns_seed is not None:
        X_aligned = X_aligned[:, np.random.default_rng(int(permute_columns_seed)).permutation(expected_dim)]

    probs = model.predict_proba(X_aligned)
    preds = np.argmax(probs, axis=-1)
    score = 1.0 - probs[:, 0]

    metrics = compute_classification_metrics(
        y_true=y_t,
        y_pred=preds,
        label_map=target_dataset.manifest.get("label_map", {}),
        y_score=score,
    )

    return {
        "target_dataset_name": target_dataset.manifest.get("dataset_name", "target"),
        "n_samples": int(len(X_t)),
        "aligned_features": expected_dim,
        "scaler": scaler,
        "columns_permuted": permute_columns_seed is not None,
        "metrics": metrics
    }
