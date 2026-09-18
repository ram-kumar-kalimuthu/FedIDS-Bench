import hashlib
import numpy as np
from typing import Dict, Any
from fedids_bench.data.schema import PreparedDataset
from fedids_bench.config import DatasetConfig

ATTACK_NAMES = {
    0: "Benign",
    1: "DoS",
    2: "PortScan",
    3: "BruteForce",
    4: "Botnet",
    5: "Infiltration"
}

def generate_synthetic_ids(config: DatasetConfig, seed: int = 42) -> PreparedDataset:
    """
    Generate synthetic Network Intrusion Detection Dataset.
    Class 0 is Benign (~70% of traffic).
    Classes 1..n_classes-1 are attack classes.
    """
    rng = np.random.default_rng(seed)
    
    n_samples = config.n_samples
    n_features = config.n_features
    n_classes = config.n_classes
    
    # Class probabilities: Benign 65%, remaining split among attack types
    attack_classes_count = n_classes - 1
    if attack_classes_count > 0:
        attack_prob = 0.35 / attack_classes_count
        probs = [0.65] + [attack_prob] * attack_classes_count
    else:
        probs = [1.0]
    probs = np.array(probs) / np.sum(probs)
    
    y = rng.choice(n_classes, size=n_samples, p=probs).astype(np.int64)
    attack_type = y.copy()
    
    # Feature generation per class (multivariate normal with distinct centers)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    
    benign_mean = rng.uniform(0.0, 1.0, size=n_features)
    benign_std = rng.uniform(0.5, 1.2, size=n_features)
    
    for c in range(n_classes):
        mask = (y == c)
        c_samples = np.sum(mask)
        if c_samples == 0:
            continue
        
        if c == 0:
            X[mask] = rng.normal(benign_mean, benign_std, size=(c_samples, n_features)).astype(np.float32)
        else:
            center_shift = rng.uniform(1.5 * c, 2.5 * c, size=n_features)
            attack_mean = benign_mean + center_shift
            attack_std = rng.uniform(0.8, 1.8, size=n_features)
            X[mask] = rng.normal(attack_mean, attack_std, size=(c_samples, n_features)).astype(np.float32)

    # Train / Test split stratification FIRST
    test_size = int(n_samples * config.test_fraction)
    split = np.zeros(n_samples, dtype=np.int8)  # 0 = train, 2 = test
    
    test_indices = []
    for c in range(n_classes):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        c_test_count = int(len(c_idx) * config.test_fraction)
        test_indices.extend(c_idx[:c_test_count])
        
    split[test_indices] = 2  # 2 = test set
    
    # CRITICAL-1 FIX: Fit Standard Scaler STRICTLY on training split (split == 0)
    train_mask = (split == 0)
    train_mean = X[train_mask].mean(axis=0, keepdims=True)
    train_std = X[train_mask].std(axis=0, keepdims=True) + 1e-6

    # Transform all features using ONLY training split scaler parameters
    X = (X - train_mean) / train_std
    
    label_map = {int(k): ATTACK_NAMES.get(int(k), f"Attack_{k}") for k in range(n_classes)}
    
    data_bytes = X.tobytes() + y.tobytes()
    dataset_hash = hashlib.sha256(data_bytes).hexdigest()[:16]
    
    manifest: Dict[str, Any] = {
        "dataset_name": "synthetic",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "label_map": label_map,
        "class_counts": {int(c): int(np.sum(y == c)) for c in range(n_classes)},
        "train_samples": int(np.sum(split == 0)),
        "test_samples": int(np.sum(split == 2)),
        "dataset_hash": dataset_hash,
        "scaler": {
            "mean": train_mean.flatten().tolist(),
            "std": train_std.flatten().tolist()
        },
        "seed": seed
    }
    
    return PreparedDataset(
        X=X.astype(np.float32),
        y=y,
        attack_type=attack_type,
        split=split,
        manifest=manifest
    )
