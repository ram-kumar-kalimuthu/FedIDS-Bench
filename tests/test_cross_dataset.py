import numpy as np
from fedids_bench.models.mlp import SmallMLP
from fedids_bench.data.schema import PreparedDataset
from fedids_bench.evaluation.cross_dataset import evaluate_cross_dataset, align_features

def test_feature_alignment():
    X_large = np.ones((10, 30), dtype=np.float32)
    aligned_trunc = align_features(X_large, expected_features=20)
    assert aligned_trunc.shape == (10, 20)

    X_small = np.ones((10, 10), dtype=np.float32)
    aligned_pad = align_features(X_small, expected_features=20)
    assert aligned_pad.shape == (10, 20)
    assert np.all(aligned_pad[:, :10] == 1.0)
    assert np.all(aligned_pad[:, 10:] == 0.0)

def test_evaluate_cross_dataset():
    model = SmallMLP(n_features=20, n_classes=4)
    target_dataset = PreparedDataset(
        X=np.random.randn(50, 15).astype(np.float32),
        y=np.random.randint(0, 4, size=50, dtype=np.int64),
        attack_type=np.zeros(50, dtype=np.int64),
        split=np.zeros(50, dtype=np.int8),
        manifest={"dataset_name": "target_test", "label_map": {"0": "b", "1": "a1", "2": "a2", "3": "a3"}}
    )

    res = evaluate_cross_dataset(model, target_dataset)
    assert res["aligned_features"] == 20
    assert "metrics" in res
    assert "accuracy" in res["metrics"]
