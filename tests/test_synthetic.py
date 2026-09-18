import numpy as np
from fedids_bench.config import DatasetConfig
from fedids_bench.data.synthetic import generate_synthetic_ids

def test_generate_synthetic_ids():
    cfg = DatasetConfig(n_samples=500, n_features=15, n_classes=3, test_fraction=0.2)
    ds = generate_synthetic_ids(cfg, seed=42)
    
    assert ds.X.shape == (500, 15)
    assert ds.y.shape == (500,)
    assert ds.X.dtype == np.float32
    assert ds.y.dtype == np.int64
    assert np.sum(ds.split == 2) == 100  # 20% of 500
    assert "dataset_hash" in ds.manifest

def test_synthetic_reproducibility():
    cfg = DatasetConfig(n_samples=200, n_features=10, n_classes=2)
    ds1 = generate_synthetic_ids(cfg, seed=123)
    ds2 = generate_synthetic_ids(cfg, seed=123)
    
    np.testing.assert_array_equal(ds1.X, ds2.X)
    np.testing.assert_array_equal(ds1.y, ds2.y)
    assert ds1.manifest["dataset_hash"] == ds2.manifest["dataset_hash"]
