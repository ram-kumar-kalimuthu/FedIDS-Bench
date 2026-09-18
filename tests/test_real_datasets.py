import pytest
from fedids_bench.config import DatasetConfig
from fedids_bench.data.loaders import get_dataset_loader

REAL_DATASETS = ["unsw_nb15", "cicids2017", "cse_cic_ids2018", "iot23", "ton_iot"]

@pytest.mark.parametrize("dataset_name", REAL_DATASETS)
def test_real_dataset_loader_properties(dataset_name):
    """Verify each real dataset loader loads real data, non-synthetic, with valid schema."""
    loader = get_dataset_loader(dataset_name)
    cfg = DatasetConfig(name=dataset_name, n_samples=250, test_fraction=0.2)
    ds = loader.load(cfg, seed=42)

    # Validate shape and samples
    assert ds.X.shape[0] == 250, f"{dataset_name} expected 250 samples, got {ds.X.shape[0]}"
    assert ds.X.shape[1] >= 10, f"{dataset_name} expected >=10 features, got {ds.X.shape[1]}"
    assert len(ds.y) == 250
    assert len(ds.attack_type) == 250
    
    # Train / test split checks
    assert (ds.split == 0).sum() == 200, f"{dataset_name} train split count mismatch"
    assert (ds.split == 2).sum() == 50, f"{dataset_name} test split count mismatch"

    # Multi-class representation
    label_map = ds.manifest["label_map"]
    assert len(label_map) >= 2, f"{dataset_name} expected at least 2 classes, got {len(label_map)}"
    
    # Verify non-synthetic (manifest dataset_name should match real loader)
    assert ds.manifest["dataset_name"] == dataset_name, f"Expected {dataset_name}, got {ds.manifest['dataset_name']}"
    assert len(ds.manifest["feature_names"]) == ds.X.shape[1]
