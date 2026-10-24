"""
Properties of the five real corpora.  These need the raw files, which are NOT shipped with the repository
(see docs/DATA.md): set FEDIDS_DATA_DIR=/path/to/corpora, otherwise every case is skipped.  The same properties are
checked on stand-in files in test_loaders.py / test_features_loaders.py, so a green suite without data is not silent about them.
"""
import os

import pytest

from fedids_bench.config import DatasetConfig
from fedids_bench.data.loaders import get_dataset_loader

REAL_DATASETS = ["unsw_nb15", "cicids2017", "cse_cic_ids2018", "iot23", "ton_iot"]
REAL_DIR = os.environ.get("FEDIDS_DATA_DIR") or ("data/raw" if os.path.isdir("data/raw") else None)


@pytest.mark.parametrize("dataset_name", REAL_DATASETS)
def test_real_dataset_loader_properties(dataset_name):
    """Verify each real dataset loader loads real data, non-synthetic, with a valid schema."""
    if not REAL_DIR:
        pytest.skip("set FEDIDS_DATA_DIR to run the checks on the real corpora (docs/DATA.md)")
    loader = get_dataset_loader(dataset_name)
    cfg = DatasetConfig(name=dataset_name, n_samples=250, test_fraction=0.2, data_dir=REAL_DIR)
    try:
        ds = loader.load(cfg, seed=42)
    except FileNotFoundError:
        pytest.skip(f"{dataset_name}: not found under {REAL_DIR}")

    assert ds.X.shape[0] == 250, f"{dataset_name} expected 250 samples, got {ds.X.shape[0]}"
    assert ds.X.shape[1] >= 5, f"{dataset_name} expected >=5 features, got {ds.X.shape[1]}"
    assert len(ds.y) == 250
    assert len(ds.attack_type) == 250
    assert (ds.split == 0).sum() == 200, f"{dataset_name} train split count mismatch"
    assert (ds.split == 2).sum() == 50, f"{dataset_name} test split count mismatch"

    label_map = ds.manifest["label_map"]
    assert len(label_map) >= 2, f"{dataset_name} expected at least 2 classes, got {len(label_map)}"
    assert ds.manifest["dataset_name"] == dataset_name
    assert len(ds.manifest["feature_names"]) == ds.X.shape[1]
    # the whole point of the harmonized core: no core field may be silently absent from a real corpus
    assert ds.manifest["canonical_fields_found"]["duration"], f"{dataset_name}: duration column not found"
    assert ds.manifest["canonical_fields_found"]["fwd_bytes"], f"{dataset_name}: forward-bytes column not found"
