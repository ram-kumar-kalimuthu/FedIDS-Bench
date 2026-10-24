"""
Every registered loader produces the harmonized schema.

* Always runs on small stand-in files that copy the REAL header style of each corpus (see test_features_loaders.py).
* Set FEDIDS_DATA_DIR=/path/to/corpora to run the same checks on the real data as well (a missing corpus is skipped).
"""
import os

import numpy as np
import pytest

from fedids_bench.config import DatasetConfig
from fedids_bench.data.loaders import LOADER_REGISTRY, get_dataset_loader
from tests.test_features_loaders import WRITERS

REAL_DIR = os.environ.get("FEDIDS_DATA_DIR")
SOURCES = ["fixture"] + (["real"] if REAL_DIR else [])


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("standin_corpora")
    for writer, fname in WRITERS.values():
        writer(d / fname, n=800)
    return str(d)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("name", list(LOADER_REGISTRY))
def test_all_dataset_loaders(name, source, fixture_dir):
    data_dir = fixture_dir if source == "fixture" else REAL_DIR
    cfg = DatasetConfig(name=name, n_samples=200, n_features=10, data_dir=data_dir)
    loader = get_dataset_loader(name)
    try:
        dataset = loader.load(cfg, seed=42)
    except FileNotFoundError:
        if source == "real":
            pytest.skip(f"{name}: not found under FEDIDS_DATA_DIR")
        raise
    assert dataset.X.shape[0] == 200
    assert dataset.X.shape[1] >= 5
    assert len(dataset.y) == 200
    assert (dataset.split == 0).any()      # train split present
    assert (dataset.split != 0).any()      # held-out split present
    assert np.isfinite(dataset.X).all()


def test_missing_corpus_error_points_to_the_docs(tmp_path):
    with pytest.raises(FileNotFoundError, match="docs/DATA.md"):
        get_dataset_loader("unsw_nb15").load(DatasetConfig(name="unsw_nb15", n_samples=50, data_dir=str(tmp_path)), seed=0)
