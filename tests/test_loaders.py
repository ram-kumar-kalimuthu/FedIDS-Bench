from fedids_bench.config import DatasetConfig
from fedids_bench.data.loaders import get_dataset_loader, LOADER_REGISTRY

def test_all_dataset_loaders():
    cfg = DatasetConfig(n_samples=200, n_features=10)
    for name in LOADER_REGISTRY.keys():
        loader = get_dataset_loader(name)
        dataset = loader.load(cfg, seed=42)
        assert dataset.X.shape[0] == 200
        assert dataset.X.shape[1] >= 10
        assert len(dataset.y) == 200
        assert (dataset.split == 0).any()  # train set split
        assert (dataset.split != 0).any()  # test set split
