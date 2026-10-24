import numpy as np
from fedids_bench.config import PartitioningConfig
from fedids_bench.data.partition import get_partitioner
from fedids_bench.data.schema import PreparedDataset


def _ds(n=1200, n_attack_cats=4, seed=0):
    rng = np.random.default_rng(seed)
    cats = rng.integers(0, n_attack_cats + 1, n)             # 0 = benign, 1..K attack categories
    y = (cats > 0).astype(np.int64)
    split = np.zeros(n, dtype=np.int8)
    split[rng.random(n) < 0.2] = 2
    return PreparedDataset(X=rng.standard_normal((n, 3)).astype(np.float32), y=y, attack_type=cats.astype(np.int64),
                           split=split, manifest={"dataset_name": "t"})


def test_single_class_clients_when_max_classes_is_one():
    ds = _ds()
    cfg = PartitioningConfig(strategy="single_domain", num_clients=10, heldout_client_fraction=0.0, max_classes=1)
    res = get_partitioner("single_domain").partition(ds, cfg, seed=1)
    for cid, idx in res.client_indices.items():
        if len(idx):
            assert len(np.unique(ds.attack_type[idx])) == 1
    covered = np.concatenate([i for i in res.client_indices.values() if len(i)])
    assert len(covered) == int((ds.split == 0).sum()) and len(set(covered)) == len(covered)   # exact cover, no overlap


def test_each_client_holds_benign_plus_at_most_L_minus_1_attack_categories():
    ds = _ds()
    cfg = PartitioningConfig(strategy="single_domain", num_clients=10, heldout_client_fraction=0.0, max_classes=2)
    res = get_partitioner("single_domain").partition(ds, cfg, seed=2)
    seen_attack_cats = set()
    for cid, idx in res.client_indices.items():
        cats = set(np.unique(ds.attack_type[idx]).tolist())
        assert 0 in cats and len(cats) <= 2
        seen_attack_cats |= cats - {0}
    assert seen_attack_cats == {1, 2, 3, 4}                   # every attack category is held by someone


def test_label_skew_is_really_a_stratified_iid_split():
    """Documents the finding: 'label_skew' gives every client every class (it does not skew labels)."""
    ds = _ds()
    cfg = PartitioningConfig(strategy="label_skew", num_clients=5, heldout_client_fraction=0.0)
    res = get_partitioner("label_skew").partition(ds, cfg, seed=3)
    for idx in res.client_indices.values():
        assert set(np.unique(ds.y[idx]).tolist()) == {0, 1}
