import torch
from fedids_bench.defenses.clipping import NormClippingDefense
from fedids_bench.defenses.trimmed_mean import TrimmedMeanDefense
from fedids_bench.defenses.median import MedianDefense
from fedids_bench.defenses.krum import KrumDefense
from fedids_bench.defenses.factory import get_defense
from fedids_bench.config import DefenseConfig

def test_norm_clipping_defense():
    global_w = {"w": torch.tensor([0.0, 0.0])}
    # Update with huge norm: delta = [100.0, 0.0], norm = 100.0
    huge_update = {"w": torch.tensor([100.0, 0.0])}
    normal_update = {"w": torch.tensor([1.0, 0.0])}

    defense = NormClippingDefense(clip_threshold=5.0)
    agg = defense.aggregate(global_w, [huge_update, normal_update], [10, 10], trust_client_sample_counts=True)
    
    # Clipped update = [5.0, 0.0]. Average with [1.0, 0.0] = [3.0, 0.0]
    assert torch.allclose(agg["w"], torch.tensor([3.0, 0.0]))

def test_trimmed_mean_defense():
    global_w = {"w": torch.tensor([0.0])}
    # 5 clients: 4 normal [1.0], 1 extreme outlier [1000.0]
    client_updates = [{"w": torch.tensor([1.0])} for _ in range(4)] + [{"w": torch.tensor([1000.0])}]
    
    # trim_ratio = 0.2 -> beta = 1 (trims 1 lowest and 1 highest)
    defense = TrimmedMeanDefense(trim_ratio=0.2)
    agg = defense.aggregate(global_w, client_updates, [10]*5)
    
    # Remaining 3 clients after trimming top 1000 and bottom 1 are [1.0, 1.0, 1.0]. Mean = 1.0
    assert torch.allclose(agg["w"], torch.tensor([1.0]))

def test_median_defense():
    global_w = {"w": torch.tensor([0.0])}
    client_updates = [
        {"w": torch.tensor([1.0])},
        {"w": torch.tensor([2.0])},
        {"w": torch.tensor([100.0])} # outlier
    ]
    defense = MedianDefense()
    agg = defense.aggregate(global_w, client_updates, [10]*3)
    # Median is 2.0
    assert torch.allclose(agg["w"], torch.tensor([2.0]))

def test_krum_defense():
    global_w = {"w": torch.tensor([0.0])}
    # 5 clients: 4 close to [1.0], 1 far away [100.0]
    client_updates = [
        {"w": torch.tensor([1.0])},
        {"w": torch.tensor([1.1])},
        {"w": torch.tensor([0.9])},
        {"w": torch.tensor([1.05])},
        {"w": torch.tensor([100.0])} # malicious outlier
    ]
    defense = KrumDefense(f=1, multi=False)
    agg = defense.aggregate(global_w, client_updates, [10]*5)
    # Krum should select one of the benign updates (not the 100.0 outlier)
    assert agg["w"].item() < 5.0

def test_defense_factory():
    cfg = DefenseConfig(type="trimmed_mean", trim_ratio=0.1)
    defense = get_defense(cfg)
    assert isinstance(defense, TrimmedMeanDefense)
