import numpy as np
import torch
from fedids_bench.attacks.label_flip import LabelFlipAttack
from fedids_bench.attacks.feature_poison import FeaturePoisonAttack
from fedids_bench.attacks.model_poison import ModelPoisonAttack
from fedids_bench.attacks.byzantine import ByzantineAttack
from fedids_bench.attacks.backdoor import BackdoorAttack
from fedids_bench.attacks.factory import get_attack
from fedids_bench.config import AttackConfig

def test_label_flip_attack():
    X = np.ones((10, 5), dtype=np.float32)
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    
    # Cyclic flip
    attack = LabelFlipAttack(target_label=None, n_classes=4)
    _, y_flipped = attack.apply_data_attack(X, y, client_id=0, seed=42)
    assert np.array_equal(y_flipped, (y + 1) % 4)

    # Targeted flip
    attack_target = LabelFlipAttack(target_label=0, n_classes=4)
    _, y_target = attack_target.apply_data_attack(X, y, client_id=0, seed=42)
    assert np.all(y_target == 0)

def test_feature_poison_attack():
    X = np.zeros((10, 5), dtype=np.float32)
    y = np.ones(10, dtype=int)
    attack = FeaturePoisonAttack(strength=2.0)
    X_poisoned, y_out = attack.apply_data_attack(X, y, client_id=0, seed=42)
    assert not np.array_equal(X, X_poisoned)
    assert np.array_equal(y, y_out)
    assert np.abs(np.mean(X_poisoned)) > 0.0

def test_model_poison_attack():
    global_w = {"w": torch.tensor([1.0, 2.0, 3.0])}
    local_w = {"w": torch.tensor([2.0, 4.0, 6.0])} # delta = [1, 2, 3]
    
    # Sign inversion (strength = -1)
    attack = ModelPoisonAttack(strength=-1.0)
    poisoned = attack.apply_model_attack(local_w, global_w, client_id=0, seed=42)
    # Expected: global - delta = [0, 0, 0]
    assert torch.allclose(poisoned["w"], torch.tensor([0.0, 0.0, 0.0]))

    # Scaling (strength = 10.0)
    attack_scale = ModelPoisonAttack(strength=10.0)
    poisoned_scale = attack_scale.apply_model_attack(local_w, global_w, client_id=0, seed=42)
    # Expected: global + 10 * delta = [11, 22, 33]
    assert torch.allclose(poisoned_scale["w"], torch.tensor([11.0, 22.0, 33.0]))

def test_byzantine_attack():
    global_w = {"w": torch.tensor([1.0, 1.0, 1.0])}
    local_w = {"w": torch.tensor([2.0, 2.0, 2.0])}
    attack = ByzantineAttack(strength=1.0)
    poisoned = attack.apply_model_attack(local_w, global_w, client_id=0, seed=42)
    assert "w" in poisoned
    assert not torch.allclose(poisoned["w"], local_w["w"])

def test_backdoor_attack():
    X = np.zeros((10, 5), dtype=np.float32)
    y = np.ones(10, dtype=int)
    attack = BackdoorAttack(target_label=0, trigger_fraction=0.5, trigger_val=9.0, trigger_dims=2)
    X_trig, y_trig = attack.apply_data_attack(X, y, client_id=0, seed=42)
    poisoned = np.where(y_trig == 0)[0]
    assert len(poisoned) == 5                                   # trigger_fraction of the rows, exactly
    assert np.all(X_trig[poisoned, :2] == 9.0)                  # trigger stamped on the first trigger_dims features
    assert np.all(X_trig[poisoned, 2:] == 0.0)                  # ... and nowhere else
    clean = np.setdiff1d(np.arange(10), poisoned)
    assert np.all(X_trig[clean] == 0.0) and np.all(y_trig[clean] == 1)   # untouched rows keep their label

    # a seed picks a random, reproducible subset (not "the first rows", which is one class when partitions are class-sorted).
    # 100 rows, not 10: with only C(10,5)=252 subsets two seeds can coincide by chance (seeds 42 and 43 do).
    X100, y100 = np.zeros((100, 5), dtype=np.float32), np.ones(100, dtype=int)
    first = attack.apply_data_attack(X100, y100, client_id=0, seed=42)[1]
    again = attack.apply_data_attack(X100, y100, client_id=0, seed=42)[1]
    other = attack.apply_data_attack(X100, y100, client_id=0, seed=43)[1]
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert (first == 0).sum() == (other == 0).sum() == 50
    assert not np.all(first[:50] == 0)                          # not simply "the first half"
    # without a seed the legacy behavior (first rows) is kept
    legacy_X, legacy_y = attack.inject_trigger(X, y)
    assert np.all(legacy_y[:5] == 0) and np.all(legacy_y[5:] == 1)

def test_attack_factory():
    cfg = AttackConfig(type="label_flip", target_label=2)
    attack = get_attack(cfg)
    assert isinstance(attack, LabelFlipAttack)
