from fedids_bench.config import AttackConfig
from fedids_bench.attacks.base import BaseAttack
from fedids_bench.attacks.label_flip import LabelFlipAttack
from fedids_bench.attacks.feature_poison import FeaturePoisonAttack
from fedids_bench.attacks.model_poison import ModelPoisonAttack
from fedids_bench.attacks.byzantine import ByzantineAttack
from fedids_bench.attacks.backdoor import BackdoorAttack

def get_attack(config: AttackConfig, n_classes: int = 4) -> BaseAttack:
    """Factory function to build Attack instance from AttackConfig."""
    attack_type = config.type.lower()
    if attack_type == "none":
        return BaseAttack()
    elif attack_type == "label_flip":
        return LabelFlipAttack(target_label=config.target_label, n_classes=n_classes)
    elif attack_type == "feature_poison":
        return FeaturePoisonAttack(strength=config.strength)
    elif attack_type == "model_poison":
        return ModelPoisonAttack(strength=-config.strength if config.strength > 0 else config.strength)
    elif attack_type == "byzantine":
        return ByzantineAttack(strength=config.strength)
    elif attack_type == "backdoor":
        target = config.target_label if config.target_label is not None else 0
        return BackdoorAttack(target_label=target)
    else:
        raise ValueError(f"Unknown attack type: '{config.type}'")
