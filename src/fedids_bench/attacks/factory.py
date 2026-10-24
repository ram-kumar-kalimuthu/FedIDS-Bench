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
        return BackdoorAttack(target_label=target, trigger_fraction=config.trigger_fraction,
                              trigger_val=config.trigger_value, trigger_dims=config.trigger_dims)
    elif attack_type == "sample_inflation":
        # The lie is in the REPORTED sample count (applied by the server).  The malicious client's
        # update is honest unless it is combined with another attack via `combine_with`.
        if config.combine_with == "none":
            return BaseAttack()
        combined = config.model_copy(update={"type": config.combine_with})
        return get_attack(combined, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown attack type: '{config.type}'")
