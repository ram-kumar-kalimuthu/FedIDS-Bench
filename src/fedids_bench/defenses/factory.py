from fedids_bench.config import DefenseConfig
from fedids_bench.defenses.base import BaseDefense
from fedids_bench.defenses.clipping import NormClippingDefense
from fedids_bench.defenses.trimmed_mean import TrimmedMeanDefense
from fedids_bench.defenses.median import MedianDefense
from fedids_bench.defenses.krum import KrumDefense

def get_defense(config: DefenseConfig) -> BaseDefense:
    """Factory function to instantiate robust aggregation defenses."""
    def_type = config.type.lower()
    if def_type == "none":
        return None
    elif def_type == "clipping":
        return NormClippingDefense(clip_threshold=config.clip_threshold)
    elif def_type == "trimmed_mean":
        return TrimmedMeanDefense(trim_ratio=config.trim_ratio)
    elif def_type == "median":
        return MedianDefense()
    elif def_type == "krum":
        return KrumDefense(f=config.krum_f, multi=config.krum_multi, m=config.krum_m)
    else:
        raise ValueError(f"Unknown defense type: '{config.type}'")
