#!/usr/bin/env python3
import sys
import os

# Ensure package is on python path if run from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fedids_bench.config import load_config
from fedids_bench.data.synthetic import generate_synthetic_ids

def main():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "smoke.yaml"))
    cfg = load_config(config_path)
    dataset = generate_synthetic_ids(cfg.dataset, seed=cfg.seed)
    print("Synthetic dataset generation complete!")
    print(f"Dataset Hash: {dataset.manifest['dataset_hash']}")
    print(f"Total samples: {dataset.manifest['n_samples']}")
    print(f"Class distribution: {dataset.manifest['class_counts']}")

if __name__ == "__main__":
    main()
