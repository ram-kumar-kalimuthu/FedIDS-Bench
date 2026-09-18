import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fedids_bench.config import DatasetConfig
from fedids_bench.data.loaders import LOADER_REGISTRY, get_dataset_loader

def main():
    parser = argparse.ArgumentParser(description="Prepare or generate datasets for FedIDS-Bench.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=list(LOADER_REGISTRY.keys()))
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_features", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print(f"Preparing dataset '{args.dataset}' in {args.output_dir}...")
    cfg = DatasetConfig(
        name=args.dataset,
        n_samples=args.n_samples,
        n_features=args.n_features,
        data_dir=args.output_dir
    )

    loader = get_dataset_loader(args.dataset)
    dataset = loader.load(cfg, seed=args.seed)

    print(f"Dataset prepared successfully! Total samples: {len(dataset.X)}, Features: {dataset.X.shape[1]}")
    print(f"Label map: {dataset.manifest.get('label_map', {})}")

if __name__ == "__main__":
    main()
