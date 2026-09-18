import argparse
import os
import sys
import json
import yaml
from fedids_bench.config import load_config
from fedids_bench.experiments.runner import run_experiment
from fedids_bench.data.synthetic import generate_synthetic_ids

def export_jobs(output_dir: str = "jobs"):
    """Export a matrix of benchmark experiment configs for parallel execution."""
    os.makedirs(os.path.join(output_dir, "configs"), exist_ok=True)
    
    datasets = ["synthetic", "unsw_nb15", "cicids2017"]
    algorithms = ["fedavg", "fedprox", "fedadam"]
    attacks = [
        {"type": "none"},
        {"type": "label_flip", "malicious_fraction": 0.2},
        {"type": "model_poison", "malicious_fraction": 0.2, "strength": -1.0},
        {"type": "byzantine", "malicious_fraction": 0.2}
    ]
    defenses = ["none", "trimmed_mean", "median", "krum"]
    seeds = [42, 100]

    job_list = []
    job_id = 0

    for ds in datasets:
        for alg in algorithms:
            for atk in attacks:
                for def_name in defenses:
                    for seed in seeds:
                        job_name = f"job_{job_id:04d}_{ds}_{alg}_{atk['type']}_{def_name}_s{seed}"
                        cfg_dict = {
                            "experiment_id": job_name,
                            "seed": seed,
                            "dataset": {"name": ds, "n_samples": 1000, "n_features": 20},
                            "partitioning": {"strategy": "dirichlet", "num_clients": 5, "alpha": 0.5},
                            "model": {"architecture": "mlp", "learning_rate": 0.1},
                            "federated": {"algorithm": alg, "rounds": 10, "local_epochs": 1},
                            "attack": atk,
                            "defense": {"type": def_name},
                            "output_dir": f"results/{job_name}"
                        }
                        cfg_path = os.path.join(output_dir, "configs", f"{job_name}.yaml")
                        with open(cfg_path, "w", encoding="utf-8") as f:
                            yaml.dump(cfg_dict, f, default_flow_style=False)
                        
                        job_list.append({"job_id": job_id, "job_name": job_name, "config_path": cfg_path})
                        job_id += 1

    manifest_path = os.path.join(output_dir, "jobs_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(job_list, f, indent=2)

    script_path = os.path.join(output_dir, "run_all.sh")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Batch execution script for FedIDS-Bench jobs\n\n")
        for j in job_list:
            f.write(f"fedids-bench run --config {j['config_path']}\n")

    print(f"Exported {len(job_list)} experiment jobs to '{output_dir}'.")
    print(f"Job manifest saved to '{manifest_path}'.")

def main():
    parser = argparse.ArgumentParser(
        prog="fedids-bench",
        description="FedIDS-Bench: Standardized Benchmark for Federated Intrusion Detection Systems"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run a federated IDS benchmark experiment")
    run_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML configuration file")

    # 'generate-synthetic' subcommand
    synth_parser = subparsers.add_parser("generate-synthetic", help="Generate synthetic IDS dataset")
    synth_parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")

    # 'export-jobs' subcommand
    export_parser = subparsers.add_parser("export-jobs", help="Export experiment matrix configuration files")
    export_parser.add_argument("--output_dir", type=str, default="jobs", help="Directory to export job configs")

    args = parser.parse_args()

    if args.command == "run":
        cfg = load_config(args.config)
        run_experiment(cfg)
    elif args.command == "generate-synthetic":
        cfg = load_config(args.config)
        dataset = generate_synthetic_ids(cfg.dataset, seed=cfg.seed)
        print(f"Successfully generated synthetic dataset: {dataset.manifest['n_samples']} samples, {dataset.manifest['n_features']} features.")
    elif args.command == "export-jobs":
        export_jobs(output_dir=args.output_dir)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
