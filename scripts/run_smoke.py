#!/usr/bin/env python3
import sys
import os

# Ensure package is on python path if run from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fedids_bench.config import load_config
from fedids_bench.experiments.runner import run_experiment

def main():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "smoke.yaml"))
    print(f"Loading smoke test config from: {config_path}")
    cfg = load_config(config_path)
    results = run_experiment(cfg)
    
    print("\n" + "="*50)
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Experiment ID: {results['experiment_id']}")
    print(f"Global Test F1 Score: {results['metrics']['f1']:.4f}")
    print(f"Multiclass Macro F1:  {results['metrics']['multiclass']['macro_f1']:.4f}")
    print(f"Communication Bytes: {results['communication']['total_communication_bytes']:,} bytes")
    print(f"Client Training Time: {results['resource_cost']['client_training_time_sec']} sec")
    print(f"Server Agg Time:      {results['resource_cost']['server_aggregation_time_sec']} sec")
    print(f"Results Directory:    {cfg.output_dir}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
