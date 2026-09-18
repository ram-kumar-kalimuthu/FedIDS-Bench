import os
import sys
import json
import argparse
import pandas as pd

def aggregate_results(results_dir: str = "results", output_csv: str = "results_summary.csv"):
    """
    Traverse results_dir for all result.json files and compile them into a summary CSV.
    """
    rows = []
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' does not exist.")
        return

    for root, _, files in os.walk(results_dir):
        if "result.json" in files:
            path = os.path.join(root, "result.json")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    res = json.load(f)
                
                cfg = res.get("config", {})
                final_round = res.get("round_logs", [])[-1] if res.get("round_logs") else {}

                row = {
                    "experiment_id": cfg.get("experiment_id"),
                    "dataset": cfg.get("dataset", {}).get("name"),
                    "algorithm": cfg.get("federated", {}).get("algorithm"),
                    "attack_type": cfg.get("attack", {}).get("type"),
                    "defense_type": cfg.get("defense", {}).get("type"),
                    "seed": cfg.get("seed"),
                    "final_global_test_f1": final_round.get("tier1_global_test_f1"),
                    "final_global_test_acc": final_round.get("tier1_global_test_acc"),
                    "total_comm_bytes": res.get("communication_cost", {}).get("total_bytes"),
                    "total_wall_clock_sec": res.get("resource_cost", {}).get("total_wall_clock_sec")
                }
                rows.append(row)
            except Exception as e:
                print(f"Failed to parse {path}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"Successfully aggregated {len(rows)} experiment results into '{output_csv}'.")
    else:
        print("No result.json files found.")

def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results into a CSV summary.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_csv", type=str, default="results_summary.csv")
    args = parser.parse_args()

    aggregate_results(args.results_dir, args.output_csv)

if __name__ == "__main__":
    main()
