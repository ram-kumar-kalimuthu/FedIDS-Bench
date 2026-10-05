import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import sys
import numpy as np
from sklearn.dummy import DummyClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fedids_bench.config import load_config
from fedids_bench.data.loaders import get_dataset_loader
from fedids_bench.evaluation.classification import compute_classification_metrics
from fedids_bench.experiments.runner import run_experiment

DATASETS = ["unsw_nb15", "cicids2017", "cse_cic_ids2018", "iot23", "ton_iot"]

def evaluate_baselines():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_file = os.path.join(project_root, "configs", "standard.yaml")
    results = {}
    
    for ds_name in DATASETS:
        print("=" * 60)
        print(f"EVALUATING BASELINES ON DATASET: {ds_name}")
        print("=" * 60)
        
        cfg = load_config(config_file)
        cfg.dataset.name = ds_name
        cfg.dataset.n_features = 5
        
        try:
            loader = get_dataset_loader(cfg.dataset.name)
            dataset = loader.load(cfg.dataset, seed=cfg.seed)
        except Exception as e:
            print(f"Skipping {ds_name}: {e}")
            continue

        train_mask = (dataset.split == 0)
        test_mask = (dataset.split == 2)

        X_train, y_train = dataset.X[train_mask], dataset.y[train_mask]
        X_test, y_test = dataset.X[test_mask], dataset.y[test_mask]

        # Dummy Most Frequent
        dummy_mf = DummyClassifier(strategy="most_frequent")
        dummy_mf.fit(X_train, y_train)
        preds_mf = dummy_mf.predict(X_test)
        metrics_mf = compute_classification_metrics(y_test, preds_mf)

        # Dummy Stratified / Uniform
        dummy_rand = DummyClassifier(strategy="uniform", random_state=cfg.seed)
        dummy_rand.fit(X_train, y_train)
        preds_rand = dummy_rand.predict(X_test)
        metrics_rand = compute_classification_metrics(y_test, preds_rand)

        print(f"Majority Class Baseline -> Binary F1: {metrics_mf.get('f1', 0):.4f}, Macro F1: {metrics_mf.get('multiclass', {}).get('macro_f1', 0):.4f}")
        print(f"Uniform Random Baseline -> Binary F1: {metrics_rand.get('f1', 0):.4f}, Macro F1: {metrics_rand.get('multiclass', {}).get('macro_f1', 0):.4f}")

        # Centralized Baseline (10 epochs)
        cfg_cent = load_config(config_file)
        cfg_cent.mode = "centralized"
        cfg_cent.dataset.name = ds_name
        cfg_cent.dataset.n_features = 5
        cfg_cent.federated.rounds = 10
        cfg_cent.federated.local_epochs = 1
        cfg_cent.output_dir = os.path.join(project_root, "results", f"centralized_{ds_name}")
        res_cent = run_experiment(cfg_cent)
        
        print(f"Centralized (10 epochs) -> Binary F1: {res_cent['metrics']['f1']:.4f}, Macro F1: {res_cent['metrics']['multiclass']['macro_f1']:.4f}")
        
        results[ds_name] = {
            "majority": metrics_mf,
            "random": metrics_rand,
            "centralized": res_cent['metrics'],
            "data_source": ds_name
        }
        
    out_dir = os.path.join(project_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "baseline_metrics.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Baseline evaluation complete. Results saved to {out_file}")

if __name__ == "__main__":
    evaluate_baselines()
