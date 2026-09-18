import os
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

def evaluate_baselines():
    cfg = load_config("configs/smoke.yaml")
    loader = get_dataset_loader(cfg.dataset.name)
    dataset = loader.load(cfg.dataset, seed=cfg.seed)

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
    dummy_rand = DummyClassifier(strategy="uniform", random_state=42)
    dummy_rand.fit(X_train, y_train)
    preds_rand = dummy_rand.predict(X_test)
    metrics_rand = compute_classification_metrics(y_test, preds_rand)

    print("=" * 60)
    print("BASELINE COMPARISONS ON SYNTHETIC DATASET (smoke.yaml)")
    print("=" * 60)
    print(f"Majority Class Baseline -> Binary F1: {metrics_mf['f1']:.4f}, Macro F1: {metrics_mf['multiclass']['macro_f1']:.4f}")
    print(f"Uniform Random Baseline -> Binary F1: {metrics_rand['f1']:.4f}, Macro F1: {metrics_rand['multiclass']['macro_f1']:.4f}")

    # Centralized Baseline (10 epochs)
    cfg_cent = load_config("configs/smoke.yaml")
    cfg_cent.mode = "centralized"
    cfg_cent.federated.rounds = 10
    cfg_cent.federated.local_epochs = 1
    res_cent = run_experiment(cfg_cent)
    print(f"Centralized (10 epochs) -> Binary F1: {res_cent['metrics']['f1']:.4f}, Macro F1: {res_cent['metrics']['multiclass']['macro_f1']:.4f}")

    # Federated FL (10 rounds)
    cfg_fl10 = load_config("configs/smoke.yaml")
    cfg_fl10.federated.rounds = 10
    res_fl10 = run_experiment(cfg_fl10)
    print(f"Federated FL (10 rounds) -> Binary F1: {res_fl10['metrics']['f1']:.4f}, Macro F1: {res_fl10['metrics']['multiclass']['macro_f1']:.4f}")

    # Federated FL (30 rounds)
    cfg_fl30 = load_config("configs/smoke.yaml")
    cfg_fl30.federated.rounds = 30
    res_fl30 = run_experiment(cfg_fl30)
    print(f"Federated FL (30 rounds) -> Binary F1: {res_fl30['metrics']['f1']:.4f}, Macro F1: {res_fl30['metrics']['multiclass']['macro_f1']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_baselines()
