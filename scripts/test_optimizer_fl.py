import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fedids_bench.config import load_config
from fedids_bench.data.loaders import get_dataset_loader
from fedids_bench.models.mlp import SmallMLP
from fedids_bench.federated.trainer import NumPyDataset
from fedids_bench.evaluation.classification import compute_classification_metrics
from fedids_bench.experiments.runner import run_experiment

def test_fl_optimizer_comparison():
    print("=" * 60)
    print("TESTING FL CONVERGENCE WITH DIFFERENT OPTIMIZERS / LEARNING RATES")
    print("=" * 60)

    # 1. Standard SGD lr=0.01 (current smoke default)
    cfg_sgd = load_config("configs/smoke.yaml")
    cfg_sgd.federated.rounds = 10
    cfg_sgd.model.learning_rate = 0.01
    res_sgd = run_experiment(cfg_sgd)
    f1_sgd = [r["tier1_global_test_f1"] for r in res_sgd["round_logs"]]
    print(f"SGD (lr=0.01) Round F1s: {f1_sgd}")

    # 2. SGD with lr=0.1
    cfg_sgd_01 = load_config("configs/smoke.yaml")
    cfg_sgd_01.federated.rounds = 10
    cfg_sgd_01.model.learning_rate = 0.1
    res_sgd_01 = run_experiment(cfg_sgd_01)
    f1_sgd_01 = [r["tier1_global_test_f1"] for r in res_sgd_01["round_logs"]]
    print(f"SGD (lr=0.1) Round F1s:  {f1_sgd_01}")

if __name__ == "__main__":
    test_fl_optimizer_comparison()
