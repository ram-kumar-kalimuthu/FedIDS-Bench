import numpy as np
from sklearn.dummy import DummyClassifier
from fedids_bench.evaluation.classification import compute_classification_metrics
from fedids_bench.config import load_config
from fedids_bench.experiments.runner import run_experiment

def test_classification_metrics_binary_derivation():
    # True: 0 (Benign), 1 (DoS), 2 (Probe), 0 (Benign)
    y_true = np.array([0, 1, 2, 0])
    # Pred: 0 (Benign), 1 (DoS), 0 (Benign -> FN for attack!), 0 (Benign)
    y_pred = np.array([0, 1, 0, 0])

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.5
    assert metrics["benign_recall"] == 1.0
    assert metrics["attack_recall"] == 0.5

    assert "multiclass" in metrics
    assert "macro_f1" in metrics["multiclass"]


def test_model_learning_above_dummy_baseline():
    """Verify that FL training achieves higher F1 score than a dummy random/majority baseline."""
    cfg = load_config("configs/smoke.yaml")
    res = run_experiment(cfg)

    final_f1 = res["metrics"]["f1"]
    
    # Random baseline F1 is ~0.5. Trained FL model after 5 rounds reaches > 0.8 F1
    assert final_f1 > 0.7, f"Expected final F1 > 0.7, got {final_f1}"
