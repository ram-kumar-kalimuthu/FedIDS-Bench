from fedids_bench.config import load_config
from fedids_bench.experiments.runner import run_experiment

def test_fl_under_model_poisoning_attack():
    # Load smoke config
    cfg = load_config("configs/smoke.yaml")
    cfg.federated.rounds = 3
    cfg.model.learning_rate = 0.1
    cfg.attack.type = "model_poison"
    cfg.attack.malicious_fraction = 0.2
    cfg.attack.strength = -1.0 # sign-flip attack

    # Run without defense
    cfg.defense.type = "none"
    res_no_def = run_experiment(cfg)
    assert len(res_no_def["round_logs"]) == 3

    # Run with Trimmed Mean defense
    cfg.defense.type = "trimmed_mean"
    cfg.defense.trim_ratio = 0.2
    res_def = run_experiment(cfg)
    assert len(res_def["round_logs"]) == 3

def test_fl_under_label_flip_attack():
    cfg = load_config("configs/smoke.yaml")
    cfg.federated.rounds = 3
    cfg.model.learning_rate = 0.1
    cfg.attack.type = "label_flip"
    cfg.attack.malicious_fraction = 0.2
    cfg.defense.type = "median"
    
    res = run_experiment(cfg)
    assert len(res["round_logs"]) == 3
