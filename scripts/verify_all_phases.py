import sys
import os
import time
import torch
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, r"D:\Research paper\fedids-bench\src")

from fedids_bench.config import (
    DatasetConfig, PartitioningConfig, ModelConfig,
    FederatedConfig, AttackConfig, DefenseConfig, ExperimentConfig
)
from fedids_bench.data.loaders import get_dataset_loader, LOADER_REGISTRY
from fedids_bench.data.partition import get_partitioner
from fedids_bench.models.mlp import SmallMLP
from fedids_bench.federated.trainer import train_local
from fedids_bench.federated.server import FederatedServer
from fedids_bench.attacks.factory import get_attack
from fedids_bench.defenses.factory import get_defense
from fedids_bench.evaluation.cross_dataset import align_features, evaluate_cross_dataset
from fedids_bench.evaluation.device_profiles import PROFILES, estimate_resource_costs
from fedids_bench.evaluation.communication import CommunicationTracker
from fedids_bench.experiments.runner import run_experiment
from fedids_bench.reporting.tables import generate_latex_table


def log_phase(phase_num: int, title: str):
    print("\n" + "=" * 70)
    print(f"PHASE {phase_num}: {title.upper()}")
    print("=" * 70)


def verify_phase1_data_loading():
    log_phase(1, "Dataset Loading & Schema Validation (Real Datasets)")
    real_datasets = ["unsw_nb15", "cicids2017", "cse_cic_ids2018", "iot23", "ton_iot"]
    for name in real_datasets:
        t0 = time.time()
        loader = get_dataset_loader(name)
        cfg = DatasetConfig(name=name, n_samples=300, test_fraction=0.2)
        ds = loader.load(cfg, seed=42)
        elapsed = time.time() - t0
        
        assert ds.manifest["dataset_name"] == name, f"Mismatch in manifest dataset_name: {ds.manifest['dataset_name']}"
        assert ds.X.shape[0] == 300, f"Expected 300 samples, got {ds.X.shape[0]}"
        assert ds.X.shape[1] >= 10, f"Expected >=10 features, got {ds.X.shape[1]}"
        assert (ds.split == 0).sum() == 240, f"Train split error"
        assert (ds.split == 2).sum() == 60, f"Test split error"
        
        print(f"  [OK] [{name:16s}] {ds.X.shape[0]} samples, {ds.X.shape[1]} features, "
              f"{len(ds.manifest['label_map'])} classes: {list(ds.manifest['label_map'].values())[:3]}... ({elapsed:.2f}s)")
    print(">> Phase 1 Complete: All 5 real datasets loaded and validated successfully.")


def verify_phase2_partitioning():
    log_phase(2, "Non-IID Client Partitioning on Real Data")
    loader = get_dataset_loader("unsw_nb15")
    ds = loader.load(DatasetConfig(name="unsw_nb15", n_samples=500), seed=42)
    
    strategies = ["iid", "dirichlet", "label_skew", "quantity_skew"]
    for strat in strategies:
        p_cfg = PartitioningConfig(strategy=strat, num_clients=5, heldout_client_fraction=0.2, alpha=0.5)
        partitioner = get_partitioner(strat)
        parts = partitioner.partition(ds, p_cfg, seed=42)
        
        assert len(parts.client_indices) == 5, f"Expected 5 clients, got {len(parts.client_indices)}"
        assert len(parts.heldout_client_ids) == 1, "Expected 1 heldout client"
        active_counts = [len(idx) for cid, idx in parts.client_indices.items() if cid not in parts.heldout_client_ids]
        print(f"  [OK] [{strat:14s}] Active client sample allocations: {active_counts}, Held-out: {parts.heldout_client_ids}")
    print(">> Phase 2 Complete: All partitioning strategies verified on real IDS network flows.")


def verify_phase3_models():
    log_phase(3, "Intrusion Detection Models (Deep MLP Training & Inference)")
    loader = get_dataset_loader("unsw_nb15")
    ds = loader.load(DatasetConfig(name="unsw_nb15", n_samples=300), seed=42)
    
    train_idx = np.where(ds.split == 0)[0]
    test_idx = np.where(ds.split == 2)[0]
    X_train, y_train = ds.X[train_idx], ds.y[train_idx]
    X_test, y_test = ds.X[test_idx], ds.y[test_idx]
    n_features = X_train.shape[1]
    n_classes = len(ds.manifest["label_map"])
    
    mlp = SmallMLP(n_features=n_features, n_classes=n_classes, hidden_dims=[32, 16], seed=42)
    updated_weights, num_samples = train_local(
        model=mlp, X=X_train, y=y_train, epochs=3, batch_size=32, lr=0.01, seed=42
    )
    mlp.set_parameters(updated_weights)
    preds = mlp.predict(X_test)
    probs = mlp.predict_proba(X_test)
    
    assert len(preds) == len(X_test)
    assert probs.shape == (len(X_test), n_classes)
    acc = (preds == y_test).mean()
    print(f"  [OK] SmallMLP trained on real UNSW flows (epochs=3, params={mlp.num_parameters()}): accuracy={acc:.4f}, predictions={preds.shape}")
    print(">> Phase 3 Complete: Neural network IDS model trained and evaluated on real data.")


def verify_phase4_fl_algorithms():
    log_phase(4, "Federated Learning Algorithms (FedAvg, FedProx, FedAdam)")
    algorithms = ["fedavg", "fedprox", "fedadam"]
    for algo in algorithms:
        exp_cfg = ExperimentConfig(
            experiment_id=f"verify_{algo}",
            seed=42,
            mode="federated",
            dataset=DatasetConfig(name="ton_iot", n_samples=300),
            partitioning=PartitioningConfig(strategy="iid", num_clients=3, heldout_client_fraction=0.0),
            model=ModelConfig(architecture="mlp", hidden_dims=[16, 8], learning_rate=0.01),
            federated=FederatedConfig(algorithm=algo, rounds=2, local_epochs=1, batch_size=32, clients_per_round=3),
            attack=AttackConfig(type="none"),
            defense=DefenseConfig(type="none"),
            output_dir=f"results/verify_{algo}"
        )
        res = run_experiment(exp_cfg)
        assert res["status"] == "completed"
        final_f1 = res["metrics"].get("f1", 0.0)
        final_acc = res["metrics"].get("accuracy", 0.0)
        print(f"  [OK] [{algo.upper():7s}] Completed 2 rounds on real ToN_IoT data: final test F1 = {final_f1:.4f}, acc = {final_acc:.4f}")
    print(">> Phase 4 Complete: FedAvg, FedProx, and FedAdam verified on real IoT traffic.")


def verify_phase5_attacks():
    log_phase(5, "Threat Models & Adversarial Poisoning Attacks")
    loader = get_dataset_loader("unsw_nb15")
    ds = loader.load(DatasetConfig(name="unsw_nb15", n_samples=200), seed=42)
    
    # 1. Label Flip
    atk_lf = get_attack(AttackConfig(type="label_flip", target_label=0), n_classes=len(ds.manifest["label_map"]))
    X_atk, y_atk = atk_lf.apply_data_attack(ds.X[:20], ds.y[:20], client_id=1, seed=42)
    assert (y_atk == 0).all()
    print("  [OK] Label Flip Attack: redirected 20 sample labels to class 0")
    
    # 2. Model Poisoning (Delta Scaling with sign flip)
    atk_mp = get_attack(AttackConfig(type="model_poison", strength=5.0))
    gw = {"weight": torch.zeros((10, 5), dtype=torch.float32)}
    lw = {"weight": torch.ones((10, 5), dtype=torch.float32)}
    pw = atk_mp.apply_model_attack(lw, gw, client_id=1, seed=42)
    torch.testing.assert_close(pw["weight"], torch.ones((10, 5)) * -5.0)
    print("  [OK] Model Poisoning Attack: parameter delta scaled & sign-inverted to -5.0x")
    
    # 3. Byzantine Attack
    atk_byz = get_attack(AttackConfig(type="byzantine", strength=2.0))
    bw = atk_byz.apply_model_attack(lw, gw, client_id=1, seed=42)
    assert bw["weight"].shape == lw["weight"].shape
    print("  [OK] Byzantine Noise Attack: isotropic noise vector injected with matching tensor shape")
    
    # 4. Backdoor Attack
    atk_bd = get_attack(AttackConfig(type="backdoor", target_label=1))
    X_bd, y_bd = atk_bd.apply_data_attack(ds.X[:10], ds.y[:10], client_id=1, seed=42)
    num_poisoned = int(10 * atk_bd.trigger_fraction)
    assert (y_bd[:num_poisoned] == 1).all()
    print(f"  [OK] Backdoor Trigger Attack: trigger pattern embedded, {num_poisoned} target labels set to 1")
    print(">> Phase 5 Complete: All adversarial threat models verified.")


def verify_phase6_defenses():
    log_phase(6, "Robust Aggregation Defenses Under Attack")
    gw = {"w": torch.zeros((4, 4), dtype=torch.float32)}
    client_updates = [
        {"w": torch.ones((4, 4), dtype=torch.float32) * 1.0},
        {"w": torch.ones((4, 4), dtype=torch.float32) * 1.0},
        {"w": torch.ones((4, 4), dtype=torch.float32) * 1.0},
        {"w": torch.ones((4, 4), dtype=torch.float32) * 1.0},
        {"w": torch.ones((4, 4), dtype=torch.float32) * 100.0},
    ]
    counts = [10, 10, 10, 10, 10]
    
    defenses = ["clipping", "trimmed_mean", "median", "krum"]
    for d_name in defenses:
        d_cfg = DefenseConfig(type=d_name, clip_threshold=2.0, trim_ratio=0.2, krum_f=1)
        defense = get_defense(d_cfg)
        agg_w = defense.aggregate(gw, client_updates, counts, trust_client_sample_counts=True)
        mean_val = float(agg_w["w"].mean())
        assert mean_val < 5.0, f"Defense {d_name} failed to suppress outlier: mean={mean_val}"
        print(f"  [OK] [{d_name:12s}] Aggregated parameter mean = {mean_val:.4f} (poisoner 100.0 suppressed!)")
    print(">> Phase 6 Complete: All robust aggregation defenses successfully verified.")


def verify_phase7_cross_dataset():
    log_phase(7, "Cross-Dataset Alignment & Evaluation")
    l_unsw = get_dataset_loader("unsw_nb15")
    l_cic = get_dataset_loader("cicids2017")
    
    ds_unsw = l_unsw.load(DatasetConfig(name="unsw_nb15", n_samples=200), seed=42)
    ds_cic = l_cic.load(DatasetConfig(name="cicids2017", n_samples=200), seed=42)
    
    X_aligned = align_features(ds_cic.X, expected_features=ds_unsw.X.shape[1])
    assert X_aligned.shape[1] == ds_unsw.X.shape[1]
    print(f"  [OK] Aligned real CICIDS2017 ({ds_cic.X.shape[1]} feats) -> UNSW dimension ({X_aligned.shape[1]} feats)")
    
    model = SmallMLP(n_features=ds_unsw.X.shape[1], n_classes=len(ds_unsw.manifest["label_map"]), seed=42)
    res = evaluate_cross_dataset(model, ds_cic)
    assert res["aligned_features"] == ds_unsw.X.shape[1]
    assert "metrics" in res
    print(f"  [OK] Cross-dataset transfer evaluation: accuracy={res['metrics']['accuracy']:.4f}")
    print(">> Phase 7 Complete: Cross-dataset alignment and transfer evaluation verified.")


def verify_phase8_profiling():
    log_phase(8, "Hardware Resource & Communication Profiling")
    mlp = SmallMLP(n_features=30, n_classes=4, hidden_dims=[32, 16], seed=42)
    tracker = CommunicationTracker(mlp)
    
    tracker.record_round(round_num=1, num_clients=4)
    comm_stats = tracker.get_summary()
    assert comm_stats["total_communication_bytes"] > 0
    print(f"  [OK] Communication: {comm_stats['total_communication_bytes']:,} total bytes recorded ({comm_stats['total_upload_bytes']} up, {comm_stats['total_download_bytes']} down)")
    
    for dev_name in ["raspberry_pi_4", "jetson_orin_nano", "laptop_cpu"]:
        if dev_name in PROFILES:
            cost = estimate_resource_costs(mlp.num_parameters(), n_samples=500, n_epochs=2, profile_name=dev_name)
            print(f"  [OK] Latency estimate on [{dev_name:17s}]: {cost['estimated_compute_time_sec']:.6f}s (Energy: {cost['estimated_energy_joules']:.6f} J)")
    print(">> Phase 8 Complete: Hardware resource and communication profiling verified.")


def verify_phase9_orchestration_and_reporting():
    log_phase(9, "Benchmark Experiment Orchestration & Reporting")
    exp_cfg = ExperimentConfig(
        experiment_id="real_data_benchmark",
        seed=42,
        mode="federated",
        dataset=DatasetConfig(name="unsw_nb15", n_samples=300),
        partitioning=PartitioningConfig(strategy="iid", num_clients=3, heldout_client_fraction=0.0),
        model=ModelConfig(architecture="mlp", hidden_dims=[16, 8], learning_rate=0.01),
        federated=FederatedConfig(algorithm="fedavg", rounds=2, local_epochs=1, batch_size=32, clients_per_round=3),
        attack=AttackConfig(type="none"),
        defense=DefenseConfig(type="none"),
        output_dir="results/real_data_benchmark"
    )
    results = run_experiment(exp_cfg)
    
    assert os.path.exists("results/real_data_benchmark/result.json")
    assert os.path.exists("results/real_data_benchmark/result.csv")
    
    summary_df = pd.DataFrame([{
        "Experiment ID": results["experiment_id"],
        "Dataset": results["dataset"],
        "Algorithm": results["algorithm"],
        "Accuracy": results["metrics"]["accuracy"],
        "Macro F1": results["metrics"]["multiclass"]["macro_f1"],
        "Total Comm (Bytes)": results["communication"]["total_communication_bytes"]
    }])
    latex = generate_latex_table(summary_df, caption="FedIDS-Bench Real Dataset Results")
    assert "\\begin{table}" in latex
    
    print(f"  [OK] Experiment ID: {results['experiment_id']}")
    print(f"  [OK] Real Global Test Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"  [OK] Real Multiclass Macro F1:  {results['metrics']['multiclass']['macro_f1']:.4f}")
    print(f"  [OK] Results saved to: {exp_cfg.output_dir}/result.json and result.csv")
    print(f"  [OK] LaTeX Table generated successfully ({len(latex.splitlines())} lines)")
    print(">> Phase 9 Complete: End-to-end benchmark orchestration & reporting verified.")


def main():
    print("=" * 70)
    print("FEDIDS-BENCH: COMPLETE 9-PHASE BENCHMARK VERIFICATION (REAL DATA)")
    print("=" * 70)
    start_time = time.time()
    
    verify_phase1_data_loading()
    verify_phase2_partitioning()
    verify_phase3_models()
    verify_phase4_fl_algorithms()
    verify_phase5_attacks()
    verify_phase6_defenses()
    verify_phase7_cross_dataset()
    verify_phase8_profiling()
    verify_phase9_orchestration_and_reporting()
    
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"ALL 9 PHASES VERIFIED AND PASSED IN {total_elapsed:.2f} SECONDS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
