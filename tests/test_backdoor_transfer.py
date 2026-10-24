"""Backdoor success rate, the column-shuffled transfer control and the transfer aggregation."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from fedids_bench.config import AttackConfig, ExperimentConfig
from fedids_bench.data.loaders import get_dataset_loader
from fedids_bench.evaluation.backdoor import apply_trigger, backdoor_success, is_backdoor_run
from fedids_bench.evaluation.cross_dataset import evaluate_cross_dataset
from fedids_bench.experiments.aggregate import (
    load_transfer, transfer_diagnostics, transfer_matrix, transfer_summary,
)
from fedids_bench.experiments.runner import run_experiment
from fedids_bench.experiments.transfer import run_transfer_seed
from fedids_bench.models.mlp import SmallMLP
from tests.test_features_loaders import WRITERS


def _fit(model, X, y, steps=400, lr=0.05, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(model(Xt), yt).backward()
        opt.step()
    model.eval()
    return model


def _toy(n=4000, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = ((X[:, 2] + X[:, 3]) > 0).astype(np.int64)                    # attack iff x2 + x3 > 0
    return X, y


def test_apply_trigger_sets_only_the_first_dims():
    X = np.zeros((4, 6), dtype=np.float32)
    Xt = apply_trigger(X, 5.0, 2)
    assert (Xt[:, :2] == 5.0).all() and (Xt[:, 2:] == 0.0).all() and (X == 0).all()
    assert (apply_trigger(X, 3.0, 99) == 3.0).all()                    # more dims than features -> all of them


def test_backdoor_success_separates_a_backdoored_from_a_clean_model():
    X, y = _toy()
    Xtr, ytr, Xte, yte = X[:3000], y[:3000], X[3000:], y[3000:]
    atk = AttackConfig(type="backdoor", malicious_fraction=0.2, trigger_value=5.0, trigger_dims=2)

    clean = _fit(SmallMLP(8, 2, [16, 8], seed=0), Xtr, ytr)
    Xp, yp = Xtr.copy(), ytr.copy()
    k = int(0.3 * len(Xp))                                             # poison 30% of the training rows
    Xp[:k] = apply_trigger(Xp[:k], 5.0, 2)
    yp[:k] = 0
    poisoned = _fit(SmallMLP(8, 2, [16, 8], seed=0), Xp, yp)

    r_clean = backdoor_success(clean, Xte, yte, atk)
    r_bd = backdoor_success(poisoned, Xte, yte, atk)
    assert r_clean["n_victims"] == int((yte == 1).sum()) == r_bd["n_victims"]
    assert r_clean["asr"] < 0.25 and r_clean["clean_miss"] < 0.1       # a clean model is not fooled by the trigger
    assert r_bd["asr"] > 0.8 and r_bd["asr_lift"] > 0.6                # the poisoned one is
    assert r_bd["clean_miss"] < 0.15                                   # ... while behaving normally without it
    assert r_bd["asr_lift"] == pytest.approx(r_bd["asr"] - r_bd["clean_miss"])


def test_backdoor_success_handles_no_victims():
    X, _ = _toy(50)
    out = backdoor_success(SmallMLP(8, 2, seed=0), X, np.zeros(50, dtype=np.int64), AttackConfig(type="backdoor"))
    assert out["n_victims"] == 0 and out["asr"] is None


def test_is_backdoor_run_flags():
    assert is_backdoor_run(AttackConfig(type="backdoor"))
    assert is_backdoor_run(AttackConfig(type="sample_inflation", combine_with="backdoor"))
    assert not is_backdoor_run(AttackConfig(type="sample_inflation"))
    assert not is_backdoor_run(AttackConfig(type="none"))


def _fixture_cfg(tmp_path, name="ton_iot", **over):
    writer, fname = WRITERS[name]
    writer(tmp_path / fname, n=1500)
    cfg = dict(
        seed=0,
        dataset=dict(name=name, n_samples=1200, val_fraction=0.0, data_dir=str(tmp_path), use_cache=False),
        partitioning=dict(strategy="dirichlet", alpha=0.5, num_clients=5, heldout_client_fraction=0.0),
        model=dict(hidden_dims=[16, 8], learning_rate=0.05),
        federated=dict(rounds=2, clients_per_round=5, batch_size=32),
        output_dir=str(tmp_path / "out"),
    )
    cfg.update(over)
    return ExperimentConfig(**cfg)


def test_runner_reports_asr_only_for_backdoor_runs(tmp_path):
    res = run_experiment(_fixture_cfg(tmp_path, attack=dict(type="backdoor", malicious_fraction=0.2)), save=False)
    bd = res["backdoor"]
    assert bd is not None and bd["n_victims"] > 0 and 0.0 <= bd["asr"] <= 1.0 and 0.0 <= bd["clean_miss"] <= 1.0
    assert run_experiment(_fixture_cfg(tmp_path), save=False)["backdoor"] is None


def test_column_shuffle_control_destroys_a_model_that_uses_feature_semantics(tmp_path):
    cfg = _fixture_cfg(tmp_path, mode="centralized", federated=dict(rounds=15, local_epochs=1, batch_size=32, clients_per_round=5))
    res, art = run_experiment(cfg, save=False, return_artifacts=True)
    model, ds = art["model"], art["dataset"]
    strict = evaluate_cross_dataset(model, ds, source_manifest=ds.manifest, scaler="source", split=2)
    shuffled = evaluate_cross_dataset(model, ds, source_manifest=ds.manifest, scaler="source", split=2, permute_columns_seed=3)
    assert not strict["columns_permuted"] and shuffled["columns_permuted"]
    assert strict["metrics"]["balanced_accuracy"] > 0.9                # separable fixture
    assert shuffled["metrics"]["balanced_accuracy"] < strict["metrics"]["balanced_accuracy"] - 0.1
    # same seed -> same permutation -> same result
    again = evaluate_cross_dataset(model, ds, source_manifest=ds.manifest, scaler="source", split=2, permute_columns_seed=3)
    assert again["metrics"]["balanced_accuracy"] == shuffled["metrics"]["balanced_accuracy"]


def test_transfer_study_and_aggregation(tmp_path):
    names = ["ton_iot", "cicids2017"]
    for n in names:
        w, f = WRITERS[n]
        w(tmp_path / f, n=1500)
    spec = {
        "name": "transfer", "datasets": names, "seeds": [0, 1], "group": "fedavg",
        "base": {
            "dataset": {"n_samples": 1000, "val_fraction": 0.0, "data_dir": str(tmp_path), "use_cache": False},
            "partitioning": {"strategy": "dirichlet", "alpha": 0.5, "num_clients": 4, "heldout_client_fraction": 0.0},
            "model": {"hidden_dims": [16, 8], "learning_rate": 0.05},
            "federated": {"rounds": 3, "clients_per_round": 4, "batch_size": 32},
        },
    }
    out = tmp_path / "tr"
    for seed in spec["seeds"]:
        status = run_transfer_seed(spec, seed, str(out), tuned=None)
        assert {s for _, s in status} == {"ok"}
    # a second pass is fully cached
    assert {s for _, s in run_transfer_seed(spec, 0, str(out), tuned=None)} == {"cached"}

    df = load_transfer(str(out))
    assert set(df["mode"]) == {"strict", "target_scaler", "strict_shuffled"}
    assert len(df) == 2 * 2 * 2 * 3                                    # seeds x sources x targets x modes
    summ = transfer_summary(df, "balanced_accuracy", "strict")
    mat = transfer_matrix(summ, "mean")
    assert mat.shape == (2, 2) and set(mat.index) == set(names) == set(mat.columns)
    assert (summ.loc[summ.is_diagonal, "mean"] > 0.5).all()           # in-domain cells learn the separable fixture
    diag = transfer_diagnostics(df)
    assert len(diag) == 2 and {"beats_chance", "mean_drop_from_in_domain", "control_gap"} <= set(diag.columns)
    # the diagonal of the strict mode equals the in-domain federated metric of the very same model
    import json, glob
    for fn in glob.glob(str(out / "transfer__*__s*.json")):
        d = json.load(open(fn))
        assert d["targets"][d["source"]]["strict"]["balanced_accuracy"] == pytest.approx(
            d["in_domain_federated"]["balanced_accuracy"])
