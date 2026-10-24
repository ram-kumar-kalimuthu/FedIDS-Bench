import json
import math
import numpy as np
import pytest
from scipy import stats

from fedids_bench.experiments.grid import expand_spec, make_run_id
from fedids_bench.experiments.aggregate import (
    beats_majority, convergence, fmt_pm, load_runs, paired_compare, summarize, t_ci95,
)

SPEC = {
    "name": "g", "seeds": [0, 1, 2], "datasets": ["ton_iot", "iot23"],
    "base": {"federated": {"rounds": 20, "batch_size": 64, "class_weight": "balanced"}, "model": {"learning_rate": 0.01}},
    "variants": [
        {"name": "fedavg_iid", "group": "fedavg", "overrides": {"partitioning": {"strategy": "iid"}}},
        {"name": "fedadam_a0.1", "group": "fedadam", "overrides": {"federated": {"algorithm": "fedadam"},
                                                                    "partitioning": {"strategy": "dirichlet", "alpha": 0.1}}},
    ],
}
TUNED = {"ton_iot": {"fedavg": {"model": {"learning_rate": 0.1}, "federated": {"client_momentum": 0.9}}}}


def test_expand_spec_size_ids_and_merge_order():
    jobs = expand_spec(SPEC, TUNED)
    assert len(jobs) == 2 * 2 * 3
    j = next(x for x in jobs if x["dataset"] == "ton_iot" and x["variant"] == "fedavg_iid" and x["seed"] == 1)
    assert j["run_id"] == make_run_id("g", "ton_iot", "fedavg_iid", 1) == "g__ton_iot__fedavg_iid__s1"
    ov = j["overrides"]
    assert ov["model"]["learning_rate"] == 0.1                      # tuned beats base
    assert ov["federated"]["client_momentum"] == 0.9 and ov["federated"]["rounds"] == 20 and ov["federated"]["class_weight"] == "balanced"
    assert ov["partitioning"]["strategy"] == "iid" and ov["dataset"]["name"] == "ton_iot" and ov["seed"] == 1
    other = next(x for x in jobs if x["dataset"] == "iot23" and x["variant"] == "fedavg_iid")
    assert other["overrides"]["model"]["learning_rate"] == 0.01      # no tuned entry for iot23 -> base value
    assert SPEC["base"]["model"]["learning_rate"] == 0.01            # spec itself is not mutated


def _write_run(d, ds, var, seed, bal, f1, maj_f1, rounds=3):
    payload = {"run_id": f"g__{ds}__{var}__s{seed}", "grid": "g", "dataset": ds, "variant": var, "group": "x", "seed": seed,
               "wall_time_sec": 1.0,
               "result": {"metrics": {"balanced_accuracy": bal, "f1": f1, "mcc": 0.1, "auroc": 0.7, "accuracy": 0.7},
                          "baseline_majority": {"balanced_accuracy": 0.5, "f1": maj_f1, "mcc": 0.0, "auroc": 0.5, "accuracy": 0.6},
                          "round_logs": [{"round_num": r + 1, "tier1_global_test_f1": f1 * (r + 1) / rounds,
                                          "tier1_global_test_bal_acc": bal * (r + 1) / rounds, "tier1_global_test_mcc": 0.0}
                                         for r in range(rounds)]}}
    (d / f"{payload['run_id']}.json").write_text(json.dumps(payload))


@pytest.fixture()
def runs(tmp_path):
    rng = np.random.default_rng(0)
    truth = {}
    for ds in ("ton_iot", "iot23"):
        for var, base in (("fedavg_iid", 0.62), ("fedadam_a0.1", 0.70)):
            for seed in range(5):
                bal = base + rng.normal(0, 0.01)
                truth[(ds, var, seed)] = bal
                _write_run(tmp_path, ds, var, seed, bal, f1=bal + 0.05, maj_f1=0.60)
    return tmp_path, truth


def test_load_and_summarize_matches_numpy(runs):
    d, truth = runs
    final, rounds = load_runs(str(d))
    assert len(final) == 20 and len(rounds) == 20 * 3
    s = summarize(final, ["balanced_accuracy"])
    row = s[(s.dataset == "ton_iot") & (s.variant == "fedavg_iid")].iloc[0]
    vals = np.array([truth[("ton_iot", "fedavg_iid", k)] for k in range(5)])
    assert row["balanced_accuracy_mean"] == pytest.approx(vals.mean())
    assert row["balanced_accuracy_std"] == pytest.approx(vals.std(ddof=1))
    assert row["balanced_accuracy_ci95"] == pytest.approx(stats.t.ppf(0.975, 4) * vals.std(ddof=1) / math.sqrt(5))
    assert t_ci95(0.01, 5) == pytest.approx(2.7764451 * 0.01 / math.sqrt(5), rel=1e-5)


def test_paired_compare_and_majority_gain(runs):
    d, truth = runs
    final, _ = load_runs(str(d))
    cmp = paired_compare(final, "fedadam_a0.1", "fedavg_iid", "balanced_accuracy")
    r = cmp[cmp.dataset == "iot23"].iloc[0]
    diff = np.array([truth[("iot23", "fedadam_a0.1", k)] - truth[("iot23", "fedavg_iid", k)] for k in range(5)])
    assert r["mean_diff"] == pytest.approx(diff.mean()) and r["p_value"] < 0.001
    bm = beats_majority(final, "balanced_accuracy")
    assert bm["beats_majority"].all()                                  # 0.62 / 0.70 are clearly above 0.5


def test_beats_majority_false_when_at_chance(tmp_path):
    for seed in range(5):
        _write_run(tmp_path, "ton_iot", "fedavg_iid", seed, bal=0.5, f1=0.60, maj_f1=0.60)
    final, _ = load_runs(str(tmp_path))
    bm = beats_majority(final, "balanced_accuracy")
    assert not bm["beats_majority"].iloc[0]


def test_convergence_and_formatting(runs):
    d, _ = runs
    _, rounds = load_runs(str(d))
    c = convergence(rounds, "balanced_accuracy")
    assert set(c["round"]) == {1, 2, 3} and (c["n_seeds"] == 5).all()
    assert fmt_pm(0.8341, 0.0123) == "0.834±.012" and fmt_pm(float("nan"), 0.1) == "--" and fmt_pm(0.5, float("nan")) == "0.500"


# ---------------------------------------------------------------------------------------------- matrix grids
def test_expand_matrix_product_names_overrides_and_exclusions(tmp_path):
    import yaml
    from fedids_bench.experiments.grid import expand_matrix, load_spec

    matrix = {
        "group": "fedavg",
        "factors": {
            "attack": [{"name": "clean", "overrides": {"attack": {"type": "none"}}},
                       {"name": "flip", "overrides": {"attack": {"type": "label_flip", "malicious_fraction": 0.2}}}],
            "defense": [{"name": "wtd", "overrides": {"defense": {"type": "none"}, "federated": {"trust_client_sample_counts": True}}},
                        {"name": "krum", "overrides": {"defense": {"type": "krum", "krum_f": 2}}}],
        },
        "exclude": [{"attack": "clean", "defense": "krum"}],
    }
    v = expand_matrix(matrix)
    assert [x["name"] for x in v] == ["clean__wtd", "flip__wtd", "flip__krum"]
    assert v[2]["overrides"] == {"attack": {"type": "label_flip", "malicious_fraction": 0.2}, "defense": {"type": "krum", "krum_f": 2}}
    assert v[0]["group"] == "fedavg" and v[1]["factors"] == {"attack": "flip", "defense": "wtd"}

    spec = {"name": "m", "datasets": ["ton_iot"], "seeds": [0, 1], "base": {}, "matrix": matrix,
            "variants": [{"name": "extra", "overrides": {}}]}
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(spec))
    loaded = load_spec(str(p))
    assert [x["name"] for x in loaded["variants"]] == ["extra", "clean__wtd", "flip__wtd", "flip__krum"]
    assert len(expand_spec(loaded)) == 4 * 2

    spec["variants"] = [{"name": "clean__wtd", "overrides": {}}]
    p.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        load_spec(str(p))


def test_matrix_level_groups_prefix_and_factors(tmp_path):
    """A level may carry its own tuning group; several matrices share one spec; every job knows its factor levels."""
    import yaml
    from fedids_bench.experiments.grid import expand_spec, load_spec

    spec = {
        "name": "m", "datasets": ["ton_iot"], "seeds": [0, 1],
        "base": {"federated": {"rounds": 3}},
        "matrices": [
            {"prefix": "alg", "factors": {
                "algorithm": [
                    {"name": "fedavg", "group": "fedavg", "overrides": {"federated": {"algorithm": "fedavg"}}},
                    {"name": "fedprox", "group": "fedprox", "overrides": {"federated": {"algorithm": "fedprox"}}}],
                "partition": [
                    {"name": "iid", "overrides": {"partitioning": {"strategy": "iid"}}},
                    {"name": "dir0.1", "overrides": {"partitioning": {"strategy": "dirichlet", "alpha": 0.1}}}]}},
            {"prefix": "gamma", "group": "fedavg", "factors": {
                "g": [{"name": "g1", "overrides": {"attack": {"strength": 1.0}}},
                      {"name": "g5", "overrides": {"attack": {"strength": 5.0}}}]},
             "exclude": [{"g": "g1"}]},
        ],
    }
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(spec))
    loaded = load_spec(str(p))
    names = [v["name"] for v in loaded["variants"]]
    assert names == ["alg__fedavg__iid", "alg__fedavg__dir0.1", "alg__fedprox__iid", "alg__fedprox__dir0.1", "gamma__g5"]
    groups = {v["name"]: v["group"] for v in loaded["variants"]}
    assert groups["alg__fedprox__iid"] == "fedprox" and groups["gamma__g5"] == "fedavg"

    tuned = {"ton_iot": {"fedprox": {"federated": {"mu": 0.5}}, "fedavg": {"federated": {"mu": 0.0}}}}
    jobs = expand_spec(loaded, tuned)
    j = next(x for x in jobs if x["variant"] == "alg__fedprox__dir0.1" and x["seed"] == 1)
    assert j["factors"] == {"algorithm": "fedprox", "partition": "dir0.1"}
    assert j["overrides"]["federated"]["mu"] == 0.5 and j["overrides"]["federated"]["algorithm"] == "fedprox"
    assert j["overrides"]["partitioning"]["alpha"] == 0.1 and j["overrides"]["federated"]["rounds"] == 3


def test_load_runs_exposes_factor_columns(tmp_path):
    payload = {
        "run_id": "g__ton_iot__a__s0", "grid": "g", "dataset": "ton_iot", "variant": "a", "group": "fedavg", "seed": 0,
        "factors": {"attack": "clean", "defense": "krum"},
        "result": {"metrics": {"balanced_accuracy": 0.9, "f1": 0.8, "mcc": 0.7, "confusion_matrix": {"tp": 1, "fp": 1, "tn": 1, "fn": 1}},
                   "baseline_majority": {"balanced_accuracy": 0.5}, "round_logs": []},
    }
    (tmp_path / "g__ton_iot__a__s0.json").write_text(json.dumps(payload))
    final, _ = load_runs(str(tmp_path))
    assert final.loc[0, "f_attack"] == "clean" and final.loc[0, "f_defense"] == "krum"
