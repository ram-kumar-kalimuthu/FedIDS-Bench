"""Regression tests for findings of the second audit (docs/audit_findings.md).

The AUROC test and the attacker-selection test fail on the code as submitted; the aggregation and no-attack tests pin behavior
that the paper's claims about sample-count inflation rely on."""
import numpy as np
import pytest
import torch

from fedids_bench.config import load_config
from fedids_bench.evaluation.classification import compute_classification_metrics
from fedids_bench.experiments.runner import run_experiment
from fedids_bench.federated.algorithms.fedavg import fedavg_aggregate

LABELS = {0: "Benign", 1: "Attack"}


def test_auroc_is_computed_from_scores_not_copied_from_balanced_accuracy():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    score = np.array([0.1, 0.6, 0.9, 0.8, 0.4, 0.2])          # P(attack)
    m = compute_classification_metrics(y_true, y_pred, LABELS, y_score=score)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    assert m["auroc"] == pytest.approx(8 / 9)                  # 8 of the 9 (attack, benign) pairs are ranked correctly
    assert compute_classification_metrics(y_true, y_pred, LABELS)["auroc"] is None   # no scores -> no AUROC (not a stand-in)


def test_inflated_count_dominates_count_weighted_fedavg_but_not_uniform_fedavg():
    honest = {"w": torch.zeros(2)}
    liar = {"w": torch.full((2,), 10.0)}
    updates, counts = [honest, honest, liar], [100, 100, 1_000_000]
    weighted = fedavg_aggregate(updates, counts, trust_client_sample_counts=True)
    uniform = fedavg_aggregate(updates, counts, trust_client_sample_counts=False)
    assert float(weighted["w"][0]) > 9.99                      # the liar owns > 99.97 % of the weight
    assert float(uniform["w"][0]) == pytest.approx(10.0 / 3)   # every client counts once, whatever it reports


def _smoke(rounds=2, **attack):
    cfg = load_config("configs/smoke.yaml")                    # 5 clients, heldout_client_fraction 0.2 -> 4 active clients
    cfg.federated.rounds = rounds
    for key, value in attack.items():
        setattr(cfg.attack, key, value)
    return cfg


def test_attackers_are_active_clients_even_when_clients_are_held_out():
    """Old code drew the attackers from range(n), i.e. from the held-out clients, so nobody participating was attacked."""
    cfg = _smoke(type="sample_inflation", malicious_fraction=0.5, reported_count=1_000_000)
    assert cfg.partitioning.heldout_client_fraction > 0
    res = run_experiment(cfg, save=False)
    for log in res["round_logs"]:
        counts = np.array(log["sample_counts"])
        assert len(counts) == 4                                # all four active clients take part
        assert int((counts == 1_000_000).sum()) == 2           # int(4 * 0.5) attackers, all among the participants
        assert int((counts < 1_000_000).sum()) == 2            # the others report their true size


def test_no_attack_means_no_inflated_counts():
    res = run_experiment(_smoke(), save=False)
    for log in res["round_logs"]:
        assert max(log["sample_counts"]) < 10_000


# --------------------------------------------------------------------------------------------------- third pass
from fedids_bench.config import ExperimentConfig, n_heldout_clients, n_malicious      # noqa: E402
from fedids_bench.data.partition import _calculate_heldout_count                      # noqa: E402
from fedids_bench.defenses.median import MedianDefense                               # noqa: E402
from fedids_bench.utils.reproducibility import derive_seed                           # noqa: E402


def test_derived_seeds_do_not_collide_across_runs_clients_and_rounds():
    # the old arithmetic  seed + 10000 * client + round  gave (seed 0, round 1) and (seed 1, round 0) ONE random stream
    assert 0 + 10000 * 3 + 1 == 1 + 10000 * 3 + 0
    assert derive_seed(0, 3, 1) != derive_seed(1, 3, 0)
    streams = {derive_seed(s, c, r) for s in range(5) for c in range(10) for r in range(20)}
    assert len(streams) == 5 * 10 * 20                           # 5 seeds x 10 clients x 20 rounds, all different
    assert derive_seed(7, 2, 5) == derive_seed(7, 2, 5)          # and reproducible
    assert 0 <= derive_seed(2 ** 40, 99, 3) < 2 ** 32


def test_median_of_an_even_number_of_clients_averages_the_two_middle_values():
    updates = [{"w": torch.tensor([v])} for v in (1.0, 2.0, 3.0, 10.0)]
    agg = MedianDefense().aggregate({"w": torch.tensor([0.0])}, updates, [10] * 4)
    assert float(agg["w"][0]) == pytest.approx(2.5)
    # why: torch.median returns the LOWER middle value, a systematic downward bias with the usual 10 clients
    assert float(torch.stack([u["w"] for u in updates]).median()) == 2.0


def test_krum_feasibility_counts_participating_clients_and_simulated_attackers():
    def cfg(heldout, per_round, krum_f, attack=None):
        kw = dict(partitioning={"num_clients": 10, "heldout_client_fraction": heldout},
                  federated={"clients_per_round": per_round}, defense={"type": "krum", "krum_f": krum_f})
        if attack:
            kw["attack"] = attack
        return ExperimentConfig(**kw)

    cfg(0.0, 10, 2)                                              # 10 > 2*2 + 2
    cfg(0.2, 8, 2)                                               # 2 clients held out -> 8 take part: 8 > 2*2 + 2
    with pytest.raises(ValueError, match="Krum defense requires"):
        cfg(0.2, 8, 3)                                           # 8 > 2*3 + 2 is false (10 > 8 would have passed)
    with pytest.raises(ValueError, match="Krum defense requires"):
        cfg(0.0, 10, 1, {"type": "model_poison", "malicious_fraction": 0.4})   # 4 attackers: 10 > 2*4 + 2 is false
    cfg(0.0, 10, 1, {"type": "model_poison", "malicious_fraction": 0.3})       # 3 attackers: 10 > 8


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 20])
@pytest.mark.parametrize("fraction", [0.0, 0.05, 0.1, 0.2, 0.25, 0.5])
def test_heldout_rule_is_shared_by_config_validation_and_partitioner(n, fraction):
    assert n_heldout_clients(n, fraction) == _calculate_heldout_count(n, fraction)


def test_number_of_attackers_rule():
    assert (n_malicious(10, 0.2), n_malicious(10, 0.3), n_malicious(7, 0.3), n_malicious(10, 0.0)) == (2, 3, 2, 0)
