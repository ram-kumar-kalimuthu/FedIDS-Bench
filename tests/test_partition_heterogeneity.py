"""Partition-heterogeneity statistics (the numbers that justify the Dirichlet / Pareto / single-domain settings)."""
import json

import numpy as np
import pytest

from fedids_bench.experiments.aggregate import load_partition_stats, partition_stats
from fedids_bench.reporting.paper_tables import build_partition_heterogeneity


def _client(cid, n0, n1, heldout=False):
    dist = {"0": n0, "1": n1} if (n0 or n1) else {}
    return {"client_id": cid, "is_heldout": heldout, "n_samples": n0 + n1, "class_distribution": dist}


def _manifest(specs, heldout=()):
    return {"strategy": "x", "clients": {str(i): _client(i, n0, n1, i in heldout) for i, (n0, n1) in enumerate(specs)}}


def test_extreme_label_skew_matches_hand_computation():
    # two benign-only and two attack-only clients; pooled distribution (.5, .5)
    # JS(q=(1,0) || p=(.5,.5)) = .5*log2(4/3) + .5*(.5*log2(2/3) + .5*log2(2)) = 0.3112781...
    st = partition_stats(_manifest([(100, 0), (0, 100), (100, 0), (0, 100)]))
    assert st["js_mean"] == pytest.approx(0.31127812445913283, abs=1e-9)
    assert st["tv_mean"] == pytest.approx(0.5)
    assert st["single_class_frac"] == 1.0 and st["no_attack_frac"] == 0.5
    assert st["attack_share_min"] == 0.0 and st["attack_share_max"] == 1.0
    assert st["size_ratio"] == 1.0 and st["size_cv"] == 0.0


def test_iid_like_split_has_no_divergence():
    st = partition_stats(_manifest([(70, 30)] * 5))
    assert st["js_mean"] == pytest.approx(0.0, abs=1e-12) and st["tv_mean"] == pytest.approx(0.0, abs=1e-12)
    assert st["single_class_frac"] == 0.0 and st["no_attack_frac"] == 0.0
    assert st["attack_share_min"] == pytest.approx(0.3) == pytest.approx(st["attack_share_max"])


def test_heldout_and_empty_clients_are_ignored_and_size_spread_is_reported():
    st = partition_stats(_manifest([(7, 3), (700, 300), (0, 0), (70, 30)], heldout={3}))
    assert st["n_clients"] == 2.0                                   # the empty client and the held-out client do not count
    assert st["size_ratio"] == pytest.approx(100.0) and st["size_min"] == 10.0 and st["size_max"] == 1000.0
    assert st["size_cv"] == pytest.approx(np.std([10, 1000]) / np.mean([10, 1000]))
    assert partition_stats({"clients": {}}) == {}


def _write_run(d, ds, part, weights, seed, manifest):
    rid = f"partitioning__{ds}__{part}__{weights}__s{seed}"
    payload = {"run_id": rid, "grid": "partitioning", "dataset": ds, "variant": f"part_{part}_{weights}", "group": "fedavg",
               "seed": seed, "factors": {"partition": part, "weights": weights}, "result": {"partition_manifest": manifest}}
    (d / f"{rid}.json").write_text(json.dumps(payload))


@pytest.fixture()
def run_dir(tmp_path):
    iid = _manifest([(70, 30)] * 10)
    skew = _manifest([(100, 0), (0, 100)] * 5)
    for ds in ("ton_iot", "iot23"):
        for seed in range(3):
            for weights in ("weighted", "uniform"):
                _write_run(tmp_path, ds, "iid", weights, seed, iid)
                _write_run(tmp_path, ds, "dir0.1", weights, seed, skew)
    (tmp_path / "_provenance_partitioning.json").write_text("{}")   # metadata files are not runs
    return tmp_path


def test_each_partition_is_counted_once_per_corpus_and_seed(run_dir):
    df = load_partition_stats(str(run_dir))
    assert len(df) == 2 * 3 * 2                                     # corpora x seeds x partitions, not x weights
    assert set(df["f_partition"]) == {"iid", "dir0.1"}
    assert df[df["f_partition"] == "iid"]["js_mean"].max() < 1e-12


def test_runs_without_a_manifest_are_skipped(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"run_id": "a", "dataset": "ton_iot", "variant": "v", "seed": 0, "result": {}}))
    assert load_partition_stats(str(tmp_path)).empty
    assert build_partition_heterogeneity(load_partition_stats(str(tmp_path))) == []


def test_table_reports_measured_values_and_carries_its_label(run_dir):
    tables = {t.name: t for t in build_partition_heterogeneity(load_partition_stats(str(run_dir)), banner=None)}
    assert set(tables) == {"partition_heterogeneity", "partition_heterogeneity_by_corpus"}
    data = tables["partition_heterogeneity"].data.set_index("partition")
    assert data.loc["iid", "js_mean"] < 1e-9 and data.loc["dir0.1", "js_mean"] == pytest.approx(0.31127812445913283)
    assert data.loc["dir0.1", "single_class_pct"] == pytest.approx(100.0) and data.loc["iid", "single_class_pct"] == 0.0
    assert data.loc["dir0.1", "no_attack_pct"] == pytest.approx(50.0)
    tex = tables["partition_heterogeneity"].tex
    assert r"\label{tab:partition_heterogeneity}" in tex and "nan" not in tex.lower()
    assert "IID" in tex and "Dir(0.1)" in tex and "0.00--1.00" in tex
    by_corpus = tables["partition_heterogeneity_by_corpus"].data
    assert set(by_corpus["dataset"]) == {"ton_iot", "iot23"}
