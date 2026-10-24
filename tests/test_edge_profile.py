import csv
import pytest
from fedids_bench.evaluation.edge_profile import (
    DEFAULT_DEVICES, build_profile, count_flops, count_params, estimate, implied_eta,
    measure_host, mlp_dims, write_csv,
)


def test_param_and_flop_counts_match_hand_calculation():
    dims = mlp_dims(5, [32, 16], 2)
    assert count_params(dims) == 754                         # 5*32+32 + 32*16+16 + 16*2+2
    assert 4 * count_params(dims) == 3016                    # float32 bytes
    # (2*5*32+32)+32  +  (2*32*16+16)+16  +  (2*16*2+2)  = 1506
    assert count_flops(dims) == 1506


def test_widened_core_model_sizes():
    dims = mlp_dims(22, [64, 32], 2)
    assert count_params(dims) == 22 * 64 + 64 + 64 * 32 + 32 + 32 * 2 + 2 == 3618


def test_latency_energy_follow_the_stated_formula_for_every_device():
    dims = mlp_dims(22, [64, 32], 2)
    flops = count_flops(dims)
    for dev in DEFAULT_DEVICES:
        for eta in (0.01, 0.10, 0.25):
            r = estimate(dev, flops, batch_size=32, eta=eta)
            lat_s = flops / (dev.peak_gflops * 1e9 * eta)
            assert r["latency_ms_per_sample"] == pytest.approx(lat_s * 1e3)
            assert r["energy_mj_per_batch"] == pytest.approx(32 * lat_s * dev.tdp_w * 1e3)


def test_profile_is_computed_not_typed_in():
    rows_small = build_profile(mlp_dims(5, [32, 16], 2))
    rows_big = build_profile(mlp_dims(22, [64, 32], 2))
    for a, b in zip(rows_small, rows_big):
        ratio = b["flops_per_sample"] / a["flops_per_sample"]
        assert b["latency_ms_per_sample_worst"] / a["latency_ms_per_sample_worst"] == pytest.approx(ratio)
        assert a["latency_ms_per_sample_worst"] == pytest.approx(10 * a["latency_ms_per_sample_best"])


def test_fl_traffic_is_per_client_per_round_up_plus_down():
    row = build_profile(mlp_dims(5, [32, 16], 2))[0]
    assert row["upload_bytes_per_client_round"] == row["download_bytes_per_client_round"] == 3016
    assert row["traffic_bytes_per_client_round"] == 6032


def test_host_measurement_and_implied_eta(tmp_path):
    dims = mlp_dims(22, [64, 32], 2)
    m = measure_host(dims, batch_size=32, repeats=200)
    assert m["latency_ms_per_sample_measured"] > 0
    eta = implied_eta(m["flops_per_sample"], m["latency_ms_per_sample_measured"], peak_gflops=220.0)
    assert 0 < eta
    out = tmp_path / "edge.csv"
    write_csv(str(out), build_profile(dims) + [m])
    with open(out, newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == len(DEFAULT_DEVICES) + 1
