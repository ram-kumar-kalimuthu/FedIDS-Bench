"""
Tests for the widened feature core (port range, connection outcome), multi-file pooling, the pool cache and
the IoT-23 detailed labels.  No real data is needed: fixtures mimic the real header styles.
"""
import os

import numpy as np
import pandas as pd
import pytest

from fedids_bench.config import DatasetConfig
from fedids_bench.data.features import (
    CONN_OUTCOMES, CORE22_FEATURE_NAMES, CORE5_FEATURE_NAMES, CORE_FEATURE_NAMES, FEATURE_SETS, PORT_RANGES,
    build_features, conn_outcome_from_flags, conn_outcome_from_state, extract_canonical, port_range,
    resolve_columns,
)
from fedids_bench.data.loaders import get_dataset_loader


def test_feature_set_sizes_and_nesting():
    assert len(CORE5_FEATURE_NAMES) == 5 and len(CORE22_FEATURE_NAMES) == 22 and len(CORE_FEATURE_NAMES) == 31
    assert CORE22_FEATURE_NAMES[:5] == CORE5_FEATURE_NAMES
    assert CORE_FEATURE_NAMES[:22] == CORE22_FEATURE_NAMES
    assert len(set(CORE_FEATURE_NAMES)) == 31
    assert set(FEATURE_SETS) == {"core5", "core22", "core"}


def test_port_range_boundaries():
    p = pd.Series([0, 1, 80, 1023, 1024, 49151, 49152, 65535, np.nan, "x"])
    out = port_range(p, p.index)
    assert list(out) == ["unknown", "wellknown", "wellknown", "wellknown", "registered", "registered",
                         "dynamic", "dynamic", "unknown", "unknown"]


def test_connection_outcome_from_state_codes():
    s = pd.Series(["SF", "S0", "REJ", "RSTO", "RSTR", "S1", "OTH", "SH", "FIN", "RST", "REQ", "INT", "CON", "-", "",
                   "somethingnew"])
    out = list(conn_outcome_from_state(s))
    assert out == ["complete", "unanswered", "reset", "reset", "reset", "other", "other", "unanswered", "complete",
                   "reset", "unanswered", "other", "complete", "unknown", "unknown", "other"]


def test_connection_outcome_from_flag_counts():
    idx = pd.RangeIndex(5)
    fin = pd.Series([1, 0, 0, 0, 1]); syn = pd.Series([1, 1, 1, 0, 1])
    rst = pd.Series([0, 0, 1, 0, 1]); ack = pd.Series([1, 0, 0, 0, 1])
    out = list(conn_outcome_from_flags(idx, fin, syn, rst, ack))
    # complete | SYN without ACK | reset wins | no flags (UDP) | reset wins over FIN
    assert out == ["complete", "unanswered", "reset", "other", "reset"]
    assert list(conn_outcome_from_flags(idx, None, None, None, None)) == ["unknown"] * 5


def test_flag_columns_resolve_for_both_cic_spellings():
    cic17 = [" FIN Flag Count", " SYN Flag Count", " RST Flag Count", " ACK Flag Count", " Destination Port"]
    f17 = resolve_columns(cic17, "cicids2017")
    assert all(f17[k] is not None for k in ("flag_fin", "flag_syn", "flag_rst", "flag_ack"))
    cic18 = ["FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "ACK Flag Cnt", "Dst Port"]
    f18 = resolve_columns(cic18, "cse_cic_ids2018")
    assert all(f18[k] is not None for k in ("flag_fin", "flag_syn", "flag_rst", "flag_ack"))
    assert resolve_columns(["conn_state"], "iot23")["conn_state"] == "conn_state"
    assert resolve_columns(["state"], "unsw_nb15")["conn_state"] == "state"


def test_one_hot_groups_sum_to_one():
    df = pd.DataFrame({"dur": [0.1, 2.0, 0.0], "proto": ["tcp", "udp", "icmp"], "service": ["http", "-", "dns"],
                       "sbytes": [10, 20, 30], "dbytes": [5, 0, 1], "spkts": [1, 2, 3], "dpkts": [1, 0, 1],
                       "state": ["FIN", "INT", "REQ"]})
    canon, _ = extract_canonical(df, "unsw_nb15")
    X, names = build_features(canon, "core")
    assert X.shape == (3, 31)
    for prefix in ("proto_", "svc_", "port_", "conn_"):
        cols = [i for i, n in enumerate(names) if n.startswith(prefix)]
        assert (X[:, cols].sum(axis=1) == 1).all(), prefix
    assert (X[:, names.index("port_unknown")] == 1).all()            # UNSW ships no destination port
    assert list(X[:, names.index("conn_complete")]) == [1, 0, 0]
    X22, n22 = build_features(canon, "core22")
    X5, n5 = build_features(canon, "core5")
    assert X22.shape[1] == 22 and X5.shape[1] == 5
    assert np.allclose(X[:, :22], X22) and np.allclose(X22[:, :5], X5)


# ------------------------------------------------------------------------------------------------
# Loader tests (fixtures with realistic headers)
# ------------------------------------------------------------------------------------------------
def _cic_frame(n, seed, with_flags=True, with_protocol=False, labels=("BENIGN", "DDoS")):
    rng = np.random.default_rng(seed)
    atk = rng.random(n) < 0.5
    port = np.where(atk, rng.choice([80, 8080], n), rng.choice([443, 53, 22], n))
    d = {
        " Destination Port": port,
        " Flow Duration": np.where(atk, rng.integers(1, 5_000, n), rng.integers(10_000, 5_000_000, n)),
        " Total Fwd Packets": np.where(atk, rng.integers(1, 4, n), rng.integers(2, 60, n)),
        " Total Backward Packets": np.where(atk, rng.integers(0, 2, n), rng.integers(2, 60, n)),
        "Total Length of Fwd Packets": np.where(atk, rng.integers(0, 120, n), rng.integers(100, 50_000, n)),
        " Total Length of Bwd Packets": np.where(atk, rng.integers(0, 40, n), rng.integers(100, 80_000, n)),
    }
    if with_flags:
        d[" FIN Flag Count"] = np.where(atk, 0, 1)
        d[" SYN Flag Count"] = np.where(atk, 1, 1)
        d[" RST Flag Count"] = 0
        d[" ACK Flag Count"] = np.where(atk, 0, 1)
    if with_protocol:
        d[" Protocol"] = 6
    d[" Label"] = np.where(atk, labels[1], labels[0])
    return pd.DataFrame(d), atk


def test_cic_flag_features_reach_the_model(tmp_path):
    df, atk = _cic_frame(800, 1)
    df.to_csv(tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", index=False)
    cfg = DatasetConfig(name="cicids2017", n_samples=600, data_dir=str(tmp_path), deduplicate=False, use_cache=False)
    ds = get_dataset_loader("cicids2017").load(cfg, seed=0)
    names = ds.manifest["feature_names"]
    unanswered = ds.X_feat[:, names.index("conn_unanswered")]
    complete = ds.X_feat[:, names.index("conn_complete")]
    assert np.array_equal(unanswered.astype(bool), ds.y.astype(bool))          # attacks: SYN without ACK
    assert np.array_equal(complete.astype(bool), ~ds.y.astype(bool))
    assert ds.manifest["connection_source_fields"]["flag_syn"] == "SYN Flag Count"


def test_multi_file_pooling_different_headers_and_latin1(tmp_path):
    a, _ = _cic_frame(500, 2, with_flags=True)
    b, _ = _cic_frame(500, 3, with_flags=False, labels=("BENIGN", "Web Attack \x96 Brute Force"))   # cp1252 en dash
    a.to_csv(tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", index=False)
    b.to_csv(tmp_path / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", index=False, encoding="latin-1")
    base = dict(name="cicids2017", n_samples=900, data_dir=str(tmp_path), deduplicate=False, use_cache=False)

    one = get_dataset_loader("cicids2017").load(DatasetConfig(pool="first", **base), seed=1)
    assert one.manifest["source_files"] == ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
    assert set(one.manifest["attack_type_map"].values()) == {"Benign", "DDoS"}

    both = get_dataset_loader("cicids2017").load(DatasetConfig(pool="all", **base), seed=1)
    assert len(both.manifest["source_files"]) == 2
    assert any("Web Attack" in v for v in both.manifest["attack_type_map"].values())
    assert "DDoS" in both.manifest["attack_type_map"].values()
    assert np.isfinite(both.X).all() and len(both.y) == 900
    # the flag columns exist in only one of the two files -> reported as partially missing
    assert "flag_syn" in both.manifest["fields_missing_in_some_files"]


def test_pool_cache_is_written_reused_and_keyed_on_the_files(tmp_path):
    df, _ = _cic_frame(600, 4)
    f = tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    df.to_csv(f, index=False)
    cache = tmp_path / "cache"
    cfg = DatasetConfig(name="cicids2017", n_samples=400, data_dir=str(tmp_path), cache_dir=str(cache), deduplicate=False)
    ld = get_dataset_loader("cicids2017")
    d1 = ld.load(cfg, seed=5)
    pkls = sorted(os.listdir(cache))
    assert len(pkls) == 1 and pkls[0].endswith(".pkl")
    d2 = ld.load(cfg, seed=5)                                              # served from the cache
    assert np.array_equal(d1.X, d2.X) and np.array_equal(d1.y, d2.y)
    ld.load(cfg, seed=6)                                                   # another seed -> another pool
    assert len(os.listdir(cache)) == 2
    df2, _ = _cic_frame(700, 4)
    df2.to_csv(f, index=False)                                             # file changed -> new key
    ld.load(cfg, seed=5)
    assert len(os.listdir(cache)) == 3
    off = cfg.model_copy(update={"use_cache": False, "cache_dir": str(tmp_path / "other")})
    ld.load(off, seed=5)
    assert not (tmp_path / "other").exists()


def test_iot23_detailed_malicious_categories(tmp_path):
    n = 600
    rng = np.random.default_rng(7)
    atk = rng.random(n) < 0.6
    cat = rng.choice(["PartOfAHorizontalPortScan", "DDoS", "C&C"], n)
    lab = np.where(atk, [f"(empty)   Malicious   {c}" for c in cat], "(empty)   Benign   -")
    pd.DataFrame({
        "ts": np.arange(n), "id.resp_p": rng.choice([80, 23, 6667], n), "proto": "tcp", "service": "-",
        "duration": rng.exponential(1.0, n), "orig_bytes": rng.integers(0, 500, n), "resp_bytes": rng.integers(0, 500, n),
        "conn_state": rng.choice(["S0", "SF", "REJ"], n), "orig_pkts": rng.integers(1, 9, n),
        "resp_pkts": rng.integers(0, 9, n), "tunnel_parents   label   detailed-label": lab,
    }).to_csv(tmp_path / "dataset5.csv", index=False)
    cfg = DatasetConfig(name="iot23", n_samples=500, data_dir=str(tmp_path), deduplicate=False, use_cache=False)
    ds = get_dataset_loader("iot23").load(cfg, seed=0)
    assert set(ds.manifest["attack_type_map"].values()) == {"Benign", "PartOfAHorizontalPortScan", "DDoS", "C&C"}
    assert set(np.unique(ds.y)) == {0, 1}
    assert ds.manifest["attack_type_counts"]["Benign"] == int((ds.y == 0).sum())
    names = ds.manifest["feature_names"]
    assert ds.X_feat[:, names.index("conn_unanswered")].sum() > 0 and ds.X_feat[:, names.index("conn_reset")].sum() > 0


def test_describe_reports_nothing_hand_typed(tmp_path):
    df, atk = _cic_frame(500, 8)
    df.to_csv(tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", index=False)
    cfg = DatasetConfig(name="cicids2017", data_dir=str(tmp_path), use_cache=False)
    d = get_dataset_loader("cicids2017").describe(cfg)
    assert d["n_rows"] == 500 and d["native_label_classes"] == 2
    assert d["attack_share"] == pytest.approx(atk.mean(), abs=1e-3)
    assert d["fields_found"]["duration"] == "Flow Duration" or d["fields_found"]["duration"].strip() == "Flow Duration"
    assert "protocol" in d["fields_missing"] and "conn_state" in d["fields_missing"]
    assert 0.0 <= d["canonical_duplicate_rate"] <= 1.0


def test_unknown_feature_set_rejected():
    with pytest.raises(ValueError):
        build_features(pd.DataFrame({"duration": [1.0]}), "nope")
