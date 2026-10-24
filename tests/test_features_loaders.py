"""
Loader / feature tests that need no real data: each fixture mimics the REAL header style of its corpus
(leading-space CICIDS2017 headers, repeated header rows in CSE-CIC-IDS2018, Zeek '-' placeholders and a
compound label column in IoT-23, ...), so a mapping regression (like the CICIDS2017 one) is caught.
"""
import numpy as np
import pandas as pd
import pytest

from fedids_bench.config import DatasetConfig
from fedids_bench.data.features import (
    CORE_FEATURE_NAMES, CORE5_FEATURE_NAMES, build_features, extract_canonical,
    normalize_protocol, resolve_columns, service_group,
)
from fedids_bench.data.loaders import get_dataset_loader


def _flows(n, seed=0, attack_share=0.5):
    """Generic informative flows: attacks are short, tiny, mostly UDP / unusual ports."""
    rng = np.random.default_rng(seed)
    atk = rng.random(n) < attack_share
    dur = np.where(atk, rng.exponential(0.05, n), rng.exponential(2.0, n))
    fb = np.where(atk, rng.integers(0, 120, n), rng.integers(100, 50_000, n))
    bb = np.where(atk, rng.integers(0, 40, n), rng.integers(100, 80_000, n))
    fp = np.where(atk, rng.integers(1, 4, n), rng.integers(2, 60, n))
    bp = np.where(atk, rng.integers(0, 2, n), rng.integers(2, 60, n))
    port = np.where(atk, rng.choice([23, 445, 31337, 6667], n), rng.choice([80, 443, 53, 22], n))
    proto = np.where(atk, rng.choice(["udp", "tcp"], n, p=[0.8, 0.2]), rng.choice(["tcp", "udp"], n, p=[0.9, 0.1]))
    return atk, dict(dur=dur, fb=fb, bb=bb, fp=fp, bp=bp, port=port, proto=proto)


def _write_unsw(path, n=600, seed=0):
    atk, f = _flows(n, seed)
    cat = np.where(atk, "Exploits", "Normal")
    pd.DataFrame({
        "id": np.arange(n), "dur": f["dur"], "proto": f["proto"], "service": np.where(atk, "-", "http"),
        "state": "FIN", "spkts": f["fp"], "dpkts": f["bp"], "sbytes": f["fb"], "dbytes": f["bb"],
        "attack_cat": cat, "label": atk.astype(int),
    }).to_csv(path, index=False)


def _write_cicids(path, n=600, seed=1):
    atk, f = _flows(n, seed)
    # real CICIDS2017 headers carry LEADING SPACES (except a few)
    pd.DataFrame({
        " Destination Port": f["port"], " Flow Duration": (f["dur"] * 1e6).astype(int),
        " Total Fwd Packets": f["fp"], " Total Backward Packets": f["bp"],
        "Total Length of Fwd Packets": f["fb"], " Total Length of Bwd Packets": f["bb"],
        " Label": np.where(atk, "DDoS", "BENIGN"),
    }).to_csv(path, index=False)


def _write_cse(path, n=600, seed=2):
    atk, f = _flows(n, seed)
    df = pd.DataFrame({
        "Dst Port": f["port"], "Protocol": np.where(f["proto"] == "udp", 17, 6), "Timestamp": "21/02/2018 10:00:00",
        "Flow Duration": (f["dur"] * 1e6).astype(int), "Tot Fwd Pkts": f["fp"], "Tot Bwd Pkts": f["bp"],
        "TotLen Fwd Pkts": f["fb"], "TotLen Bwd Pkts": f["bb"], "Label": np.where(atk, "DDOS attack-HOIC", "Benign"),
    })
    rep = pd.DataFrame([dict(zip(df.columns, df.columns))])           # repeated header row inside the data
    pd.concat([df.iloc[:300], rep, df.iloc[300:]], ignore_index=True).to_csv(path, index=False)


def _write_iot23(path, n=600, seed=3):
    atk, f = _flows(n, seed)
    dash = lambda a: pd.Series(a).astype(object).where(np.arange(n) % 7 != 0, "-")   # Zeek '-' placeholders
    lab = np.where(atk, "(empty)   Malicious   PartOfAHorizontalPortScan", "(empty)   Benign   -")
    pd.DataFrame({
        "ts": 1.5e9 + np.arange(n), "uid": [f"C{i}" for i in range(n)], "id.orig_h": "192.168.1.5",
        "id.orig_p": 40000, "id.resp_h": "10.0.0.1", "id.resp_p": f["port"], "proto": f["proto"],
        "service": np.where(f["port"] == 53, "dns", "-"), "duration": dash(f["dur"]), "orig_bytes": dash(f["fb"]),
        "resp_bytes": dash(f["bb"]), "conn_state": "SF", "orig_pkts": f["fp"], "resp_pkts": f["bp"],
        "tunnel_parents   label   detailed-label": lab,
    }).to_csv(path, index=False)


def _write_ton(path, n=600, seed=4):
    atk, f = _flows(n, seed)
    pd.DataFrame({
        "ts": 1.5e9 + np.arange(n), "src_ip": "1.1.1.1", "src_port": 4444, "dst_ip": "2.2.2.2", "dst_port": f["port"],
        "proto": f["proto"], "service": np.where(f["port"] == 53, "dns", "-"), "duration": f["dur"],
        "src_bytes": f["fb"], "dst_bytes": f["bb"], "conn_state": "SF", "src_pkts": f["fp"], "dst_pkts": f["bp"],
        "label": atk.astype(int), "type": np.where(atk, "ddos", "normal"),
    }).to_csv(path, index=False)


WRITERS = {
    "unsw_nb15": (_write_unsw, "UNSW_NB15_training-set.csv"),
    "cicids2017": (_write_cicids, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    "cse_cic_ids2018": (_write_cse, "02-21-2018.csv"),
    "iot23": (_write_iot23, "dataset5.csv"),
    "ton_iot": (_write_ton, "train_test_network.csv"),
}


def _cfg(tmp_path, name, **kw):
    writer, fname = WRITERS[name]
    writer(tmp_path / fname)
    base = dict(name=name, n_samples=400, test_fraction=0.2, val_fraction=0.1, data_dir=str(tmp_path))
    base.update(kw)
    return DatasetConfig(**base)


@pytest.mark.parametrize("name", list(WRITERS))
def test_loader_shapes_splits_and_zero_leakage(tmp_path, name):
    cfg = _cfg(tmp_path, name, deduplicate=False)
    ds = get_dataset_loader(name).load(cfg, seed=7)
    assert ds.X.shape == (400, len(CORE_FEATURE_NAMES)) and ds.X.dtype == np.float32
    sc = ds.manifest["split_counts"]
    assert sc["train"] + sc["val"] + sc["test"] == 400
    assert sc["test"] == 80 and sc["val"] == 40                         # 20% / 10%
    # stratified: class prevalence is (almost) identical in every split
    prev = ds.manifest["attack_prevalence"]
    assert max(prev.values()) - min(prev.values()) < 0.03
    # scaler fitted on TRAIN only: train columns are standardized, test columns are not exactly
    tr = ds.X[ds.split == 0]
    varying = tr.std(axis=0) > 1e-6
    assert np.allclose(tr.mean(axis=0)[varying], 0.0, atol=1e-4)
    te = ds.X[ds.split == 2]
    assert not np.allclose(te.mean(axis=0)[varying], 0.0, atol=1e-6)
    assert ds.X_feat is not None and ds.X_feat.shape == ds.X.shape
    assert set(np.unique(ds.y)) == {0, 1}


def test_cicids_leading_space_headers_are_mapped():
    """Regression: with stripped headers a mapping keyed on ' Flow Duration' silently produced zeros."""
    cols = [" Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets",
            "Total Length of Fwd Packets", " Total Length of Bwd Packets", " Label"]
    found = resolve_columns(cols, "cicids2017")
    for field in ["duration", "fwd_bytes", "bwd_bytes", "fwd_pkts", "bwd_pkts", "dst_port"]:
        assert found[field] is not None, field
    assert found["protocol"] is None            # MachineLearningCVE CSVs ship no protocol column


def test_cicids_end_to_end_no_constant_volume_features(tmp_path):
    ds = get_dataset_loader("cicids2017").load(_cfg(tmp_path, "cicids2017", deduplicate=False), seed=3)
    names = ds.manifest["feature_names"]
    for f in CORE5_FEATURE_NAMES:                                   # all five volume features must vary
        assert ds.X_feat[:, names.index(f)].std() > 0, f
    assert ds.manifest["canonical_fields_found"]["protocol"] is None
    assert (ds.X_feat[:, names.index("proto_unknown")] == 1).all()   # explicit 'unknown', not a fake number


def test_units_microseconds_to_seconds():
    df = pd.DataFrame({" Flow Duration": [2_000_000], "Total Length of Fwd Packets": [10],
                       " Total Length of Bwd Packets": [5], " Total Fwd Packets": [2], " Total Backward Packets": [1]})
    canon, _ = extract_canonical(df, "cicids2017")
    assert canon["duration"].iloc[0] == pytest.approx(2.0)
    X, names = build_features(canon, "core5")
    assert X[0, names.index("log_duration_ms")] == pytest.approx(np.log1p(2000.0), rel=1e-5)


def test_availability_unsw_has_service_but_no_port(tmp_path):
    ds = get_dataset_loader("unsw_nb15").load(_cfg(tmp_path, "unsw_nb15"), seed=1)
    found = ds.manifest["canonical_fields_found"]
    assert found["dst_port"] is None and found["service"] == "service" and found["protocol"] == "proto"


def test_iot23_placeholders_and_compound_label(tmp_path):
    ds = get_dataset_loader("iot23").load(_cfg(tmp_path, "iot23", deduplicate=False), seed=2)
    assert np.isfinite(ds.X).all()
    assert set(np.unique(ds.y)) == {0, 1}
    assert ds.manifest["canonical_fields_found"]["dst_port"] == "id.resp_p"


def test_cse_cic_repeated_header_row_removed(tmp_path):
    ds = get_dataset_loader("cse_cic_ids2018").load(_cfg(tmp_path, "cse_cic_ids2018", n_samples=500, deduplicate=False), seed=2)
    assert len(ds.y) == 500 and np.isfinite(ds.X).all()


def test_sampling_is_uniform_over_the_whole_file(tmp_path):
    """The old loaders read the first N rows only; a time-ordered file then gives a biased slice."""
    n = 4000
    atk = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(bool)         # first half attacks, second half benign
    f = _flows(n, 5)[1]
    pd.DataFrame({
        "ts": np.arange(n), "id.resp_p": f["port"], "proto": f["proto"], "service": "-", "duration": f["dur"],
        "orig_bytes": f["fb"], "resp_bytes": f["bb"], "orig_pkts": f["fp"], "resp_pkts": f["bp"],
        "tunnel_parents   label   detailed-label": np.where(atk, "(empty)   Malicious   x", "(empty)   Benign   -"),
    }).to_csv(tmp_path / "dataset5.csv", index=False)
    cfg = DatasetConfig(name="iot23", n_samples=800, data_dir=str(tmp_path), deduplicate=False)
    ds = get_dataset_loader("iot23").load(cfg, seed=0)
    assert 0.40 < ds.y.mean() < 0.60


def test_exact_duplicates_are_removed_before_splitting(tmp_path):
    atk, f = _flows(300, 9)
    base = pd.DataFrame({"ts": 0, "dst_port": f["port"], "proto": f["proto"], "service": "-", "duration": f["dur"],
                         "src_bytes": f["fb"], "dst_bytes": f["bb"], "src_pkts": f["fp"], "dst_pkts": f["bp"],
                         "label": atk.astype(int), "type": np.where(atk, "ddos", "normal")})
    pd.concat([base, base.iloc[:100]], ignore_index=True).to_csv(tmp_path / "train_test_network.csv", index=False)
    cfg = DatasetConfig(name="ton_iot", n_samples=10_000, data_dir=str(tmp_path), deduplicate=True)
    ds = get_dataset_loader("ton_iot").load(cfg, seed=0)
    assert ds.manifest["n_duplicates_dropped"] >= 100
    assert len(ds.y) <= 300


def test_protocol_and_service_normalization():
    p = normalize_protocol(pd.Series(["tcp", "UDP", 6, 17, 1, "arp", "-", "icmp"]))
    assert list(p) == ["tcp", "udp", "tcp", "udp", "icmp", "other", "unknown", "icmp"]
    s = service_group(pd.Series([80, 53, 22, 445, 9999, np.nan]), pd.Series(["-", "-", "-", "-", "-", "-"]), pd.RangeIndex(6))
    assert list(s) == ["web", "dns", "remote", "mail_file", "other", "unknown"]
    # a named service wins over the port
    s2 = service_group(pd.Series([8080]), pd.Series(["dns"]), pd.RangeIndex(1))
    assert list(s2) == ["dns"]
