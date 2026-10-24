"""
Discovery and real-file quirks that no earlier test could exercise because no real corpus was ever available here:
mixed-case and nested file names, two candidate folders, macOS resource files, a UTF-8 BOM, raw Zeek logs for IoT-23,
separate label columns, header-less UNSW files and the completeness check.  Every fixture copies a layout or header that the
published corpus is documented to have; none of this is real data.
"""
import logging
import os

import numpy as np
import pandas as pd
import pytest

from fedids_bench.config import DatasetConfig
from fedids_bench.data import loaders as L
from fedids_bench.data.discovery import MAX_DEPTH, find_files, name_matches
from fedids_bench.data.features import CORE5_FEATURE_NAMES, clean_header
from fedids_bench.data.loaders import _count_rows, get_dataset_loader, zeek_fields
from tests.test_features_loaders import _flows, _write_cicids, _write_cse

CIC_DAYS = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",                       # sic: lower-case "w" in the published archive
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


@pytest.fixture(autouse=True)
def _fresh_warning_registry():
    L._WARNED.clear()
    yield
    L._WARNED.clear()


def _cfg(root, name, **kw):
    base = dict(name=name, n_samples=300, test_fraction=0.2, val_fraction=0.1, data_dir=str(root), pool="all", use_cache=False)
    base.update(kw)
    return DatasetConfig(**base)


def _cic_dir(d, names=CIC_DAYS, n=120):
    d.mkdir(parents=True, exist_ok=True)
    for i, nm in enumerate(names):
        _write_cicids(d / nm, n=n, seed=10 + i)
    return d


# ------------------------------------------------------------------------------------------------ discovery
def test_cicids_nested_mixed_case_all_eight_days_found(tmp_path):
    _cic_dir(tmp_path / "MachineLearningCSV" / "MachineLearningCVE")
    ld, cfg = get_dataset_loader("cicids2017"), _cfg(tmp_path, "cicids2017")
    files = ld.locate_all(cfg)
    assert len(files) == 8
    assert "Wednesday-workingHours.pcap_ISCX.csv" in [os.path.basename(f) for f in files]
    info = ld.describe(cfg, with_duplicates=False)
    assert info["n_files"] == 8 and info["files_expected"] == 8 and info["file_count_ok"] is True
    assert info["n_rows"] == 8 * 120
    ds = ld.load(cfg, seed=1)
    assert ds.manifest["source_dir"].endswith("MachineLearningCVE") and len(ds.manifest["source_files"]) == 8


def test_missing_day_is_reported_not_silent(tmp_path, caplog):
    _cic_dir(tmp_path / "cic", CIC_DAYS[:7])
    ld, cfg = get_dataset_loader("cicids2017"), _cfg(tmp_path, "cicids2017")
    with caplog.at_level(logging.WARNING, logger="fedids_bench.data"):
        info = ld.describe(cfg, with_duplicates=False)
    assert info["n_files"] == 7 and info["file_count_ok"] is False
    assert any("published corpus has 8" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="fedids_bench.data"):          # warned once per process, not once per run
        ld.locate_all(cfg)
        ld.locate_all(cfg)
    assert not any("published corpus has 8" in r.getMessage() for r in caplog.records)


def test_two_candidate_folders_are_never_pooled(tmp_path, caplog):
    _cic_dir(tmp_path / "MachineLearningCVE")
    _cic_dir(tmp_path / "TrafficLabelling")                                         # same names, same count (a tie)
    ld, cfg = get_dataset_loader("cicids2017"), _cfg(tmp_path, "cicids2017")
    with caplog.at_level(logging.WARNING, logger="fedids_bench.data"):
        files = ld.locate_all(cfg)
    assert len(files) == 8 and len({os.path.dirname(f) for f in files}) == 1       # one folder, deterministic tie-break
    assert any("IGNORED" in r.getMessage() for r in caplog.records)
    info = ld.describe(cfg, with_duplicates=False)
    assert info["n_rows"] == 8 * 120 and info["ignored_directories"]


def test_the_folder_with_more_files_wins(tmp_path):
    _cic_dir(tmp_path / "A_partial", CIC_DAYS[:5])
    _cic_dir(tmp_path / "B_complete")
    files = get_dataset_loader("cicids2017").locate_all(_cfg(tmp_path, "cicids2017"))
    assert len(files) == 8 and all("B_complete" in f for f in files)


def test_macos_resource_files_and_hidden_folders_are_ignored(tmp_path):
    d = _cic_dir(tmp_path / "x", CIC_DAYS[:2])
    (d / "._Monday-WorkingHours.pcap_ISCX.csv").write_bytes(b"\x00\x05\x16\x07 resource fork")
    hidden = tmp_path / ".fedids_cache" / "y"
    hidden.mkdir(parents=True)
    _write_cicids(hidden / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", n=50)
    files = get_dataset_loader("cicids2017").locate_all(_cfg(tmp_path, "cicids2017"))
    assert len(files) == 2 and not any(os.path.basename(f).startswith(".") for f in files)
    assert not any(".fedids_cache" in f for f in files)


def test_not_found_message_names_what_was_searched(tmp_path):
    with pytest.raises(FileNotFoundError, match="case-insensitively"):
        get_dataset_loader("ton_iot").locate_all(_cfg(tmp_path, "ton_iot"))


def test_unsw_raw_headerless_files_give_a_helpful_error(tmp_path):
    (tmp_path / "UNSW-NB15_1.csv").write_text("59.166.0.9,1390,149.171.126.6,53,udp,CON,0.001055,132,164,31,29,0,0,dns\n")
    with pytest.raises(FileNotFoundError, match="PARTITIONED"):
        get_dataset_loader("unsw_nb15").locate_all(_cfg(tmp_path, "unsw_nb15"))


def test_ton_iot_full_network_captures_give_a_helpful_error(tmp_path):
    (tmp_path / "Network_dataset_1.csv").write_text("ts,src_ip\n1,2\n")
    with pytest.raises(FileNotFoundError, match="train_test_network.csv"):
        get_dataset_loader("ton_iot").locate_all(_cfg(tmp_path, "ton_iot"))


# --------------------------------------------------------------------------------------- find_files, unit level
def test_name_matches_is_case_insensitive():
    assert name_matches("Wednesday-workingHours.pcap_ISCX.csv", ["*.PCAP_iscx.CSV"])
    assert not name_matches("notes.txt", ["*.csv"])


def test_find_files_depth_limits(tmp_path):
    deep = tmp_path.joinpath(*"abcdef")                                               # 6 levels below the root
    deep.mkdir(parents=True)
    (deep / "x.csv").write_text("a\n")
    assert MAX_DEPTH < 6
    assert find_files([str(tmp_path)], ["x.csv"]) is None                              # flat corpora: bounded
    assert find_files([str(tmp_path)], [], tree_patterns=["x.csv"]) is not None        # tree corpora may sit deeper


def test_find_files_first_mode_prefers_the_names_order(tmp_path):
    (tmp_path / "B.csv").write_text("x\n")
    (tmp_path / "a.csv").write_text("x\n")
    f = find_files([str(tmp_path)], ["*.csv"], names=["a.CSV", "b.csv"], pool_all=False)
    assert [os.path.basename(p) for p in f.files] == ["a.csv"]


def test_find_files_literal_names_may_contain_glob_characters(tmp_path):
    (tmp_path / "data[1].csv").write_text("x\n")
    assert find_files([str(tmp_path)], [], names=["data[1].csv"], pool_all=False) is not None


@pytest.mark.parametrize("text,header,expected", [
    ("a\n1\n2\n", 1, 2), ("a\n1\n2", 1, 2), ("#x\n#y\n1\n2\n#close\n", 3, 2), ("", 1, 0)])
def test_count_rows_blockwise(tmp_path, text, header, expected):
    p = tmp_path / "f.csv"
    p.write_text(text)
    assert _count_rows(str(p), header) == expected


# ---------------------------------------------------------------------------------------------- header quirks
def test_utf8_bom_in_front_of_the_first_header_cell_is_tolerated(tmp_path):
    d = tmp_path / "cic"
    d.mkdir()
    f = d / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    _write_cicids(f, n=300)
    f.write_bytes(b"\xef\xbb\xbf" + f.read_bytes())                                   # BOM + " Destination Port"
    ds = get_dataset_loader("cicids2017").load(_cfg(tmp_path, "cicids2017", pool="first"), seed=1)
    assert ds.manifest["canonical_fields_found"]["dst_port"] is not None
    assert clean_header("\u00ef\u00bb\u00bf Destination Port") == "Destination Port"


def test_cse_cic_2018_tuesday_file_with_extra_id_columns_and_unb_file_names(tmp_path):
    d = tmp_path / "Processed Traffic Data for ML Algorithms"
    d.mkdir()
    _write_cse(d / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", n=200, seed=21)
    # the 02-20 file carries four extra leading identifier columns
    atk, f = _flows(200, 22)
    extra = pd.DataFrame({
        "Flow ID": "1.1.1.1-2.2.2.2-1-2-6", "Src IP": "1.1.1.1", "Src Port": 1234, "Dst IP": "2.2.2.2",
        "Dst Port": f["port"], "Protocol": 6, "Timestamp": "20/02/2018 10:00:00", "Flow Duration": (f["dur"] * 1e6).astype(int),
        "Tot Fwd Pkts": f["fp"], "Tot Bwd Pkts": f["bp"], "TotLen Fwd Pkts": f["fb"], "TotLen Bwd Pkts": f["bb"],
        "Label": np.where(atk, "DDoS attacks-LOIC-HTTP", "Benign"),
    })
    rep = pd.DataFrame([dict(zip(extra.columns, extra.columns))])
    pd.concat([extra.iloc[:100], rep, extra.iloc[100:]], ignore_index=True).to_csv(
        d / "Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", index=False)
    ld, cfg = get_dataset_loader("cse_cic_ids2018"), _cfg(tmp_path, "cse_cic_ids2018")
    assert len(ld.locate_all(cfg)) == 2
    info = ld.describe(cfg, with_duplicates=False)
    assert info["n_rows"] == 2 * 200 and info["file_count_ok"] is False              # 2 of the 10 published days
    ds = ld.load(cfg, seed=3)
    for name in CORE5_FEATURE_NAMES:
        assert ds.X_feat[:, ds.manifest["feature_names"].index(name)].std() > 0, name
    assert "label" not in {k.lower() for k in ds.manifest["attack_type_map"].values()}   # repeated header rows were dropped


# ---------------------------------------------------------------------------------------------------- IoT-23
ZEEK_HEADER = (
    "#separator \\x09\n#set_separator\t,\n#empty_field\t(empty)\n#unset_field\t-\n#path\tconn\n#open\t2018-10-03-14-36-55\n"
    "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state"
    "\tlocal_orig\tlocal_resp\tmissed_bytes\thistory\torig_pkts\torig_ip_bytes\tresp_pkts\tresp_ip_bytes"
    "\ttunnel_parents   label   detailed-label\n"
    "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\tcount\tcount\tstring\tbool\tbool\tcount\tstring"
    "\tcount\tcount\tcount\tcount\tset[string]\tstring\tstring\n"
)


def _write_zeek(path, n=300, seed=5, attack="PartOfAHorizontalPortScan"):
    atk, f = _flows(n, seed)
    rows = []
    for i in range(n):
        gap = i % 11 == 0                                                              # Zeek writes '-' for unset fields
        lab = f"(empty)   Malicious   {attack}" if atk[i] else "(empty)   Benign   -"
        rows.append("\t".join([
            f"{1.5e9 + i:.6f}", f"C{i:016d}", "192.168.1.5", "40000", "10.0.0.1", str(f["port"][i]), str(f["proto"][i]),
            "dns" if f["port"][i] == 53 else "-", "-" if gap else f"{f['dur'][i]:.6f}", "-" if gap else str(int(f["fb"][i])),
            "-" if gap else str(int(f["bb"][i])), "SF", "-", "-", "0", "ShADad", str(int(f["fp"][i])), "100",
            str(int(f["bp"][i])), "100", lab]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ZEEK_HEADER + "\n".join(rows) + "\n#close\t2018-10-03-15-28-49\n")


def test_iot23_raw_zeek_logs_in_the_nested_archive_layout(tmp_path):
    base = tmp_path / "iot_23_datasets_full" / "opt" / "Malware-Project" / "BigDataset" / "IoTScenarios"
    _write_zeek(base / "CTU-IoT-Malware-Capture-1-1" / "bro" / "conn.log.labeled", seed=5)
    _write_zeek(base / "CTU-IoT-Malware-Capture-34-1" / "bro" / "conn.log.labeled", seed=6, attack="C&C")
    ld, cfg = get_dataset_loader("iot23"), _cfg(tmp_path, "iot23")
    files = ld.locate_all(cfg)
    assert len(files) == 2
    assert zeek_fields(files[0])[-1] == "tunnel_parents   label   detailed-label" and zeek_fields(files[0])[0] == "ts"
    info = ld.describe(cfg, with_duplicates=False)
    assert info["n_rows"] == 600                                                       # '#' lines are not rows
    assert set(info["label_counts_top20"]) == {"Benign", "PartOfAHorizontalPortScan", "C&C"}
    assert info["file_count_ok"] is True and "dst_port" not in info["fields_missing"]
    ds = ld.load(cfg, seed=2)
    names = ds.manifest["feature_names"]
    for f in CORE5_FEATURE_NAMES:
        assert ds.X_feat[:, names.index(f)].std() > 0, f
    assert set(np.unique(ds.y)) == {0, 1}
    assert ds.manifest["canonical_fields_found"]["dst_port"] == "id.resp_p"
    assert "C&C" in ds.manifest["attack_type_map"].values()
    assert (ds.X_feat[:, names.index("conn_complete")] == 1).all()                    # conn_state SF everywhere


def test_iot23_first_mode_reads_one_zeek_file(tmp_path):
    base = tmp_path / "scen"
    _write_zeek(base / "CTU-IoT-Malware-Capture-1-1" / "bro" / "conn.log.labeled", seed=5)
    _write_zeek(base / "CTU-IoT-Malware-Capture-3-1" / "bro" / "conn.log.labeled", seed=6)
    ld = get_dataset_loader("iot23")
    assert len(ld.locate_all(_cfg(tmp_path, "iot23", pool="first"))) == 1


def test_iot23_csv_with_separate_label_columns_keeps_dash_detail_attacks(tmp_path):
    n = 400
    atk, f = _flows(n, 7)
    detail = np.where(atk, np.where(np.arange(n) % 3 == 0, "-", "DDoS"), "-")          # 'Malicious' + '-' really occurs
    pd.DataFrame({
        "ts": 1.5e9 + np.arange(n), "id.resp_p": f["port"], "proto": f["proto"], "service": "-", "duration": f["dur"],
        "orig_bytes": f["fb"], "resp_bytes": f["bb"], "conn_state": "SF", "orig_pkts": f["fp"], "resp_pkts": f["bp"],
        "label": np.where(atk, "Malicious", "Benign"), "detailed-label": detail,
    }).to_csv(tmp_path / "dataset1.csv", index=False)
    ds = get_dataset_loader("iot23").load(_cfg(tmp_path, "iot23", deduplicate=False, n_samples=400), seed=4)
    by_name = {v: int(k) for k, v in ds.manifest["attack_type_map"].items()}
    assert set(by_name) == {"Benign", "DDoS", "Malicious"}
    assert (ds.y[ds.attack_type == by_name["Malicious"]] == 1).all()                  # a '-' detail never makes an attack benign
    assert (ds.y[ds.attack_type == by_name["Benign"]] == 0).all()
    assert (ds.y == 1).sum() == (ds.attack_type != by_name["Benign"]).sum()
