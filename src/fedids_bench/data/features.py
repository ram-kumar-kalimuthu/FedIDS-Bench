"""
Harmonized NetFlow-style feature construction for FedIDS-Bench.

Every corpus is first mapped onto the same *canonical* flow fields

    duration, fwd/bwd bytes, fwd/bwd packets, protocol, destination port, service,
    connection outcome (from a Zeek/Argus state code or from TCP flag counts)

and then expanded into ONE fixed feature vector:

    ``core``    31 features (default): log-scaled volume features, direction asymmetry / rate
                features, and one-hot protocol, service group, destination-port range and
                connection outcome
    ``core22``  the first 22 of them (no port range, no connection outcome)
    ``core5``   the 5 log-scaled volume features only (ablation)

Fields that a corpus does not ship (e.g. UNSW-NB15's training-set CSV has no destination port,
the CICIDS2017 "MachineLearningCVE" CSVs have no protocol column) are recorded as *unavailable*
in the manifest and encoded as the explicit ``unknown`` category -- never silently as a number.

Column matching is done on ``strip().lower()`` of the native header, because the CICIDS2017 CSVs
ship headers with leading spaces (" Flow Duration").  A mapping keyed on the un-stripped names
silently fails once headers are stripped, which turns four of the five CICIDS2017 features into
constant zeros.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CANONICAL_NUMERIC: List[str] = ["duration", "fwd_bytes", "bwd_bytes", "fwd_pkts", "bwd_pkts"]
CANONICAL_OTHER: List[str] = ["protocol", "dst_port", "service"]
CANONICAL_FIELDS: List[str] = CANONICAL_NUMERIC + CANONICAL_OTHER
# Native fields that only feed the connection-outcome feature (a state code, or TCP flag counts).
CONN_SOURCE_FIELDS: List[str] = ["conn_state", "flag_fin", "flag_syn", "flag_rst", "flag_ack"]
SOURCE_FIELDS: List[str] = CANONICAL_FIELDS + CONN_SOURCE_FIELDS

PROTOCOLS: List[str] = ["tcp", "udp", "icmp", "other", "unknown"]
SERVICE_GROUPS: List[str] = ["web", "dns", "remote", "mail_file", "other", "unknown"]
PORT_RANGES: List[str] = ["wellknown", "registered", "dynamic", "unknown"]
CONN_OUTCOMES: List[str] = ["complete", "reset", "unanswered", "other", "unknown"]

CORE5_FEATURE_NAMES: List[str] = [
    "log_duration_ms", "log_fwd_bytes", "log_bwd_bytes", "log_fwd_pkts", "log_bwd_pkts",
]
_DERIVED_NAMES: List[str] = [
    "log_fwd_pkt_size", "log_bwd_pkt_size", "byte_asym", "pkt_asym", "log_byte_rate", "log_pkt_rate",
]
CORE22_FEATURE_NAMES: List[str] = (
    CORE5_FEATURE_NAMES + _DERIVED_NAMES
    + [f"proto_{p}" for p in PROTOCOLS]
    + [f"svc_{g}" for g in SERVICE_GROUPS]
)
CORE_FEATURE_NAMES: List[str] = (
    CORE22_FEATURE_NAMES
    + [f"port_{r}" for r in PORT_RANGES]
    + [f"conn_{c}" for c in CONN_OUTCOMES]
)
FEATURE_SETS: Dict[str, List[str]] = {
    "core5": CORE5_FEATURE_NAMES, "core22": CORE22_FEATURE_NAMES, "core": CORE_FEATURE_NAMES,
}
FEATURE_GROUPS: Dict[str, List[str]] = {
    "volume": CORE5_FEATURE_NAMES,
    "size / asymmetry / rate": _DERIVED_NAMES,
    "protocol": [f"proto_{p}" for p in PROTOCOLS],
    "service group": [f"svc_{g}" for g in SERVICE_GROUPS],
    "destination-port range": [f"port_{r}" for r in PORT_RANGES],
    "connection outcome": [f"conn_{c}" for c in CONN_OUTCOMES],
}

# Native column names (compared after strip + lowercase), first match wins.
SOURCE_COLUMNS: Dict[str, Dict[str, List[str]]] = {
    "unsw_nb15": {
        "duration": ["dur"], "fwd_bytes": ["sbytes"], "bwd_bytes": ["dbytes"],
        "fwd_pkts": ["spkts"], "bwd_pkts": ["dpkts"],
        "protocol": ["proto"], "dst_port": ["dsport"], "service": ["service"],
        "conn_state": ["state"],
    },
    "cicids2017": {
        "duration": ["flow duration"], "fwd_bytes": ["total length of fwd packets"],
        "bwd_bytes": ["total length of bwd packets"], "fwd_pkts": ["total fwd packets"],
        "bwd_pkts": ["total backward packets"],
        "protocol": ["protocol"], "dst_port": ["destination port", "dst port"], "service": [],
        "flag_fin": ["fin flag count"], "flag_syn": ["syn flag count"],
        "flag_rst": ["rst flag count"], "flag_ack": ["ack flag count"],
    },
    "cse_cic_ids2018": {
        "duration": ["flow duration"], "fwd_bytes": ["totlen fwd pkts"],
        "bwd_bytes": ["totlen bwd pkts"], "fwd_pkts": ["tot fwd pkts"], "bwd_pkts": ["tot bwd pkts"],
        "protocol": ["protocol"], "dst_port": ["dst port", "destination port"], "service": [],
        "flag_fin": ["fin flag cnt", "fin flag count"], "flag_syn": ["syn flag cnt", "syn flag count"],
        "flag_rst": ["rst flag cnt", "rst flag count"], "flag_ack": ["ack flag cnt", "ack flag count"],
    },
    "iot23": {
        "duration": ["duration"], "fwd_bytes": ["orig_bytes"], "bwd_bytes": ["resp_bytes"],
        "fwd_pkts": ["orig_pkts"], "bwd_pkts": ["resp_pkts"],
        "protocol": ["proto"], "dst_port": ["id.resp_p", "id_resp_p", "resp_p"], "service": ["service"],
        "conn_state": ["conn_state"],
    },
    "ton_iot": {
        "duration": ["duration"], "fwd_bytes": ["src_bytes"], "bwd_bytes": ["dst_bytes"],
        "fwd_pkts": ["src_pkts"], "bwd_pkts": ["dst_pkts"],
        "protocol": ["proto"], "dst_port": ["dst_port"], "service": ["service"],
        "conn_state": ["conn_state"],
    },
}

# Multiply the native duration by this to obtain SECONDS.
DURATION_TO_SECONDS: Dict[str, float] = {
    "unsw_nb15": 1.0, "cicids2017": 1e-6, "cse_cic_ids2018": 1e-6, "iot23": 1.0, "ton_iot": 1.0,
}

_MISSING_TOKENS = {"", "-", "nan", "none", "null", "(empty)"}
_PROTO_NUMBERS = {6: "tcp", 17: "udp", 1: "icmp", 58: "icmp"}
_SERVICE_NAMES = {
    "http": "web", "https": "web", "ssl": "web", "http-alt": "web",
    "dns": "dns",
    "ssh": "remote", "telnet": "remote", "rdp": "remote", "vnc": "remote",
    "ftp": "mail_file", "ftp-data": "mail_file", "smtp": "mail_file", "pop3": "mail_file",
    "imap": "mail_file", "smb": "mail_file", "netbios-ssn": "mail_file", "microsoft-ds": "mail_file",
}
_PORT_GROUPS = {
    "web": {80, 443, 8000, 8080, 8443, 8888},
    "dns": {53, 5353},
    "remote": {22, 23, 3389, 5900},
    "mail_file": {20, 21, 25, 110, 139, 143, 445, 465, 587, 993, 995},
}
# Connection-state codes: Zeek ``conn_state`` (IoT-23, ToN-IoT) and Argus ``state`` (UNSW-NB15).
# Anything not listed (but present) maps to "other".
_STATE_TO_OUTCOME = {
    # Zeek
    "sf": "complete",
    "s0": "unanswered", "sh": "unanswered", "shr": "unanswered",
    "rej": "reset", "rsto": "reset", "rstr": "reset", "rstos0": "reset", "rstrh": "reset",
    "s1": "other", "s2": "other", "s3": "other", "oth": "other",
    # Argus
    "fin": "complete", "con": "complete", "acc": "complete", "clo": "complete",
    "rst": "reset", "req": "unanswered", "int": "other",
}


def clean_header(name: object) -> str:
    """Native column name without a byte-order mark or surrounding blanks.

    CICIDS2017 headers carry LEADING SPACES (" Flow Duration"); some re-uploads additionally start with a UTF-8 BOM, which
    reads as ``\ufeff`` (utf-8) or ``ï»¿`` (latin-1, what the CIC loaders use).  Either would hide the first column."""
    s = str(name)
    for bom in ("\ufeff", "\u00ef\u00bb\u00bf"):
        if s.startswith(bom):
            s = s[len(bom):]
    return s.strip()


def _norm(name: object) -> str:
    return clean_header(name).lower()


def candidate_columns(dataset_key: str) -> List[str]:
    """All lower-cased native column names this corpus may contribute (for ``usecols``)."""
    out: List[str] = []
    for names in SOURCE_COLUMNS[dataset_key].values():
        out.extend(names)
    return out


def resolve_columns(columns: Sequence[object], dataset_key: str) -> Dict[str, Optional[str]]:
    """Map each source field to the *actual* native column name (or None if absent)."""
    if dataset_key not in SOURCE_COLUMNS:
        raise KeyError(f"No feature mapping for dataset '{dataset_key}'")
    by_norm = {}
    for c in columns:
        by_norm.setdefault(_norm(c), c)          # first occurrence wins
    found: Dict[str, Optional[str]] = {}
    for field in SOURCE_FIELDS:
        found[field] = None
        for cand in SOURCE_COLUMNS[dataset_key].get(field, []):
            if cand in by_norm:
                found[field] = by_norm[cand]
                break
    return found


def normalize_protocol(series: pd.Series) -> pd.Series:
    """Return an object Series over PROTOCOLS from names ('tcp') or IANA numbers (6)."""
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series("other", index=series.index, dtype=object)
    out[s.isin(["tcp"]).to_numpy()] = "tcp"
    out[s.isin(["udp"]).to_numpy()] = "udp"
    out[s.isin(["icmp", "icmpv6", "ipv6-icmp"]).to_numpy()] = "icmp"
    num = pd.to_numeric(s, errors="coerce")
    is_num = num.notna().to_numpy()
    if is_num.any():
        mapped = num[is_num].astype(int).map(lambda v: _PROTO_NUMBERS.get(v, "other"))
        out[is_num] = mapped.to_numpy()
    out[s.isin(_MISSING_TOKENS).to_numpy()] = "unknown"
    return out


def service_group(dst_port: Optional[pd.Series], service: Optional[pd.Series], index: pd.Index) -> pd.Series:
    """Coarse L7 group from a service name and/or destination port (named service wins)."""
    out = pd.Series("unknown", index=index, dtype=object)
    if service is not None:
        s = service.astype(str).str.strip().str.lower()
        present = ~s.isin(_MISSING_TOKENS)
        out[present.to_numpy()] = "other"
        mapped = s.map(_SERVICE_NAMES)
        hit = mapped.notna().to_numpy()
        if hit.any():
            out[hit] = mapped[hit].to_numpy()
    if dst_port is not None:
        p = pd.to_numeric(dst_port, errors="coerce")
        has_port = p.notna().to_numpy()
        refine = has_port & out.isin(["unknown", "other"]).to_numpy()
        for group, ports in _PORT_GROUPS.items():
            out[refine & p.isin(list(ports)).to_numpy()] = group
        out[refine & out.eq("unknown").to_numpy()] = "other"
    return out


def port_range(dst_port: Optional[pd.Series], index: pd.Index) -> pd.Series:
    """IANA range of the destination port; a missing port or port 0 (ICMP) is 'unknown'."""
    out = pd.Series("unknown", index=index, dtype=object)
    if dst_port is None:
        return out
    p = pd.to_numeric(dst_port, errors="coerce").to_numpy(np.float64)
    ok = np.isfinite(p) & (p >= 1)
    out[ok & (p <= 1023)] = "wellknown"
    out[ok & (p >= 1024) & (p <= 49151)] = "registered"
    out[ok & (p >= 49152)] = "dynamic"
    return out


def conn_outcome_from_state(state: pd.Series) -> pd.Series:
    """Coarse connection outcome from a Zeek / Argus state code."""
    s = state.astype(str).str.strip().str.lower()
    out = s.map(_STATE_TO_OUTCOME).fillna("other").astype(object)
    out[s.isin(_MISSING_TOKENS).to_numpy()] = "unknown"
    return out


def conn_outcome_from_flags(
    index: pd.Index, fin: Optional[pd.Series], syn: Optional[pd.Series],
    rst: Optional[pd.Series], ack: Optional[pd.Series],
) -> pd.Series:
    """Coarse connection outcome from CICFlowMeter TCP flag counts (RST > FIN > SYN-without-ACK)."""
    if all(v is None for v in (fin, syn, rst, ack)):
        return pd.Series("unknown", index=index, dtype=object)

    def _num(v: Optional[pd.Series]) -> np.ndarray:
        if v is None:
            return np.zeros(len(index))
        return pd.to_numeric(v, errors="coerce").fillna(0.0).to_numpy(np.float64)

    f, s, r, a = _num(fin), _num(syn), _num(rst), _num(ack)
    out = pd.Series("other", index=index, dtype=object)
    out[(s > 0) & (a == 0)] = "unanswered"
    out[f > 0] = "complete"
    out[r > 0] = "reset"
    return out


def extract_canonical(df: pd.DataFrame, dataset_key: str) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """Return (canonical frame, found-columns map).  Unavailable fields are zero / unknown."""
    found = resolve_columns(df.columns, dataset_key)
    n = len(df)
    canon = pd.DataFrame(index=df.index)
    unit = DURATION_TO_SECONDS[dataset_key]
    for field in CANONICAL_NUMERIC:
        col = found[field]
        if col is None:
            canon[field] = 0.0
            continue
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vals = vals.clip(lower=0.0).astype(np.float64)
        canon[field] = vals * unit if field == "duration" else vals
    canon["protocol"] = normalize_protocol(df[found["protocol"]]) if found["protocol"] else "unknown"
    port_series = df[found["dst_port"]] if found["dst_port"] else None
    svc_series = df[found["service"]] if found["service"] else None
    canon["dst_port"] = (pd.to_numeric(port_series, errors="coerce") if port_series is not None
                         else pd.Series(np.nan, index=df.index))
    canon["service_group"] = service_group(port_series, svc_series, df.index)
    canon["port_range"] = port_range(port_series, df.index)
    if found["conn_state"]:
        canon["conn_outcome"] = conn_outcome_from_state(df[found["conn_state"]])
    else:
        flag = lambda k: df[found[k]] if found[k] else None          # noqa: E731
        canon["conn_outcome"] = conn_outcome_from_flags(
            df.index, flag("flag_fin"), flag("flag_syn"), flag("flag_rst"), flag("flag_ack"))
    assert len(canon) == n
    return canon, found


def _one_hot(values: np.ndarray, categories: Sequence[str], prefix: str, cols: Dict[str, np.ndarray]) -> None:
    for c in categories:
        cols[f"{prefix}_{c}"] = (values == c).astype(np.float64)


def build_features(canon: pd.DataFrame, feature_set: str = "core") -> Tuple[np.ndarray, List[str]]:
    """Expand the canonical frame into the fixed feature matrix (float32, un-scaled)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Available: {list(FEATURE_SETS)}")
    dur = canon["duration"].to_numpy(np.float64)          # seconds
    fb = canon["fwd_bytes"].to_numpy(np.float64)
    bb = canon["bwd_bytes"].to_numpy(np.float64)
    fp = canon["fwd_pkts"].to_numpy(np.float64)
    bp = canon["bwd_pkts"].to_numpy(np.float64)

    cols: Dict[str, np.ndarray] = {
        "log_duration_ms": np.log1p(dur * 1e3),
        "log_fwd_bytes": np.log1p(fb),
        "log_bwd_bytes": np.log1p(bb),
        "log_fwd_pkts": np.log1p(fp),
        "log_bwd_pkts": np.log1p(bp),
    }
    if feature_set in ("core22", "core"):
        d = np.maximum(dur, 1e-3)
        cols["log_fwd_pkt_size"] = np.log1p(fb / np.maximum(fp, 1.0))
        cols["log_bwd_pkt_size"] = np.log1p(bb / np.maximum(bp, 1.0))
        cols["byte_asym"] = (fb - bb) / (fb + bb + 1.0)
        cols["pkt_asym"] = (fp - bp) / (fp + bp + 1.0)
        cols["log_byte_rate"] = np.log1p((fb + bb) / d)
        cols["log_pkt_rate"] = np.log1p((fp + bp) / d)
        _one_hot(canon["protocol"].to_numpy(), PROTOCOLS, "proto", cols)
        _one_hot(canon["service_group"].to_numpy(), SERVICE_GROUPS, "svc", cols)
    if feature_set == "core":
        _one_hot(canon["port_range"].to_numpy(), PORT_RANGES, "port", cols)
        _one_hot(canon["conn_outcome"].to_numpy(), CONN_OUTCOMES, "conn", cols)
    names = FEATURE_SETS[feature_set]
    X = np.column_stack([cols[n] for n in names]).astype(np.float32)
    return X, list(names)
