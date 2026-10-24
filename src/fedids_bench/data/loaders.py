import glob
import hashlib
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fedids_bench.config import DatasetConfig
from fedids_bench.data.discovery import MAX_DEPTH, Found, find_files, hint_messages, search_roots
from fedids_bench.utils.reproducibility import derive_seed
from fedids_bench.data.features import (
    CANONICAL_FIELDS,
    CANONICAL_NUMERIC,
    CONN_SOURCE_FIELDS,
    FEATURE_SETS,
    SOURCE_FIELDS,
    build_features,
    candidate_columns,
    clean_header,
    extract_canonical,
    resolve_columns,
)
from fedids_bench.data.schema import PreparedDataset
from fedids_bench.data.synthetic import generate_synthetic_ids

logger = logging.getLogger("fedids_bench.data")

# Package data directory
DATA_DIR_BASE = os.path.dirname(os.path.abspath(__file__))

# Kept for backwards compatibility with older scripts (the five un-transformed volume fields).
COMMON_FEATURES = list(CANONICAL_NUMERIC)

# Bump when canonical extraction changes: it invalidates every cached pool.
LOADER_VERSION = "5"

# A missing field in this list makes a corpus nearly useless -> warn loudly.
_CORE_FIELDS = ("duration", "fwd_bytes", "bwd_bytes", "fwd_pkts", "bwd_pkts")


def _norm(name: object) -> str:
    return clean_header(name).lower()


_WARNED: set = set()


def _warn_once(key: tuple, msg: str, *args: Any) -> None:
    """Log a warning once per process (every grid job re-enters the loader)."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(msg, *args)


def _count_rows(path: str, header_lines: int = 1) -> int:
    """Number of data rows (newline count minus the header lines); counted in 16 MB blocks, which is fast even for 10 GB."""
    n = 0
    last = b"\n"
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 24)
            if not block:
                break
            n += block.count(b"\n")
            last = block[-1:]
    if last != b"\n":
        n += 1                                   # final line without a newline
    return max(n - header_lines, 0)


def zeek_fields(path: str) -> Optional[List[str]]:
    """Field names from the ``#fields`` line of a Zeek TSV log (``conn.log.labeled``), or None if the file is not one."""
    try:
        with open(path, "rb") as fh:
            for i, raw in enumerate(fh):
                if i > 80:
                    break
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith("#fields"):
                    return line.split("\t")[1:]
                if not line.startswith("#"):
                    break
    except OSError:
        return None
    return None


def _zeek_header_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as fh:
        for raw in fh:
            if raw.startswith(b"#"):
                n += 1
            else:
                break
    return n + 1                                 # the trailing "#close" line


def _role_mask(canon: pd.DataFrame, labels: pd.Series, role: str, fraction: float) -> np.ndarray:
    """Boolean keep-mask for a row role.  The role of a row depends ONLY on its content (canonical fields + native
    label), never on its position, the sampling seed or the file layout: tuning rows and reporting rows are disjoint
    by construction and identical flows land in the same role."""
    if role == "all":
        return np.ones(len(canon), dtype=bool)
    key = canon.copy()
    key["_label"] = labels.to_numpy()
    h = pd.util.hash_pandas_object(key, index=False).to_numpy(dtype=np.uint64)
    u = (h >> np.uint64(11)).astype(np.float64) / float(2 ** 53)
    is_tune = u < float(fraction)
    return is_tune if role == "tune" else ~is_tune


def _read_file_pool(
    path: str,
    usecols: List[str],
    frac: float,
    seed: int,
    read_kwargs: Optional[Dict[str, Any]] = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Read ONLY the needed columns and keep a uniform random fraction ``frac`` of the rows of the WHOLE
    file (chunk-wise Bernoulli sampling).  Reading just the first N rows of a time-ordered capture, as
    the previous loaders did, yields a non-representative slice (e.g. 96% attacks for the head of
    CSE-CIC-IDS2018's 02-21 file).
    """
    kwargs = dict(read_kwargs or {})
    parts: List[pd.DataFrame] = []
    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False, **kwargs)):
        parts.append(chunk if frac >= 1.0 else chunk.sample(frac=frac, random_state=derive_seed(seed, i)))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
    df.columns = [clean_header(c) for c in df.columns]
    return df


def _stratified_indices(labels: np.ndarray, n_samples: Optional[int], seed: int) -> np.ndarray:
    """Pick exactly n_samples distinct rows, proportional per class (>=1 per class), no replacement."""
    labels = np.asarray(labels)
    n = len(labels)
    if n_samples is None or n_samples <= 0 or n_samples >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    quota = np.minimum(counts, np.maximum(1, np.round(counts / n * n_samples).astype(int)))
    while quota.sum() > n_samples:
        quota[int(np.argmax(quota))] -= 1
    while quota.sum() < n_samples:
        room = counts - quota
        i = int(np.argmax(room))
        if room[i] <= 0:
            break
        quota[i] += 1
    picks = [rng.choice(np.flatnonzero(labels == c), size=int(q), replace=False) for c, q in zip(classes, quota)]
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


def _stratified_split(ix: np.ndarray, test_size: float, strat: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        a, b = train_test_split(ix, test_size=test_size, stratify=strat, random_state=seed)
    except ValueError:  # a class with <2 members cannot be stratified
        a, b = train_test_split(ix, test_size=test_size, random_state=seed)
    return np.sort(a), np.sort(b)


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders in FedIDS-Bench."""

    @abstractmethod
    def load(self, config: DatasetConfig, seed: int = 42) -> PreparedDataset:
        pass


class SyntheticDatasetLoader(BaseDatasetLoader):
    def load(self, config: DatasetConfig, seed: int = 42) -> PreparedDataset:
        return generate_synthetic_ids(config, seed=seed)


class FileDatasetLoader(BaseDatasetLoader):
    """
    Shared pipeline for the five real corpora:

      read pool (uniform over the WHOLE file(s), needed columns only; cached)
        -> native labels -> binary label + attack-category code
        -> canonical flow fields -> (optional) exact-duplicate removal
        -> proportional stratified sampling to n_samples
        -> harmonized feature matrix (``core`` / ``core22`` / ``core5``)
        -> STRATIFIED train / val / test split
        -> StandardScaler fitted on the TRAIN split only.

    ``config.pool == "first"`` (default, legacy) reads the first matching file only;
    ``"all"`` pools every file of the corpus folder that matches ``pool_patterns`` (found case-insensitively, in nested folders
    too -- see ``discovery.py``; a missing file is reported against ``expected_files``).
    """

    dataset_key: str = ""
    folder: str = ""
    file_names: List[str] = []
    pool_patterns: List[str] = []
    tree_patterns: List[str] = []                 # one file per capture in many sub-folders (IoT-23's raw Zeek logs)
    expected_files: Optional[int] = None          # files in the published corpus (None: varies) -> completeness warning
    hints: Dict[str, str] = {}                    # known-but-unsupported layouts: file pattern -> what to do instead
    benign_tokens = {"benign", "normal", "0"}
    read_kwargs: Dict[str, Any] = {}

    # ---- dataset specific hooks -------------------------------------------------------------
    def _label_column(self, columns: List[str]) -> str:
        """Return the name (from `columns`) holding the native label."""
        for c in columns:
            if "label" in _norm(c):
                return c
        raise ValueError(f"{self.__class__.__name__}: no label column among {columns[:8]}...")

    def _label_columns(self, columns: List[str]) -> List[str]:
        """Every native column needed to build the label (usually one)."""
        return [self._label_column(columns)]

    def _native_labels(self, df: pd.DataFrame) -> pd.Series:
        col = self._label_column(list(df.columns))
        return df[col].astype(str).str.strip()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _file_kwargs(self, path: str) -> Dict[str, Any]:
        """``pandas.read_csv`` options for this file (encoding, separator, header names)."""
        return dict(self.read_kwargs)

    def _header_lines(self, path: str) -> int:
        """Lines before the first data row (one CSV header line; Zeek logs have several ``#`` lines)."""
        return 1

    # ---- file access ------------------------------------------------------------------------
    def _roots(self, config: DatasetConfig) -> List[str]:
        extra = [os.path.join(DATA_DIR_BASE, self.folder),
                 os.path.join(DATA_DIR_BASE, "..", "..", "data", self.folder),
                 os.path.join(DATA_DIR_BASE, "..", "..", "..", "data", self.folder)]
        return search_roots(config.data_dir, self.folder, extra)

    def _discover(self, config: DatasetConfig) -> Found:
        """Find the corpus (case-insensitive, nested folders allowed); remembers the answer for this loader instance."""
        key = (os.path.abspath(config.data_dir), config.pool)
        memo = self.__dict__.setdefault("_found_memo", {})
        if key in memo:
            return memo[key]
        roots = self._roots(config)
        found = find_files(roots, self.pool_patterns, self.file_names, self.tree_patterns, pool_all=(config.pool == "all"))
        if found is None:
            wanted = list(self.tree_patterns) + list(self.pool_patterns) + list(self.file_names)
            msg = (f"Data for {self.__class__.__name__} not found. Looked (case-insensitively, up to {MAX_DEPTH} folder levels "
                   f"deep) in {roots or [config.data_dir]} for {wanted}. Check --data-dir and docs/DATA.md.")
            raise FileNotFoundError(" ".join([msg] + hint_messages(roots, self.hints)))
        if found.alternatives:
            _warn_once(("alt", self.dataset_key, found.directory),
                       "%s: using the %d file(s) in %s; other folders also matched and were IGNORED (not pooled): %s",
                       self.dataset_key, len(found.files), found.directory, found.alternatives)
        memo[key] = found
        return found

    def locate(self, config: DatasetConfig) -> str:
        """The first file of the corpus."""
        return self._discover(config).files[0]

    def locate_all(self, config: DatasetConfig) -> List[str]:
        """The file(s) that make up this corpus for ``config.pool``."""
        found = self._discover(config)
        files = list(found.files) if config.pool == "all" else list(found.files[:1])
        if config.pool == "all" and self.expected_files and len(files) != self.expected_files:
            _warn_once(("count", self.dataset_key, found.directory),
                       "%s: found %d file(s) in %s but the published corpus has %d -- the benchmark runs on a partial or "
                       "different corpus (see docs/DATA.md); the files used are recorded in every run's manifest",
                       self.dataset_key, len(files), found.directory, self.expected_files)
        return files

    def _header(self, path: str) -> List[str]:
        return list(pd.read_csv(path, nrows=0, **self._file_kwargs(path)).columns)

    def _usecols(self, header: List[str]) -> Tuple[List[str], List[str]]:
        want = set(candidate_columns(self.dataset_key))
        labs = self._label_columns(header)
        return [c for c in header if _norm(c) in want or c in labs], labs

    def _read_one(
        self, path: str, frac: float, seed: int, role: str = "all", role_fraction: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Optional[str]]]:
        header = self._header(path)
        usecols, _ = self._usecols(header)
        df = _read_file_pool(path, usecols, frac, seed, self._file_kwargs(path))
        df = self._clean(df)
        labels = self._native_labels(df)
        canon, found = extract_canonical(df, self.dataset_key)
        canon, labels = canon.reset_index(drop=True), labels.reset_index(drop=True)
        if role != "all":
            keep = _role_mask(canon, labels, role, role_fraction)
            canon, labels = canon[keep].reset_index(drop=True), labels[keep].reset_index(drop=True)
        return canon, labels, found

    # ---- pool + cache -----------------------------------------------------------------------
    def _cache_file(self, config: DatasetConfig, files: List[str], pool_total: Optional[int], seed: int) -> Optional[str]:
        if not config.use_cache:
            return None
        meta = [(os.path.basename(f), os.path.getsize(f), os.stat(f).st_mtime_ns) for f in files]
        ident: tuple = (LOADER_VERSION, self.dataset_key, meta, pool_total, int(seed))
        if config.row_role != "all":                 # keeps every cache written before row roles existed valid
            ident += (config.row_role, round(float(config.tune_row_fraction), 6))
        key = hashlib.sha1(repr(ident).encode()).hexdigest()[:20]
        cache_dir = config.cache_dir or os.path.join(config.data_dir, ".fedids_cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            if not os.access(cache_dir, os.W_OK):
                raise PermissionError(cache_dir)
        except OSError:
            cache_dir = os.path.join(tempfile.gettempdir(), "fedids_cache")
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                return None
        return os.path.join(cache_dir, f"{self.dataset_key}_{key}.pkl")

    def read_canonical_pool(self, config: DatasetConfig, seed: int) -> Dict[str, Any]:
        """Uniform pool of the corpus as a canonical-field frame + native label strings (+ provenance)."""
        files = self.locate_all(config)
        pool_total = max(10 * config.n_samples, 250_000) if config.n_samples and config.n_samples > 0 else None
        cache = self._cache_file(config, files, pool_total, seed)
        if cache and os.path.exists(cache):
            try:
                return pd.read_pickle(cache)
            except Exception:                                    # corrupt / partial cache -> rebuild
                logger.warning("ignoring unreadable cache %s", cache)

        counts = [_count_rows(f, self._header_lines(f)) for f in files]
        total = int(sum(counts))
        share = {"all": 1.0, "tune": float(config.tune_row_fraction),
                 "report": 1.0 - float(config.tune_row_fraction)}[config.row_role]
        frac = 1.0 if (pool_total is None or total * share <= pool_total) else pool_total / (total * share)
        canon_parts: List[pd.DataFrame] = []
        label_parts: List[pd.Series] = []
        found_union: Dict[str, Optional[str]] = {k: None for k in SOURCE_FIELDS}
        missing_in: Dict[str, List[str]] = {}
        for i, f in enumerate(files):
            canon, labels, found = self._read_one(f, frac, derive_seed(seed, 7919, i), config.row_role, config.tune_row_fraction)
            canon_parts.append(canon)
            label_parts.append(labels)
            for k, v in found.items():
                if v is not None and found_union.get(k) is None:
                    found_union[k] = v
            for k, v in found.items():
                if v is None:
                    missing_in.setdefault(k, []).append(os.path.basename(f))
        pool = {
            "canon": pd.concat(canon_parts, ignore_index=True),
            "labels": pd.concat(label_parts, ignore_index=True),
            "found": found_union,
            "missing_in_files": {k: v for k, v in missing_in.items() if len(v) < len(files)},   # partially missing
            "files": [os.path.basename(f) for f in files],
            "directory": self._discover(config).directory,
            "files_expected": self.expected_files,
            "rows_per_file": counts,
            "pool_fraction": float(frac),
            "row_role": config.row_role,
            "loader_version": LOADER_VERSION,
        }
        if cache:
            tmp = f"{cache}.{os.getpid()}.tmp"
            try:
                pd.to_pickle(pool, tmp)
                os.replace(tmp, cache)
            except OSError:
                logger.warning("could not write cache %s", cache)
        return pool

    # kept for older callers: raw (non-canonical) pool of the first file
    def read_pool(self, config: DatasetConfig, seed: int) -> Tuple[pd.DataFrame, pd.Series, str]:
        path = self.locate(config)
        header = self._header(path)
        usecols, _ = self._usecols(header)
        pool = max(10 * config.n_samples, 250_000) if config.n_samples and config.n_samples > 0 else None
        total = _count_rows(path, self._header_lines(path))
        frac = 1.0 if (pool is None or total <= pool) else pool / float(total)
        df = self._clean(_read_file_pool(path, usecols, frac, seed, self._file_kwargs(path)))
        return df, self._native_labels(df), path

    # ---- description (dataset table) ----------------------------------------------------------
    def describe(self, config: DatasetConfig, with_duplicates: bool = True) -> Dict[str, Any]:
        """Facts about the native file(s); nothing here is hand-typed."""
        files = self.locate_all(config)
        found_files = self._discover(config)
        counts: Dict[str, int] = {}
        rows_per_file: Dict[str, int] = {}
        found_union: Dict[str, Optional[str]] = {k: None for k in SOURCE_FIELDS}
        missing_in: Dict[str, List[str]] = {}
        n_cols = None
        for f in files:
            header = self._header(f)
            n_cols = len(header) if n_cols is None else n_cols
            found = resolve_columns(header, self.dataset_key)
            for k, v in found.items():
                if v is not None and found_union.get(k) is None:
                    found_union[k] = v
                if v is None:
                    missing_in.setdefault(k, []).append(os.path.basename(f))
            n_file = 0
            for chunk in pd.read_csv(f, usecols=self._label_columns(header), chunksize=500_000, low_memory=False,
                                     **self._file_kwargs(f)):
                chunk.columns = [clean_header(c) for c in chunk.columns]
                lab = self._native_labels(self._clean(chunk))
                n_file += int(len(lab))
                for k, v in lab.value_counts().items():
                    counts[k] = counts.get(k, 0) + int(v)
            rows_per_file[os.path.basename(f)] = n_file
        n_rows = int(sum(counts.values()))
        benign = int(sum(v for k, v in counts.items() if k.strip().lower() in self.benign_tokens))
        top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:20])
        out: Dict[str, Any] = {
            "dataset": self.dataset_key,
            "files": [os.path.basename(f) for f in files],
            "n_files": len(files),
            "rows_per_file": rows_per_file,
            "n_rows": n_rows,
            "n_native_columns": n_cols,
            "native_label_classes": len(counts),
            "label_counts_top20": top,
            "attack_share": round(1.0 - benign / max(n_rows, 1), 4),
            "fields_found": dict(found_union),
            "fields_missing": [k for k, v in found_union.items() if v is None],
            "fields_missing_in_some_files": {k: v for k, v in missing_in.items() if len(v) < len(files)},
            "data_directory": found_files.directory,
            "files_expected": self.expected_files,
            "file_count_ok": bool(self.expected_files is None or config.pool != "all" or len(files) == self.expected_files),
            "ignored_directories": [[d, n] for d, n in found_files.alternatives],
        }
        if with_duplicates:
            # duplicate rate of the CANONICAL (harmonized) fields on a ~250k-row uniform pool
            pool = self.read_canonical_pool(config.model_copy(update={"n_samples": 25_000}), seed=0)
            canon, labels = pool["canon"], pool["labels"]
            key = canon.copy()
            key["_label"] = labels.to_numpy()
            out["canonical_duplicate_rate"] = round(float(key.duplicated(keep="first").mean()), 4) if len(key) else None
            out["duplicate_pool_rows"] = int(len(key))
        return out

    # ---- main entry -------------------------------------------------------------------------
    def load(self, config: DatasetConfig, seed: int = 42) -> PreparedDataset:
        pool = self.read_canonical_pool(config, seed)
        return self._finalise(pool, config, seed)

    def _finalise(self, pool: Dict[str, Any], config: DatasetConfig, seed: int) -> PreparedDataset:
        canon = pool["canon"].reset_index(drop=True)
        lab = pool["labels"].astype(str).str.strip().reset_index(drop=True)
        found = pool["found"]

        missing_core = [f for f in _CORE_FIELDS if found.get(f) is None]
        if missing_core:
            logger.warning("%s: core fields %s were not found in %s -- those features are constant zeros",
                           self.dataset_key, missing_core, pool.get("files"))

        is_benign = lab.str.lower().isin(self.benign_tokens).to_numpy()
        cats = lab.where(~is_benign, "__benign__")
        order = ["__benign__"] + list(cats[~is_benign].value_counts().index)
        code = {c: i for i, c in enumerate(order)}
        attack_type = cats.map(code).to_numpy(np.int64)
        y = (~is_benign).astype(np.int64)

        n_dup = 0
        if config.deduplicate:
            key = canon.copy()
            key["_label"] = attack_type
            dup = key.duplicated(keep="first").to_numpy()
            n_dup = int(dup.sum())
            keep = ~dup
            canon, y, attack_type = canon[keep].reset_index(drop=True), y[keep], attack_type[keep]

        idx = _stratified_indices(attack_type, config.n_samples, seed)
        canon = canon.iloc[idx].reset_index(drop=True)
        y, attack_type = y[idx], attack_type[idx]

        X_feat, feature_names = build_features(canon, config.feature_set)
        label_map = {0: "Benign", 1: "Attack"}
        names = {str(i): (c if c != "__benign__" else "Benign") for c, i in code.items()}
        type_counts = {names[str(i)]: int((attack_type == i).sum()) for i in range(len(order))}
        return self._split_and_scale(
            X_feat, y, attack_type, label_map, feature_names, config, seed,
            extra={
                "source_file": ",".join(pool.get("files", [])),
                "source_files": list(pool.get("files", [])),
                "source_dir": pool.get("directory"),
                "files_expected": pool.get("files_expected"),
                "pool_mode": config.pool,
                "pool_rows": int(len(pool["canon"])),
                "canonical_fields_found": {k: found.get(k) for k in CANONICAL_FIELDS},
                "connection_source_fields": {k: found.get(k) for k in CONN_SOURCE_FIELDS},
                "fields_missing_in_some_files": pool.get("missing_in_files", {}),
                "n_duplicates_dropped": n_dup,
                "attack_type_map": names,
                "attack_type_counts": type_counts,
            },
        )

    def _split_and_scale(
        self,
        X_feat: np.ndarray,
        y: np.ndarray,
        attack_type: np.ndarray,
        label_map: Dict[int, str],
        feature_names: List[str],
        config: DatasetConfig,
        seed: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> PreparedDataset:
        n = len(y)
        idx = np.arange(n)
        trainval, test_idx = _stratified_split(idx, config.test_fraction, y, seed)
        if config.val_fraction > 0:
            rel = config.val_fraction / (1.0 - config.test_fraction)
            train_idx, val_idx = _stratified_split(trainval, rel, y[trainval], seed)
        else:
            train_idx, val_idx = trainval, np.array([], dtype=np.int64)

        split = np.zeros(n, dtype=np.int8)
        split[val_idx] = 1
        split[test_idx] = 2

        # Scaler statistics come from the TRAIN split only (zero-leakage property).
        scaler = StandardScaler().fit(X_feat[train_idx])
        X = scaler.transform(X_feat).astype(np.float32)

        def _prev(ix: np.ndarray) -> Optional[float]:
            return round(float(y[ix].mean()), 4) if len(ix) else None

        manifest: Dict[str, Any] = {
            "dataset_name": self.dataset_key,
            "n_samples": int(n),
            "n_features": int(X_feat.shape[1]),
            "n_classes": len(label_map),
            "label_map": {str(k): v for k, v in label_map.items()},
            "feature_set": config.feature_set,
            "feature_names": list(feature_names),
            "test_fraction": config.test_fraction,
            "val_fraction": config.val_fraction,
            "seed": seed,
            "split_counts": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "attack_prevalence": {"train": _prev(train_idx), "val": _prev(val_idx), "test": _prev(test_idx)},
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
        }
        manifest.update(extra or {})
        return PreparedDataset(
            X=X, y=y.astype(np.int64), attack_type=attack_type.astype(np.int64),
            split=split, manifest=manifest, X_feat=X_feat.astype(np.float32),
        )


class UnswNb15Loader(FileDatasetLoader):
    """UNSW-NB15 (training/testing-set CSVs: no ports; service name is used for the L7 group)."""
    dataset_key = "unsw_nb15"
    folder = "UNSW_NB15"
    file_names = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv", "unsw_nb15.csv"]
    pool_patterns = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
    expected_files = 2
    hints = {"UNSW-NB15_[1-4].csv": "Found the raw, header-less UNSW-NB15_1..4.csv files; this benchmark reads the PARTITIONED "
                                    "UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv (they carry a header and attack_cat)."}
    benign_tokens = {"normal", "0", "benign"}

    def _label_column(self, columns: List[str]) -> str:
        for c in columns:
            if _norm(c) == "attack_cat":
                return c
        for c in columns:
            if _norm(c) == "label":
                return c
        raise ValueError("UNSW-NB15: neither 'attack_cat' nor 'label' column found")

    def _native_labels(self, df: pd.DataFrame) -> pd.Series:
        col = self._label_column(list(df.columns))
        return df[col].fillna("Normal").astype(str).str.strip()


class CicIds2017Loader(FileDatasetLoader):
    """CICIDS2017 flow CSVs ('MachineLearningCVE' or 'TrafficLabelling'; headers carry leading spaces)."""
    dataset_key = "cicids2017"
    folder = "Network Intrusion dataset(CIC-IDS- 2017)"
    file_names = ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "cicids2017.csv"]
    pool_patterns = ["*.pcap_ISCX.csv"]
    expected_files = 8
    benign_tokens = {"benign", "normal", "0"}
    read_kwargs = {"encoding": "latin-1"}          # the web-attack file contains cp1252 bytes (en dash)


class CseCicIds2018Loader(FileDatasetLoader):
    """CSE-CIC-IDS2018 processed CSVs (some files repeat the header row inside the data)."""
    dataset_key = "cse_cic_ids2018"
    folder = "IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)"
    file_names = ["02-21-2018.csv", "02-14-2018.csv", "cse_cic_ids2018.csv"]
    pool_patterns = ["??-??-2018.csv", "*TrafficForML_CICFlowMeter.csv"]
    expected_files = 10
    benign_tokens = {"benign", "normal", "0"}
    read_kwargs = {"encoding": "latin-1"}

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self._label_column(list(df.columns))
        return df[df[col].astype(str).str.strip().str.lower() != "label"].reset_index(drop=True)


class Iot23Loader(FileDatasetLoader):
    """
    IoT-23 in either common form:

    * the raw Zeek ``conn.log.labeled`` files (one per capture, tab-separated, ``#fields`` header, the last field being the
      compound ``tunnel_parents   label   detailed-label``), found anywhere below the data directory;
    * CSV exports named ``dataset*.csv`` with that compound last column or with separate ``label`` / ``detailed-label`` columns.

    A malicious flow is labeled with its detailed category (e.g. PartOfAHorizontalPortScan, C&C, DDoS).
    """
    dataset_key = "iot23"
    folder = "IOT-23 Full Dataset"
    file_names = ["dataset5.csv", "dataset1.csv", "iot23.csv"]
    pool_patterns = ["dataset*.csv"]
    tree_patterns = ["conn.log.labeled"]
    benign_tokens = {"benign", "normal", "0", "-"}
    _DETAILED = ("detailed-label", "detailed_label", "detailed label")

    def _file_kwargs(self, path: str) -> Dict[str, Any]:
        names = zeek_fields(path)
        if names is None:                                          # a CSV export
            return dict(self.read_kwargs)
        return dict(sep="\t", comment="#", header=None, names=names, quoting=3, encoding="utf-8",
                    encoding_errors="replace", on_bad_lines="skip")

    def _header_lines(self, path: str) -> int:
        return _zeek_header_lines(path) if zeek_fields(path) is not None else 1

    def _separate_label_columns(self, columns: List[str]) -> Optional[Tuple[str, Optional[str]]]:
        by = {_norm(c): c for c in columns}
        if "label" not in by:
            return None
        return by["label"], next((by[k] for k in self._DETAILED if k in by), None)

    def _label_column(self, columns: List[str]) -> str:
        sep = self._separate_label_columns(columns)
        return sep[0] if sep else columns[-1]                       # compound 'tunnel_parents label detailed-label'

    def _label_columns(self, columns: List[str]) -> List[str]:
        sep = self._separate_label_columns(columns)
        if sep:
            return [sep[0]] + ([sep[1]] if sep[1] else [])
        return [columns[-1]]

    def _native_labels(self, df: pd.DataFrame) -> pd.Series:
        sep = self._separate_label_columns(list(df.columns))
        if sep:                                                     # separate label / detailed-label columns
            coarse = df[sep[0]].astype(str).str.strip()
            benign = coarse.str.lower().isin({"benign", "normal", "0"})
            if sep[1] is None:
                return coarse
            detail = df[sep[1]].astype(str).str.strip()
            cat = detail.where(~detail.str.lower().isin({"-", "", "nan", "(empty)"}), coarse)
            return cat.where(~benign, "Benign")

        def _extract(val: object) -> str:                           # compound column
            parts = str(val).strip().split()
            if len(parts) >= 3 and parts[1].lower() != "benign" and parts[2] != "-":
                return parts[2]                                     # detailed malicious category
            return parts[1] if len(parts) >= 2 else (parts[0] if parts else "-")

        return df[self._label_column(list(df.columns))].map(_extract)


class TonIotLoader(FileDatasetLoader):
    """ToN_IoT *Network* dataset (train_test_network.csv)."""
    dataset_key = "ton_iot"
    folder = "TON_IoT Network Dataset"
    file_names = ["train_test_network.csv", "ton_iot.csv"]
    pool_patterns = ["train_test_network.csv"]
    expected_files = 1
    hints = {"Network_dataset_*.csv": "Found the full 'Network_dataset_N.csv' captures; this benchmark uses the 461k-row "
                                      "train_test_network.csv from the Train_Test_datasets folder."}
    benign_tokens = {"normal", "benign", "0"}

    def _label_column(self, columns: List[str]) -> str:
        for c in columns:
            if _norm(c) == "type":
                return c
        for c in columns:
            if _norm(c) == "label":
                return c
        raise ValueError("ToN-IoT: neither 'type' nor 'label' column found")


LOADER_REGISTRY: Dict[str, Type[BaseDatasetLoader]] = {
    "synthetic": SyntheticDatasetLoader,
    "unsw_nb15": UnswNb15Loader,
    "cicids2017": CicIds2017Loader,
    "cse_cic_ids2018": CseCicIds2018Loader,
    "iot23": Iot23Loader,
    "ton_iot": TonIotLoader,
}


def get_dataset_loader(name: str) -> BaseDatasetLoader:
    name_clean = name.lower().replace("-", "_")
    if name_clean not in LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset loader: '{name}'. Available: {list(LOADER_REGISTRY.keys())}")
    return LOADER_REGISTRY[name_clean]()
