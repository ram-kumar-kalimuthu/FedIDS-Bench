import os
from abc import ABC, abstractmethod
from typing import Dict, Type, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fedids_bench.data.schema import PreparedDataset
from fedids_bench.data.synthetic import generate_synthetic_ids
from fedids_bench.config import DatasetConfig

# Package data directory
DATA_DIR_BASE = os.path.dirname(os.path.abspath(__file__))


def _resolve_data_path(config: DatasetConfig, folder_name: str, file_names: List[str]) -> Optional[str]:
    """
    Search for dataset files across multiple candidate locations:
    1. config.data_dir
    2. config.data_dir / folder_name
    3. Package data directory: src/fedids_bench/data / folder_name
    4. Project root data directory: data / folder_name
    """
    candidate_dirs = [
        config.data_dir,
        os.path.join(config.data_dir, folder_name),
        os.path.join(DATA_DIR_BASE, folder_name),
        os.path.join(DATA_DIR_BASE, "..", "..", "data", folder_name),
        os.path.join(DATA_DIR_BASE, "..", "..", "..", "data", folder_name),
    ]
    for d in candidate_dirs:
        if os.path.exists(d):
            if os.path.isfile(d):
                return d
            for fn in file_names:
                target = os.path.join(d, fn)
                if os.path.exists(target):
                    return target
    return None


def _sample_df(df: pd.DataFrame, label_col: str, n_samples: Optional[int], seed: int) -> pd.DataFrame:
    """
    Sample exactly n_samples rows deterministically while preserving
    multi-class representation via proportional stratification.
    """
    if n_samples is None or n_samples <= 0 or n_samples >= len(df):
        return df

    class_counts = df[label_col].value_counts()
    n_classes = len(class_counts)
    if n_classes > 1 and n_samples >= n_classes:
        fractions = class_counts / len(df)
        samples_per_class = (fractions * n_samples).round().astype(int)
        samples_per_class = samples_per_class.clip(lower=1)
        
        sampled_dfs = []
        for cls_val, count in samples_per_class.items():
            cls_sub = df[df[label_col] == cls_val]
            take_n = min(count, len(cls_sub))
            sampled_dfs.append(cls_sub.sample(n=take_n, random_state=seed))
        sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        if len(sampled_df) > n_samples:
            sampled_df = sampled_df.sample(n=n_samples, random_state=seed)
        elif len(sampled_df) < n_samples:
            diff = n_samples - len(sampled_df)
            extra = df.sample(n=diff, replace=True, random_state=seed)
            sampled_df = pd.concat([sampled_df, extra], ignore_index=True)
        return sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        return df.sample(n=n_samples, random_state=seed).reset_index(drop=True)


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders in FedIDS-Bench."""

    @abstractmethod
    def load(self, config: DatasetConfig, seed: int = 42) -> PreparedDataset:
        pass

    def _prepare_and_scale(
        self,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        label_map: Dict[int, str],
        feature_names: list,
        test_fraction: float = 0.2,
        seed: int = 42,
        dataset_name: str = "dataset"
    ) -> PreparedDataset:
        """
        Split dataset into train (split=0) and test (split=-1),
        fit StandardScaler strictly on split=0 to prevent data leakage,
        and return PreparedDataset.
        """
        rng = np.random.default_rng(seed)
        n = len(X_raw)
        indices = rng.permutation(n)
        test_size = int(n * test_fraction)
        
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]

        split = np.zeros(n, dtype=np.int8)
        split[test_idx] = 2

        # Fit scaler ONLY on train set
        scaler = StandardScaler()
        X_scaled = X_raw.copy().astype(np.float32)
        X_scaled[train_idx] = scaler.fit_transform(X_raw[train_idx])
        X_scaled[test_idx] = scaler.transform(X_raw[test_idx])

        manifest = {
            "dataset_name": dataset_name,
            "n_samples": n,
            "n_features": X_raw.shape[1],
            "n_classes": len(label_map),
            "label_map": {str(k): v for k, v in label_map.items()},
            "feature_names": feature_names,
            "test_fraction": test_fraction,
            "seed": seed
        }

        return PreparedDataset(
            X=X_scaled,
            y=y_raw.astype(np.int64),
            attack_type=y_raw.astype(np.int64),
            split=split,
            manifest=manifest
        )


class SyntheticDatasetLoader(BaseDatasetLoader):
    def load(self, config: DatasetConfig, seed: int = 42) -> PreparedDataset:
        return generate_synthetic_ids(config, seed=seed)


class UnswNb15Loader(BaseDatasetLoader):
    """Loader for UNSW-NB15 intrusion detection dataset."""
    def load(self, config: DatasetConfig, seed: int = 42) -> PreparedDataset:
        data_path = _resolve_data_path(
            config,
            "UNSW_NB15",
            ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv", "unsw_nb15.csv"]
        )
        if not data_path:
            return generate_synthetic_ids(config, seed=seed)
        
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        label_col = 'attack_cat' if 'attack_cat' in df.columns else 'label'
        
        df = _sample_df(df, label_col, config.n_samples, seed)
        y_raw, uniques = pd.factorize(df[label_col])
        label_map = {i: str(val) for i, val in enumerate(uniques)}
        
        drop_cols = [c for c in [label_col, 'label', 'attack_cat', 'id'] if c in df.columns]
        X_df = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        X_raw = X_df.values
        feature_names = list(X_df.columns)

        return self._prepare_and_scale(X_raw, y_raw, label_map, feature_names, config.test_fraction, seed, "unsw_nb15")


class CicIds2017Loader(BaseDatasetLoader):
    """Loader for CICIDS2017 network intrusion dataset."""
    def load(self, config: DatasetConfig, seed: int = 42) -> PreparedDataset:
        data_path = _resolve_data_path(
            config,
            "Network Intrusion dataset(CIC-IDS- 2017)",
            ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "cicids2017.csv"]
        )
        if not data_path:
            return generate_synthetic_ids(config, seed=seed)
        
        # Read dataset (sample row limit for reading huge CSVs efficiently)
        nrows = max(100000, config.n_samples * 10) if config.n_samples and config.n_samples > 0 else None
        df = pd.read_csv(data_path, nrows=nrows)
        df.columns = df.columns.str.strip()
        label_col = [c for c in df.columns if 'label' in c.lower()][0]
        
        # Replace inf and fill na
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        df = _sample_df(df, label_col, config.n_samples, seed)
        
        y_raw, uniques = pd.factorize(df[label_col])
        label_map = {i: str(val) for i, val in enumerate(uniques)}
        
        X_df = df.drop(columns=[label_col], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        X_raw = X_df.values
        feature_names = list(X_df.columns)

        return self._prepare_and_scale(X_raw, y_raw, label_map, feature_names, config.test_fraction, seed, "cicids2017")


class CseCicIds2018Loader(BaseDatasetLoader):
    """Loader for CSE-CIC-IDS2018 intrusion dataset."""
    def load(self, config: DatasetConfig, seed: int = 42) -> PreparedDataset:
        data_path = _resolve_data_path(
            config,
            "IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)",
            ["02-21-2018.csv", "02-14-2018.csv", "cse_cic_ids2018.csv"]
        )
        if not data_path:
            return generate_synthetic_ids(config, seed=seed)
        
        nrows = max(50000, config.n_samples * 10) if config.n_samples and config.n_samples > 0 else None
        df = pd.read_csv(data_path, nrows=nrows, low_memory=False)
        df.columns = df.columns.str.strip()
        label_col = [c for c in df.columns if 'label' in c.lower()][0]
        
        # Filter out internal header repetitions common in 2018 files
        df = df[df[label_col].astype(str).str.lower() != 'label']
        df = _sample_df(df, label_col, config.n_samples, seed)
        
        y_raw, uniques = pd.factorize(df[label_col])
        label_map = {i: str(val) for i, val in enumerate(uniques)}
        
        drop_cols = [label_col, 'Timestamp']
        X_df = df.drop(columns=drop_cols, errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0)
        X_df = X_df.replace([np.inf, -np.inf], 0).fillna(0)
        X_raw = X_df.values
        feature_names = list(X_df.columns)

        return self._prepare_and_scale(X_raw, y_raw, label_map, feature_names, config.test_fraction, seed, "cse_cic_ids2018")


class Iot23Loader(BaseDatasetLoader):
    """Loader for IoT-23 intrusion detection dataset."""
    def load(self, config: DatasetConfig, seed: int = 42) -> PreparedDataset:
        data_path = _resolve_data_path(
            config,
            "IOT-23 Full Dataset",
            ["dataset5.csv", "dataset1.csv", "iot23.csv"]
        )
        if not data_path:
            return generate_synthetic_ids(config, seed=seed)
        
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        last_col = df.columns[-1]
        
        # Parse label from compound column if present
        def _extract_label(val):
            parts = str(val).strip().split()
            return parts[1] if len(parts) >= 2 else parts[0]
            
        df['label_clean'] = df[last_col].apply(_extract_label)
        label_col = 'label_clean'
        
        # Parse duration, orig_bytes, resp_bytes
        for col in ['duration', 'orig_bytes', 'resp_bytes']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace('-', np.nan), errors='coerce').fillna(0)
                
        df = _sample_df(df, label_col, config.n_samples, seed)
        y_raw, uniques = pd.factorize(df[label_col])
        label_map = {i: str(val) for i, val in enumerate(uniques)}
        
        drop_cols = [last_col, 'label_clean', 'uid', 'id.orig_h', 'id.resp_h', 'local_orig', 'local_resp']
        X_df = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        X_raw = X_df.values
        feature_names = list(X_df.columns)

        return self._prepare_and_scale(X_raw, y_raw, label_map, feature_names, config.test_fraction, seed, "iot23")


class TonIotLoader(BaseDatasetLoader):
    """Loader for ToN_IoT network intrusion dataset."""
    def load(self, config: DatasetConfig, seed: int = 42) -> PreparedDataset:
        data_path = _resolve_data_path(
            config,
            "TON_IoT Network Dataset",
            ["train_test_network.csv", "ton_iot.csv"]
        )
        if not data_path:
            return generate_synthetic_ids(config, seed=seed)
        
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        label_col = 'type' if 'type' in df.columns else 'label'
        
        df = _sample_df(df, label_col, config.n_samples, seed)
        y_raw, uniques = pd.factorize(df[label_col])
        label_map = {i: str(val) for i, val in enumerate(uniques)}
        
        drop_cols = [c for c in [label_col, 'label', 'type', 'src_ip', 'dst_ip', 'weird_name', 'weird_addl', 'weird_notice'] if c in df.columns]
        X_df = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        X_raw = X_df.values
        feature_names = list(X_df.columns)

        return self._prepare_and_scale(X_raw, y_raw, label_map, feature_names, config.test_fraction, seed, "ton_iot")


LOADER_REGISTRY: Dict[str, Type[BaseDatasetLoader]] = {
    "synthetic": SyntheticDatasetLoader,
    "unsw_nb15": UnswNb15Loader,
    "cicids2017": CicIds2017Loader,
    "cse_cic_ids2018": CseCicIds2018Loader,
    "iot23": Iot23Loader,
    "ton_iot": TonIotLoader
}


def get_dataset_loader(name: str) -> BaseDatasetLoader:
    name_clean = name.lower().replace("-", "_")
    if name_clean not in LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset loader: '{name}'. Available: {list(LOADER_REGISTRY.keys())}")
    return LOADER_REGISTRY[name_clean]()
