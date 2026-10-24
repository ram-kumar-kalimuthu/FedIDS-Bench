"""Row roles: hyper-parameter tuning must never touch a row that can appear in a reported test split."""
import numpy as np
import pandas as pd
import pytest

from fedids_bench.config import DatasetConfig
from fedids_bench.data.loaders import _role_mask, get_dataset_loader
from tests.test_features_loaders import WRITERS


def _pool(tmp_path, name, role, n_rows=6000, frac=0.2, seed=0):
    writer, fname = WRITERS[name]
    path = tmp_path / fname
    if not path.exists():
        writer(path, n=n_rows)
    cfg = DatasetConfig(name=name, n_samples=100_000, data_dir=str(tmp_path), row_role=role, tune_row_fraction=frac,
                        use_cache=False)
    return get_dataset_loader(name).read_canonical_pool(cfg, seed=seed)


@pytest.mark.parametrize("name", ["ton_iot", "cicids2017", "iot23"])
def test_tune_and_report_pools_are_disjoint_and_cover_everything(tmp_path, name):
    allp = _pool(tmp_path, name, "all")
    tune = _pool(tmp_path, name, "tune")
    rep = _pool(tmp_path, name, "report")
    def ident(p):                                              # a row's identity = all its canonical fields + label
        df = p["canon"].astype(str)
        return set(df.agg("|".join, axis=1) + "|" + p["labels"].astype(str))
    a, t, r = ident(allp), ident(tune), ident(rep)
    assert t.isdisjoint(r)
    assert t | r == a
    share = len(tune["canon"]) / len(allp["canon"])
    assert 0.14 < share < 0.26, share                          # about tune_row_fraction (=0.2) of the rows
    assert tune["row_role"] == "tune" and rep["row_role"] == "report" and allp["row_role"] == "all"


def test_role_depends_only_on_row_content():
    rng = np.random.default_rng(0)
    canon = pd.DataFrame({"duration": rng.random(4000), "fwd_bytes": rng.integers(0, 1000, 4000),
                          "protocol": rng.choice(["tcp", "udp"], 4000)})
    labels = pd.Series(rng.choice(["Normal", "DoS"], 4000))
    m1 = _role_mask(canon, labels, "tune", 0.2)
    perm = rng.permutation(len(canon))                         # same rows, different positions
    m2 = _role_mask(canon.iloc[perm].reset_index(drop=True), labels.iloc[perm].reset_index(drop=True), "tune", 0.2)
    assert (m1[perm] == m2).all()
    # complement property
    rep = _role_mask(canon, labels, "report", 0.2)
    assert (m1 ^ rep).all() and not (m1 & rep).any()


def test_exact_duplicates_never_straddle_roles():
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"duration": rng.random(500), "fwd_bytes": rng.integers(0, 1000, 500)})
    canon = pd.concat([base, base, base], ignore_index=True)     # every row appears three times
    labels = pd.Series(["x"] * len(canon))
    m = _role_mask(canon, labels, "tune", 0.3)
    assert (m[:500] == m[500:1000]).all() and (m[:500] == m[1000:]).all()
    # the same flow carrying a DIFFERENT label is a different row and may fall in either role
    labels2 = pd.Series(["x"] * 500 + ["y"] * 500 + ["z"] * 500)
    m2 = _role_mask(canon, labels2, "tune", 0.3)
    assert not ((m2[:500] == m2[500:1000]).all() and (m2[:500] == m2[1000:]).all())


def test_row_role_changes_the_cache_key(tmp_path):
    name = "ton_iot"
    writer, fname = WRITERS[name]
    writer(tmp_path / fname, n=3000)
    loader = get_dataset_loader(name)
    cfgs = {role: DatasetConfig(name=name, n_samples=100_000, data_dir=str(tmp_path), row_role=role) for role in ("all", "tune", "report")}
    sizes = {role: len(loader.read_canonical_pool(c, seed=0)["canon"]) for role, c in cfgs.items()}
    # reading again must hit the right cache entry, not the one of another role
    again = {role: len(loader.read_canonical_pool(c, seed=0)["canon"]) for role, c in cfgs.items()}
    assert sizes == again
    assert sizes["tune"] + sizes["report"] == sizes["all"]


def test_config_validation():
    with pytest.raises(Exception):
        DatasetConfig(row_role="tune", tune_row_fraction=0.7)
    assert DatasetConfig().row_role == "all"
