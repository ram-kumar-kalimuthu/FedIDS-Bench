"""
A rehearsal can never pass as a paper run.

* a cached result file is reused ONLY when it was produced with exactly the settings requested now,
* the report re-checks every run file against the canonical spec, whatever the last invocation's provenance says,
* --set / --limit runs are written to results/rehearsal/, never to results/runs/.
"""
import json
import os
import subprocess
import sys

import pytest
import yaml

from fedids_bench.experiments import grid
from fedids_bench.reporting import paper as P

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _job(n_samples=3000, data_dir="/a", seed=0):
    ov = {"dataset": {"name": "synthetic", "n_samples": n_samples, "data_dir": data_dir}, "seed": seed,
          "experiment_id": f"g__synthetic__a__s{seed}", "output_dir": "/tmp/ignored"}
    return {"run_id": f"g__synthetic__a__s{seed}", "grid": "g", "dataset": "synthetic", "variant": "a", "group": None,
            "seed": seed, "overrides": ov, "factors": {}}


@pytest.fixture()
def fake_runner(monkeypatch):
    calls = []

    def fake(cfg, save=True, **kw):
        calls.append(cfg.dataset.n_samples)
        return {"mode": "federated", "metrics": {"balanced_accuracy": 0.5}}

    import fedids_bench.experiments.runner as runner
    monkeypatch.setattr(runner, "run_experiment", fake)
    return calls


def test_cached_run_reused_only_with_identical_settings(tmp_path, fake_runner):
    out = str(tmp_path)
    assert grid.run_job(_job(3000), out)[1] == "ok"
    assert grid.run_job(_job(3000), out)[1] == "cached"                       # same settings -> reused
    assert grid.run_job(_job(3000, data_dir="/elsewhere"), out)[1] == "cached"  # where the data lives is not a setting
    assert grid.run_job(_job(50000), out)[1] == "ok"                          # rehearsal file must NOT be reused for a paper run
    assert fake_runner == [3000, 50000]
    stored = json.load(open(tmp_path / "g__synthetic__a__s0.json"))
    assert stored["overrides"]["dataset"]["n_samples"] == 50000               # and it was replaced
    assert grid.run_job(_job(3000), out, force=False)[1] == "ok"              # the opposite direction is refused as well


def test_corrupt_or_foreign_files_are_not_trusted(tmp_path, fake_runner):
    (tmp_path / "g__synthetic__a__s0.json").write_text("{not json")
    assert grid.run_job(_job(), str(tmp_path))[1] == "ok"
    (tmp_path / "g__synthetic__a__s0.json").write_text(json.dumps({"run_id": "x"}))      # no settings record
    assert grid.run_job(_job(), str(tmp_path))[1] == "ok"


def test_first_difference_reports_the_setting():
    a, b = {"dataset": {"n_samples": 10}, "federated": {"rounds": 3}}, {"dataset": {"n_samples": 10}, "federated": {"rounds": 20}}
    assert "federated.rounds" in grid.first_difference(a, b)
    assert grid.first_difference(a, a) is None
    assert "missing" in grid.first_difference({}, {"x": 1})


def _spec_yaml(tmp_path, seeds=(0, 1), rounds=20):
    spec = {"name": "g", "datasets": ["ton_iot"], "seeds": list(seeds), "base": {"federated": {"rounds": rounds}},
            "variants": [{"name": "a", "group": "fedavg", "overrides": {}}]}
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(spec))
    return str(p), spec


def _write(run_dir, spec_path, rounds, tuned, revision="abc123"):
    jobs = grid.expand_spec(grid.load_spec(spec_path), tuned)
    for j in jobs:
        ov = json.loads(json.dumps(j["overrides"]))
        ov["federated"]["rounds"] = rounds
        (run_dir / f"{j['run_id']}.json").write_text(json.dumps({"run_id": j["run_id"], "overrides": ov, "code_revision": revision,
                                                                   "result": {}}))
    return jobs


TUNED = {"ton_iot": {"fedavg": {"model": {"learning_rate": 0.1}, "federated": {"client_momentum": 0.9}}}}


def test_report_flags_rehearsal_runs_even_when_provenance_is_clean(tmp_path):
    spec_path, _ = _spec_yaml(tmp_path)
    tuned_path = tmp_path / "tuned.json"
    tuned_path.write_text(json.dumps(TUNED))
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "_provenance_g.json").write_text(json.dumps({"overrides_cli": {"dataset": {"data_dir": "/d"}}, "git_revision": "abc123"}))

    _write(runs, spec_path, rounds=20, tuned=TUNED)                              # exactly what the spec asks for
    st = P.grid_status("g", str(runs), spec_path, tuned_path=str(tuned_path))
    assert st.complete and st.paper_grade and not st.flags                       # --data-dir alone is not a rehearsal

    _write(runs, spec_path, rounds=4, tuned=TUNED)                               # leftovers of a 4-round rehearsal
    st = P.grid_status("g", str(runs), spec_path, tuned_path=str(tuned_path))
    assert not st.paper_grade and any("settings differ" in f and "federated.rounds" in f for f in st.flags)
    assert P.banner([st], False, False).startswith("NOT PAPER-GRADE")


def test_report_checks_tuned_values_and_extra_runs(tmp_path):
    spec_path, _ = _spec_yaml(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    _write(runs, spec_path, rounds=20, tuned={"ton_iot": {"fedavg": {"model": {"learning_rate": 0.5}, "federated": {"client_momentum": 0.9}}}})
    tuned_path = tmp_path / "tuned.json"
    tuned_path.write_text(json.dumps(TUNED))
    st = P.grid_status("g", str(runs), spec_path, tuned_path=str(tuned_path))
    assert any("learning_rate" in f for f in st.flags)

    st = P.grid_status("g", str(runs), spec_path, tuned_path=str(tmp_path / "missing.json"))       # no tuned file: say so, still check the rest
    assert any("tuned hyper-parameters not verified" in f for f in st.flags)

    _write(runs, spec_path, rounds=20, tuned=TUNED)
    (runs / "g__ton_iot__a__s7.json").write_text(json.dumps({"run_id": "g__ton_iot__a__s7", "overrides": {}, "result": {}}))
    st = P.grid_status("g", str(runs), spec_path, tuned_path=str(tuned_path))
    assert any("not declared by the spec" in f for f in st.flags)


def test_report_warns_about_mixed_or_dirty_revisions(tmp_path):
    spec_path, _ = _spec_yaml(tmp_path)
    tuned_path = tmp_path / "tuned.json"
    tuned_path.write_text(json.dumps(TUNED))
    runs = tmp_path / "runs"
    runs.mkdir()
    _write(runs, spec_path, rounds=20, tuned=TUNED, revision="aaaaaaaaaa+dirty")
    jobs = grid.expand_spec(grid.load_spec(spec_path), TUNED)
    p = runs / f"{jobs[0]['run_id']}.json"
    d = json.loads(p.read_text())
    d["code_revision"] = "bbbbbbbbbb"
    p.write_text(json.dumps(d))
    st = P.grid_status("g", str(runs), spec_path, tuned_path=str(tuned_path))
    assert st.paper_grade                                                            # warnings do not block, they are reported
    assert any("different code revisions" in w for w in st.warnings) and any("uncommitted" in w for w in st.warnings)


def _run_script(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_grid.py"), *args], capture_output=True, text=True,
                          cwd=ROOT, timeout=120)


def test_set_and_limit_runs_are_routed_to_the_rehearsal_folder():
    spec = os.path.join(ROOT, "configs", "grids", "main.yaml")
    r = _run_script("--spec", spec, "--dry-run", "--no-tuned", "--set", "federated.rounds=2")
    assert r.returncode == 0 and "REHEARSAL" in r.stdout and "results/rehearsal/main" in r.stdout
    r = _run_script("--spec", spec, "--dry-run", "--no-tuned", "--limit", "3")
    assert "results/rehearsal/main" in r.stdout
    r = _run_script("--spec", spec, "--dry-run", "--no-tuned", "--data-dir", "/somewhere", "--datasets", "ton_iot", "--seeds", "0,1")
    assert "REHEARSAL" not in r.stdout and "results/runs/main" in r.stdout          # location / subsets keep the paper folder
