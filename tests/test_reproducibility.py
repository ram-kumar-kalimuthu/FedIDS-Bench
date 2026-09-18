import os
import sys
import json
import subprocess
import pytest

def test_subprocess_end_to_end_reproducibility(tmp_path):
    """
    Test true end-to-end reproducibility across separate Python process invocations.
    This tests whether PyTorch CPU execution, NumPy RNG, and environment initialization
    produce bitwise/numeric identical results when invoked in completely independent processes.
    """
    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "run_smoke.py")
    )
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Process 1
    env1 = os.environ.copy()
    env1["PYTHONPATH"] = os.path.join(repo_root, "src")
    proc1 = subprocess.run(
        [sys.executable, script_path],
        cwd=repo_root,
        env=env1,
        capture_output=True,
        text=True
    )
    assert proc1.returncode == 0, f"Process 1 failed: {proc1.stderr}"

    # Read result 1 from smoke output directory
    res1_path = os.path.join(repo_root, "results", "smoke_test", "result.json")
    with open(res1_path, "r", encoding="utf-8") as f:
        data1 = json.load(f)

    # Move/Rename results folder so Process 2 writes fresh results
    res1_saved = os.path.join(tmp_path, "result1.json")
    with open(res1_saved, "w", encoding="utf-8") as f:
        json.dump(data1, f)

    # Process 2
    proc2 = subprocess.run(
        [sys.executable, script_path],
        cwd=repo_root,
        env=env1,
        capture_output=True,
        text=True
    )
    assert proc2.returncode == 0, f"Process 2 failed: {proc2.stderr}"

    with open(res1_path, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    # Compare key results across fresh subprocesses
    metrics1 = data1["result"]["metrics"]
    metrics2 = data2["result"]["metrics"]

    assert metrics1["accuracy"] == metrics2["accuracy"], f"Accuracy mismatch: {metrics1['accuracy']} vs {metrics2['accuracy']}"
    assert metrics1["f1"] == metrics2["f1"], f"F1 mismatch: {metrics1['f1']} vs {metrics2['f1']}"
    assert metrics1["multiclass"]["macro_f1"] == metrics2["multiclass"]["macro_f1"], "Macro F1 mismatch"
    assert data1["result"]["communication"] == data2["result"]["communication"], "Communication bytes mismatch"
