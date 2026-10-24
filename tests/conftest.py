import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Insert src directory into python path for test execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


@pytest.fixture(scope="session", autouse=True)
def _leave_tracked_smoke_results_untouched():
    """The smoke and reproducibility tests write ``results/smoke_test/``, which is tracked in git.  Restore it after the session so
    that running the tests never leaves modified result files behind (they would otherwise end up in the next commit)."""
    smoke = Path(__file__).resolve().parents[1] / "results" / "smoke_test"
    backup = None
    if smoke.is_dir():
        backup = Path(tempfile.mkdtemp()) / "smoke_test"
        shutil.copytree(smoke, backup)
    yield
    shutil.rmtree(smoke, ignore_errors=True)
    if backup is not None:
        shutil.copytree(backup, smoke)
        shutil.rmtree(backup.parent, ignore_errors=True)
