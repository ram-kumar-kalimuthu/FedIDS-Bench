"""Imported first by every script so that `python scripts/<name>.py` works without `pip install -e .`."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
for _p in (SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
