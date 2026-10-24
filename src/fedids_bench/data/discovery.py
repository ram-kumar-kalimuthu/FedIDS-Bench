"""
Find a corpus on disk.

Real downloads are inconsistent: the same CICIDS2017 day ships as ``Wednesday-workingHours.pcap_ISCX.csv`` (lower-case "w"),
the ML-ready CSVs sit in ``MachineLearningCVE/`` or ``TrafficLabelling/``, zip extraction adds an extra folder level, and
re-uploads rename everything.  An exact-name lookup then silently misses files -- for instance a whole attack day -- and the
benchmark would train on a different corpus than the one it reports.  Everything here is therefore matched
case-insensitively, at any depth up to ``MAX_DEPTH`` below the data directory, and the caller is told what was found *and*
what was expected (see ``FileDatasetLoader.expected_files``).

Two layouts exist:

* flat corpora (CIC*, UNSW-NB15, ToN_IoT): the files of one corpus live in ONE directory.  If several directories match
  (e.g. ``MachineLearningCVE/`` *and* ``TrafficLabelling/``) exactly one is used -- the one with the most matching files --
  and the others are reported as ``alternatives``, never pooled (that would double-count every flow).
* tree corpora (IoT-23's raw ``conn.log.labeled`` files): one file per capture, in many sub-folders; all of them are used.
"""
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

MAX_DEPTH = 4            # flat corpora: how deep below a search root to look
MAX_TREE_DEPTH = 9       # tree corpora (the IoT-23 archive extracts to opt/Malware-Project/BigDataset/IoTScenarios/<capture>/bro/)
SKIP_DIRS = {".fedids_cache", "__macosx", "__pycache__", "results", "node_modules"}


def name_matches(name: str, patterns: Sequence[str]) -> bool:
    """Case-insensitive ``fnmatch`` of a file name against any of the patterns."""
    low = name.lower()
    return any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns)


@dataclass
class Found:
    files: List[str]                                   # absolute paths, sorted, the files to read
    directory: str                                     # the directory (flat) or the search root (tree) they were found under
    alternatives: List[Tuple[str, int]] = field(default_factory=list)   # other matching directories (NOT read) + file counts
    how: str = "pattern"                               # "pattern" | "name" | "tree"


def _walk(root: str, max_depth: int) -> Iterator[Tuple[str, int, List[str]]]:
    """Depth-limited, deterministic ``os.walk`` that skips hidden folders and our own cache."""
    root = os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d.lower() not in SKIP_DIRS)
        if depth >= max_depth:
            dirnames[:] = []
        yield dirpath, depth, sorted(f for f in filenames if not f.startswith("."))


def search_roots(data_dir: str, folder: str, extra: Sequence[str] = ()) -> List[str]:
    """Existing, de-duplicated directories to search, most specific first."""
    out: List[str] = []
    seen = set()
    for d in [os.path.join(data_dir, folder), data_dir, *extra]:
        if d and os.path.isdir(d):
            key = os.path.realpath(d)
            if key not in seen:
                seen.add(key)
                out.append(os.path.abspath(d))
    return out


def find_files(
    roots: Sequence[str],
    pool_patterns: Sequence[str],
    names: Sequence[str] = (),
    tree_patterns: Sequence[str] = (),
    pool_all: bool = True,
) -> Optional[Found]:
    """
    Locate the file(s) of one corpus.

    ``pool_all=True``  -> every file of the best directory that matches ``pool_patterns`` (or, failing that, the first of
                          ``names``); for ``tree_patterns`` every match anywhere below the first root that has one.
    ``pool_all=False`` -> a single file: the first of ``names`` that exists, else the first pattern match.
    Returns None if nothing matches.
    """
    # ---- tree corpora (many sub-folders, one file each) ---------------------------------------------------------
    if tree_patterns:
        for root in roots:
            hits: List[str] = []
            for dirpath, _depth, files in _walk(root, MAX_TREE_DEPTH):
                hits += [os.path.join(dirpath, f) for f in files if name_matches(f, tree_patterns)]
            if hits:
                hits = sorted(set(hits))
                return Found(hits if pool_all else hits[:1], os.path.abspath(root), [], "tree")

    # ---- flat corpora ---------------------------------------------------------------------------------------------
    pooled: Dict[str, List[str]] = {}
    named: Dict[str, List[str]] = {}
    order: Dict[str, Tuple[int, int]] = {}
    for ri, root in enumerate(roots):
        for dirpath, depth, files in _walk(root, MAX_DEPTH):
            if dirpath in order:                       # reachable from two roots: count once
                continue
            order[dirpath] = (ri, depth)
            p = [f for f in files if pool_patterns and name_matches(f, pool_patterns)]
            n = [f for f in files if names and name_matches(f, [re_escape_glob(x) for x in names])]
            if p:
                pooled[dirpath] = p
            if n:
                named[dirpath] = n

    def best(d: Dict[str, List[str]]) -> Optional[str]:
        return min(d, key=lambda k: (-len(d[k]), order[k], k)) if d else None

    def first_named(files: List[str]) -> Optional[str]:
        for nm in names:
            for f in files:
                if f.lower() == nm.lower():
                    return f
        return files[0] if files else None

    if pool_all:
        if pooled:
            d = best(pooled)
            return Found([os.path.join(d, f) for f in pooled[d]], d,
                         sorted(((k, len(v)) for k, v in pooled.items() if k != d), key=lambda kv: -kv[1]), "pattern")
        if named:
            d = best(named)
            f = first_named(named[d])
            return Found([os.path.join(d, f)], d,
                         sorted(((k, len(v)) for k, v in named.items() if k != d), key=lambda kv: -kv[1]), "name")
        return None

    if named:
        d = best(named)
        return Found([os.path.join(d, first_named(named[d]))], d, [(k, len(v)) for k, v in named.items() if k != d], "name")
    if pooled:
        d = best(pooled)
        return Found([os.path.join(d, pooled[d][0])], d, [(k, len(v)) for k, v in pooled.items() if k != d], "pattern")
    return None


def re_escape_glob(name: str) -> str:
    """Make a literal file name safe to use as an fnmatch pattern (``[``, ``*``, ``?`` are escaped)."""
    return "".join(f"[{c}]" if c in "*?[" else c for c in name)


def hint_messages(roots: Sequence[str], hints: Dict[str, str]) -> List[str]:
    """Messages for known-but-unsupported layouts that are present (e.g. UNSW-NB15's header-less raw CSVs)."""
    out: List[str] = []
    for pattern, message in hints.items():
        for root in roots:
            if any(name_matches(f, [pattern]) for _d, _dp, files in _walk(root, MAX_DEPTH) for f in files):
                out.append(message)
                break
    return out
