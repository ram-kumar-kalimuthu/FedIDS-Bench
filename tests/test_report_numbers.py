"""numbers.tex: the counts the manuscript quotes come from the run folders, never from the text."""
import pandas as pd

from fedids_bench.reporting import paper as P


def test_numbers_tex_counts_runs_corpora_and_seeds():
    sts = [P.GridStatus(name="main", run_dir="x", n_expected=250, n_found=250),
           P.GridStatus(name="partitioning", run_dir="y", n_expected=350, n_found=340)]
    main = pd.DataFrame({"dataset": ["a", "a", "b", "b"], "seed": [0, 1, 0, 1]})
    tex = P.numbers_tex(sts, {"main": main})
    assert r"\newcommand{\nRunsMain}{250}" in tex
    assert r"\newcommand{\nRunsPartitioning}{340}" in tex      # found, not expected: a partial grid must not claim its full size
    assert r"\newcommand{\nRunsTotal}{590}" in tex
    assert r"\newcommand{\nCorpora}{2}" in tex and r"\newcommand{\nSeeds}{2}" in tex


def test_numbers_tex_without_main_frame_has_no_corpus_macro():
    tex = P.numbers_tex([P.GridStatus(name="sweeps", run_dir="z", n_found=5)], {})
    assert r"\nRunsSweeps}{5}" in tex and "nCorpora" not in tex
