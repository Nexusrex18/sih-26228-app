"""Reproducibility is a named PS deliverable, so it is asserted, not intended."""
import numpy as np

from attacklab.synth import make_corpus, poison


def test_same_seed_same_corpus_bytes():
    a = poison(make_corpus(200, seed=5), 0.1, 0, "patch", seed=5)
    b = poison(make_corpus(200, seed=5), 0.1, 0, "patch", seed=5)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert np.array_equal(a.poisoned, b.poisoned)


def test_different_seed_different_corpus():
    a = poison(make_corpus(200, seed=5), 0.1, 0, "patch", seed=5)
    b = poison(make_corpus(200, seed=6), 0.1, 0, "patch", seed=6)
    assert not np.array_equal(a.x, b.x)
