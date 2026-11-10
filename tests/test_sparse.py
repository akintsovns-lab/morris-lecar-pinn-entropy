import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sparse import current_split, make_mask, val_blocks


def test_split_disjoint_and_both_sides():
    s = current_split()
    a, b, c = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (a & b) and not (a & c) and not (b & c)
    assert len(a | b | c) == len(s["all"])
    for k in ("val", "test"):
        assert (s[k] < 39.963).any() and (s[k] > 39.963).any()


def test_mask_fraction_and_ends():
    n = 10001
    for rho in (0.5, 0.2, 0.1, 0.05):
        m = make_mask(n, rho, 1)
        assert m[0] and m[-1]
        assert abs(m.sum() - round(rho * n)) <= 2


def test_masks_reproducible_and_distinct():
    assert np.array_equal(make_mask(10001, 0.1, 7), make_mask(10001, 0.1, 7))
    assert not np.array_equal(make_mask(10001, 0.1, 7), make_mask(10001, 0.1, 8))


def test_val_blocks_fraction():
    t = np.round(np.arange(0, 1000.0001, 0.1), 10)
    f = val_blocks(t).mean()
    assert 0.04 < f < 0.06
