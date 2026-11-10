import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from benchmark import detect_derivative, detect_changepoint, auc_sep, holm
import ruptures as rpt


def test_step_detection():
    I = np.round(np.arange(39.5, 40.51, 0.05), 4)
    H = np.where(I > 40.0, 1.0, 0.0)
    Is, _ = detect_derivative(I, H, Ic=40.02)
    assert abs(Is - 40.025) < 1e-9
    assert abs(detect_changepoint(I, H, Ic=40.02) - 40.025) < 1e-9
    assert auc_sep(I, H, Ic=40.02) == 1.0


def test_changepoint_matches_ruptures():
    rng = np.random.default_rng(1)
    I = np.linspace(30, 50, 81); H = (I > 41).astype(float) + 0.2 * rng.standard_normal(81)
    cp = detect_changepoint(I, H, Ic=100.0, excl=0.0)
    bk = rpt.Binseg(model="l2", min_size=1, jump=1).fit(H).predict(n_bkps=1)[0]
    assert abs(cp - 0.5 * (I[bk - 1] + I[bk])) < 1e-12


def test_holm():
    a = holm([0.01, 0.04, 0.03])
    assert np.allclose(a, [0.03, 0.06, 0.06])
