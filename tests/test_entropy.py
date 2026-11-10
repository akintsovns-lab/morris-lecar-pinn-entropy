import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from entropy import sampen, perm_entropy, spectral_entropy, all_entropies


def test_white_noise_limits():
    x = np.random.default_rng(0).standard_normal(4000)
    assert perm_entropy(x, 4, 1) > 0.99
    assert perm_entropy(x, 4, 1, True) > 0.98
    assert spectral_entropy(x, nperseg=256) > 0.97
    se, _ = sampen(x, 2, 0.2)
    assert 2.0 < se < 2.5  # theoretical ~2.2 for Gaussian white noise, m=2, r=0.2


def test_sine_is_regular():
    t = np.arange(4000)
    x = np.sin(2 * np.pi * t / 50.0)
    assert perm_entropy(x, 4, 1) < 0.35
    assert spectral_entropy(x, nperseg=256) < 0.35
    se, _ = sampen(x, 2, 0.2)
    assert se < 0.3


def test_logistic_map_pe():
    x = np.empty(20000); x[0] = 0.4
    for i in range(1, len(x)):
        x[i] = 4 * x[i - 1] * (1 - x[i - 1])
    # fully chaotic logistic map has forbidden ordinal patterns: PE(d=4) clearly below 1
    pe = perm_entropy(x, 4, 1)
    assert 0.5 < pe < 0.9


def test_against_ordpy_antropy():
    import ordpy, antropy
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(3000))
    for d, tau in [(3, 1), (4, 2), (5, 1)]:
        a = perm_entropy(x, d, tau)
        b = ordpy.permutation_entropy(x, dx=d, taux=tau, normalized=True)
        assert abs(a - b) < 1e-10
        c = antropy.perm_entropy(x, order=d, delay=tau, normalize=True)
        assert abs(a - c) < 1e-10
    y = rng.standard_normal(2000)
    se, _ = sampen(y, 2, 0.2)
    se2 = antropy.sample_entropy(y, order=2, tolerance=0.2 * np.std(y))
    assert abs(se - se2) < 2e-2  # antropy uses N-m+1 templates for B; small finite-size difference


def test_wpe_against_entropyhub():
    import math
    if not hasattr(np, "math"):
        np.math = math  # EntropyHub 2.0 still references np.math (removed in NumPy 2)
    import EntropyHub as EH
    rng = np.random.default_rng(2)
    x = np.sin(np.arange(3000) / 7.0) + 0.3 * rng.standard_normal(3000)
    a = perm_entropy(x, 4, 1, weighted=True)
    Perm, Pnorm, cPE = EH.PermEn(x, m=4, tau=1, Typex="weighted", Logx=np.e, Norm=True)
    assert abs(a - Pnorm[-1]) < 1e-6


def test_scale_invariance_and_quiescent_rule():
    rng = np.random.default_rng(3)
    x = np.sin(np.arange(2000) / 9.0) + 0.1 * rng.standard_normal(2000)
    e1 = all_entropies(10 * x); e2 = all_entropies(100 * x)
    for k in ("SampEn", "PE", "WPE", "SpEn"):
        assert abs(e1[k] - e2[k]) < 1e-9
    q = all_entropies(np.full(1000, -30.0) + 1e-6 * rng.standard_normal(1000))
    assert q["quiescent"] and q["PE"] == 0.0
