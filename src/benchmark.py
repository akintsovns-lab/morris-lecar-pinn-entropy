"""Transition detection from entropy curves H_k(I) and statistics (Stage 6)."""
from __future__ import annotations
import os, json
import numpy as np, yaml
from scipy.ndimage import median_filter
from scipy.stats import wilcoxon, kendalltau

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BCFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "benchmark.yaml")))
IC = json.load(open(os.path.join(ROOT, "results", "bifurcation.json")))["I_SN"] if os.path.exists(
    os.path.join(ROOT, "results", "bifurcation.json")) else 39.96315309274537


def _prep(I, H, Ic=IC, excl=None):
    excl = BCFG["I_crit_exclusion"] if excl is None else excl
    I = np.asarray(I, float); H = np.asarray(H, float)
    o = np.argsort(I); I, H = I[o], H[o]
    keep = (np.abs(I - Ic) >= excl) & np.isfinite(H)
    return I[keep], H[keep]


def smooth(I, H, h=None, med=None):
    h = BCFG["smoothing"]["bandwidth"] if h is None else h
    med = BCFG["median_window"] if med is None else med
    Hm = median_filter(H, size=med, mode="nearest") if med and med > 1 else H
    K = np.exp(-0.5 * ((I[:, None] - I[None, :]) / h) ** 2)
    return (K @ Hm) / K.sum(1)


def detect_derivative(I, H, Ic=IC, h=None, med=None, excl=None):
    """I* = midpoint of the grid interval with maximal |slope| of the smoothed curve."""
    I, H = _prep(I, H, Ic, excl)
    S = smooth(I, H, h, med)
    sl = np.diff(S) / np.diff(I)
    k = int(np.argmax(np.abs(sl)))
    return float(0.5 * (I[k] + I[k + 1])), float(sl[k])


def detect_changepoint(I, H, Ic=IC, excl=None):
    """Least-squares single step change point (exact CUSUM/binary segmentation with one break)."""
    I, H = _prep(I, H, Ic, excl)
    n = len(H)
    c1 = np.cumsum(H); c2 = np.cumsum(H ** 2)
    best, kb = np.inf, 1
    for k in range(1, n):
        l = c2[k - 1] - c1[k - 1] ** 2 / k
        r = (c2[-1] - c2[k - 1]) - (c1[-1] - c1[k - 1]) ** 2 / (n - k)
        if l + r < best:
            best, kb = l + r, k
    return float(0.5 * (I[kb - 1] + I[kb]))


def auc_sep(I, H, Ic=IC, excl=None):
    """Mann-Whitney AUC = P(H(spiking) > H(rest)) (+0.5 for ties); Cliff's delta = 2 AUC - 1."""
    I, H = _prep(I, H, Ic, excl)
    a, b = H[I > Ic], H[I < Ic]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d = a[:, None] - b[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[k])); adj[k] = run
    return adj


def paired_wilcoxon(x, y):
    """Two-sided Wilcoxon signed-rank; returns p = 1 if all differences are zero."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    if np.all(d == 0):
        return 1.0, 0.0
    try:
        r = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return float(r.pvalue), float(np.median(d))
    except ValueError:
        return 1.0, float(np.median(d))


def kendall_rank(ref_eps, eps):
    t = kendalltau(ref_eps, eps, variant="b")
    return float(t.statistic) if np.isfinite(t.statistic) else float("nan")
