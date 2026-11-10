"""Entropy estimators (own implementations, cross-checked against antropy/ordpy/EntropyHub in tests).

SampEn  : Richman & Moorman sample entropy, tolerance r = r_frac * SD of the analysed window,
          Chebyshev distance, self-matches excluded, N-m templates for both m and m+1.
          If no (m+1)-matches (A = 0) or no m-matches (B = 0) occur, the conventional upper bound
          -ln(2 / ((N-m-1)(N-m))) is returned (flagged).
PE      : Bandt-Pompe permutation entropy, order d, delay tau (samples), normalized by ln(d!).
          Ties are broken by order of occurrence (stable argsort).
WPE     : Fadlallah et al. weighted permutation entropy; weight of each embedding vector equals its
          variance; same d and tau as PE; normalized by ln(d!).
SpEn    : normalized Shannon entropy of the Welch PSD (Hann window, fixed segment length and
          overlap, constant detrend), restricted to a fixed frequency band excluding DC,
          normalized by ln(number of frequency bins).
Quiescent rule: if SD(window) < sd_floor, the window is declared quiescent and every measure is 0
          (a constant signal is perfectly regular); the flag is returned.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit
from scipy.signal import welch


@njit(cache=True)
def _sampen_counts(x, m, r):
    N = x.shape[0]
    n_t = N - m
    B = 0
    A = 0
    for i in range(n_t - 1):
        for j in range(i + 1, n_t):
            ok = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > r:
                    ok = False
                    break
            if ok:
                B += 1
                if abs(x[i + m] - x[j + m]) <= r:
                    A += 1
    return A, B


def sampen(x, m=2, r_frac=0.2, sd=None):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    sd = np.std(x) if sd is None else sd
    r = r_frac * sd
    A, B = _sampen_counts(x, m, r)
    bound = -math.log(2.0 / ((N - m - 1) * (N - m)))
    if A == 0 or B == 0:
        return bound, True
    return -math.log(A / B), False


@njit(cache=True)
def _ordinal_codes(x, d, tau):
    n = x.shape[0] - (d - 1) * tau
    codes = np.empty(n, dtype=np.int64)
    var = np.empty(n, dtype=np.float64)
    v = np.empty(d, dtype=np.float64)
    for i in range(n):
        for k in range(d):
            v[k] = x[i + k * tau]
        idx = np.argsort(v, kind="mergesort")  # stable
        # Lehmer-like code of the permutation (base-d number)
        c = 0
        for k in range(d):
            c = c * d + idx[k]
        codes[i] = c
        mu = v.mean()
        s = 0.0
        for k in range(d):
            s += (v[k] - mu) ** 2
        var[i] = s / d
    return codes, var


def _shannon_from_counts(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def perm_entropy(x, d=4, tau=2, weighted=False):
    x = np.asarray(x, dtype=np.float64)
    codes, var = _ordinal_codes(x, d, tau)
    u, inv = np.unique(codes, return_inverse=True)
    if weighted:
        tot = var.sum()
        if tot <= 0:
            return 0.0
        p = np.bincount(inv, weights=var) / tot
    else:
        p = np.bincount(inv) / len(codes)
    return _shannon_from_counts(p) / math.log(math.factorial(d))


def spectral_entropy(x, fs=1.0, nperseg=256, noverlap=128, fmin=None, fmax=None):
    """fs in kHz if time is in ms; returns normalized SpEn on band (fmin, fmax]."""
    x = np.asarray(x, dtype=np.float64)
    nps = min(nperseg, len(x))
    f, Pxx = welch(x, fs=fs, window="hann", nperseg=nps, noverlap=min(noverlap, nps // 2), detrend="constant",
                   scaling="density")
    lo = f[1] if fmin is None else fmin
    hi = f[-1] if fmax is None else fmax
    sel = (f >= lo - 1e-12) & (f <= hi + 1e-12)
    P = Pxx[sel]
    tot = P.sum()
    if tot <= 0:
        return 0.0
    p = P / tot
    return _shannon_from_counts(p) / math.log(len(P))


DEFAULTS = dict(m=2, r_frac=0.2, d=4, tau=2, nperseg=256, noverlap=128, sd_floor=0.1, fs=1.0)


def all_entropies(x, prm=None):
    """Return dict with SampEn, PE, WPE, SpEn and flags for one window."""
    q = dict(DEFAULTS)
    if prm:
        q.update(prm)
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd < q["sd_floor"]:
        return dict(SampEn=0.0, PE=0.0, WPE=0.0, SpEn=0.0, quiescent=True, sampen_bound=False, sd=sd)
    se, flag = sampen(x, q["m"], q["r_frac"], sd)
    return dict(SampEn=se, PE=perm_entropy(x, q["d"], q["tau"]), WPE=perm_entropy(x, q["d"], q["tau"], True),
                SpEn=spectral_entropy(x, q["fs"], q["nperseg"], q["noverlap"]), quiescent=False,
                sampen_bound=flag, sd=sd)


MEASURES = ("SampEn", "PE", "WPE", "SpEn")


def windowed(x, win, step, prm=None):
    """Entropies on sliding windows of `win` samples with `step`; returns list of dicts."""
    out = []
    for a in range(0, len(x) - win + 1, step):
        out.append(all_entropies(x[a:a + win], prm))
    return out
