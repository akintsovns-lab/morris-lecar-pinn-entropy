"""Reconstruction metrics (per current) on the 0.1 ms grid of the 1000 ms window."""
from __future__ import annotations
import numpy as np
from scipy.signal import welch

THR = -10.0  # mV, spike detection threshold (upward crossing), same as the Poincare section


def crossings(t, V, thr=THR):
    i = np.where((V[:-1] < thr) & (V[1:] >= thr))[0]
    return t[i] + (thr - V[i]) * (t[i + 1] - t[i]) / (V[i + 1] - V[i])


def peak_times(t, V, thr=THR, win=30.0):
    out = []
    for c in crossings(t, V, thr):
        sel = (t >= c) & (t <= c + win)
        if sel.any():
            k = np.argmax(V[sel]); out.append((t[sel][k], V[sel][k]))
    return np.array(out).reshape(-1, 2)


def spectral_error(V, Vr, dec=10):
    a, b = V[::dec], Vr[::dec]
    if np.std(b) < 0.1:
        return float("nan")
    _, Pa = welch(a - a.mean(), fs=1.0, nperseg=256); _, Pb = welch(b - b.mean(), fs=1.0, nperseg=256)
    return float(0.5 * np.sum(np.abs(Pa / Pa.sum() - Pb / Pb.sum())))


def metrics(t, Vh, Vr, wh=None, wr=None):
    m = dict(rmse_V=float(np.sqrt(np.mean((Vh - Vr) ** 2))),
             relL2_V=float(np.linalg.norm(Vh - Vr) / np.linalg.norm(Vr)),
             maxerr_V=float(np.max(np.abs(Vh - Vr))))
    sdr = np.std(Vr)
    m["nrmse_V"] = float(m["rmse_V"] / sdr) if sdr > 0.1 else float("nan")
    if wh is not None and np.all(np.isfinite(wh)):
        m["relL2_w"] = float(np.linalg.norm(wh - wr) / np.linalg.norm(wr))
        m["rmse_w"] = float(np.sqrt(np.mean((wh - wr) ** 2)))
    else:
        m["relL2_w"] = m["rmse_w"] = float("nan")
    pr, ph = peak_times(t, Vr), peak_times(t, Vh)
    m["n_spikes_ref"], m["n_spikes_hat"] = int(len(pr)), int(len(ph))
    m["amp_err"] = float(abs((Vh.max() - Vh.min()) - (Vr.max() - Vr.min())))
    if len(pr):
        dt, dv = [], []
        for tp, vp in pr:
            if len(ph):
                k = np.argmin(np.abs(ph[:, 0] - tp))
                if abs(ph[k, 0] - tp) < 20.0:
                    dt.append(abs(ph[k, 0] - tp)); dv.append(abs(ph[k, 1] - vp))
        m["peak_time_err"] = float(np.mean(dt)) if dt else float("nan")
        m["peak_V_err"] = float(np.mean(dv)) if dv else float("nan")
        m["frac_spikes_matched"] = float(len(dt) / len(pr))
    else:
        m["peak_time_err"] = m["peak_V_err"] = m["frac_spikes_matched"] = float("nan")
    if len(pr) > 1 and len(ph) > 1:
        Tr, Th = np.mean(np.diff(pr[:, 0])), np.mean(np.diff(ph[:, 0]))
        m["period_relerr"] = float(abs(Th - Tr) / Tr)
    else:
        m["period_relerr"] = float("nan")
    m["spectral_err"] = spectral_error(Vh, Vr)
    return m
