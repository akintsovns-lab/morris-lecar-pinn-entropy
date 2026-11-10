"""High-accuracy reference solutions of the Morris-Lecar model (Revised Section 2.2).

For every applied current: integrate a nominal transient, test convergence (equilibrium or
limit-cycle criterion, Eq. 16), extend in blocks if needed, then generate a fresh observation
interval of fixed length T_obs on the uniform dense-output grid.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import yaml
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_model import rhs, jac_rhs, y0, P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def load_cfg():
    with open(os.path.join(ROOT, "configs", "reference.yaml")) as f:
        return yaml.safe_load(f)


def _ev_section(level):
    def ev(t, y, I):
        return y[0] - level
    ev.direction = 1.0
    return ev


def _ev_peak(t, y, I):
    return rhs(t, y, I)[0]
_ev_peak.direction = -1.0


def integrate(I, t_span, yinit, method="DOP853", rtol=1e-10, atol=(1e-10, 1e-12), t_eval=None, events=None,
              dense=False):
    kw = dict(method=method, rtol=rtol, atol=list(atol) if np.ndim(atol) else atol, dense_output=dense, args=(I,),
              t_eval=t_eval, events=events)
    if method in ("Radau", "BDF", "LSODA"):
        kw["jac"] = lambda t, y, I: jac_rhs(t, y, I)
    sol = solve_ivp(rhs, t_span, yinit, **kw)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def _conv_test(t_end, Ytail, cross_t, peak_V, cfg, I):
    d = rhs(0, Ytail, I)
    if np.max(np.abs(d[0])) < cfg["eq_tol_V"] and np.max(np.abs(d[1])) < cfg["eq_tol_w"]:
        return "equilibrium", dict(maxdV=float(np.max(np.abs(d[0]))), maxdw=float(np.max(np.abs(d[1]))))
    n = cfg["cyc_n"]
    if len(cross_t) >= n + 2 and len(peak_V) >= n + 1:
        per = np.diff(cross_t)[-(n + 1):]
        pk = np.asarray(peak_V)[-(n + 1):]
        rp = np.max(np.abs(np.diff(per)) / per[1:])
        rv = np.max(np.abs(np.diff(pk)) / np.abs(pk[1:]))
        if rp < cfg["cyc_rel_tol"] and rv < cfg["cyc_rel_tol"]:
            return "cycle", dict(period=float(per[-1]), rel_dper=float(rp), rel_dpeak=float(rv),
                                 n_cross=int(len(cross_t)))
    return None, dict(n_cross=int(len(cross_t)))


def reference_trajectory(I, cfg=None, T_obs=None, dt_store=1.0, keep_dense=0.0, method="DOP853", rtol=None,
                         atol=None, dt_dense=None):
    """Converged reference for current I (events-based convergence test, Eq. 16 and cycle criterion).

    Returns long-window samples at dt_store (1 ms) and the first keep_dense ms on the dt_out grid."""
    cfg = cfg or load_cfg()
    T_obs = T_obs or cfg["T_obs"]
    rtol = rtol or cfg["rtol"]
    atol = atol or cfg["atol"]
    dt_dense = dt_dense or cfg["dt_out"]
    evs = [_ev_section(cfg["poincare_V"]), _ev_peak]
    t_end = cfg["t_tr"]
    L = cfg["eq_check_len"]
    te = np.arange(t_end - L, t_end + 1e-9, cfg["dt_out"])
    sol = integrate(I, (0.0, t_end), y0(), method, rtol, atol, t_eval=te, events=evs)
    cross = list(sol.t_events[0]); peaks = [y[0] for y in sol.y_events[1] if y[0] > cfg["poincare_V"]]
    status, info = _conv_test(t_end, sol.y, cross, peaks, cfg, I)
    y_cur = sol.y[:, -1]
    n_ext = 0
    t_lim = cfg["t_tr"] + cfg["t_max_extra"]
    while status is None and t_end < t_lim:
        blk = cfg["extend_block"]
        te = np.arange(t_end + blk - L, t_end + blk + 1e-9, cfg["dt_out"])
        s2 = integrate(I, (t_end, t_end + blk), y_cur, method, rtol, atol, t_eval=te, events=evs)
        cross += list(s2.t_events[0]); peaks += [y[0] for y in s2.y_events[1] if y[0] > cfg["poincare_V"]]
        t_end += blk; y_cur = s2.y[:, -1]; n_ext += 1
        status, info = _conv_test(t_end, s2.y, cross, peaks, cfg, I)
    converged = status is not None
    ts = np.round(np.arange(0.0, T_obs + 1e-9, dt_store), 10)
    t_eval = t_end + ts
    if keep_dense > 0:
        td = np.round(np.arange(0.0, keep_dense + 1e-9, dt_dense), 10)
        t_eval = np.unique(np.r_[t_end + ts, t_end + td])
    obs = integrate(I, (t_end, t_end + T_obs), y_cur, method, rtol, atol, t_eval=t_eval, events=evs)
    tt = obs.t - t_end
    def pick(grid):
        idx = np.searchsorted(tt, grid - 1e-7)
        return obs.y[:, idx]
    Ys = pick(ts)
    out = dict(I=float(I), t_conv=float(t_end), status=status or "nonconverged", converged=bool(converged),
               n_ext=n_ext, info=info, t_store=ts, V=Ys[0], w=Ys[1], y_start=y_cur)
    if keep_dense > 0:
        Yd = pick(td)
        out.update(t_dense=td, V_dense=Yd[0], w_dense=Yd[1])
    ct = obs.t_events[0] - t_end
    pk = np.array([y[0] for y in obs.y_events[1] if y[0] > cfg["poincare_V"]])
    out.update(n_spikes=int(len(ct)), spike_times=ct, Vpeak_mean=float(pk.mean()) if len(pk) else float("nan"),
               Vmax=float(max(obs.y[0].max(), pk.max() if len(pk) else -1e9)), Vmin=float(obs.y[0].min()),
               period=float(np.mean(np.diff(ct))) if len(ct) > 1 else float("nan"), nfev=int(obs.nfev))
    if status == "equilibrium":
        out.update(V_eq=float(obs.y[0, -1]), w_eq=float(obs.y[1, -1]))
    return out
