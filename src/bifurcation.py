"""Independent bifurcation analysis (ground truth for I_crit), Revised Section 2.3.

Method 1: equilibrium branch parametrized by V* (Eq. 20), Jacobian eigenvalues, fold (det J = 0)
          located by root finding on dI/dV*, non-degeneracy and transversality checks.
Method 2: continuation of the periodic orbit with a Poincare-section shooting method
          (fixed point of the return map), Floquet multipliers from the monodromy (Liouville
          formula and return-map derivative), period scaling T ~ (I - I_c)^(-1/2) and extrapolation
          of I_c from T^-2 -> 0.
Method 3: direct-simulation bracketing of the onset of sustained firing.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_model import P, m_inf, w_inf, tau_w, dm_inf, dw_inf, I_eq, rhs, jac_rhs, y0  # noqa: E402


def dI_dV(V, p=P):
    return (p["gCa"] * (dm_inf(V, p) * (V - p["VCa"]) + m_inf(V, p))
            + p["gK"] * (dw_inf(V, p) * (V - p["VK"]) + w_inf(V, p)) + p["gL"])


def d2I_dV2(V, h=1e-4):
    return (dI_dV(V + h) - dI_dV(V - h)) / (2 * h)


def jac_eq(V, p=P):
    """Jacobian at equilibrium (Eqs. 22-24)."""
    w = w_inf(V, p)
    FV = -(p["gCa"] * (dm_inf(V, p) * (V - p["VCa"]) + m_inf(V, p)) + p["gK"] * w + p["gL"]) / p["C"]
    Fw = -p["gK"] * (V - p["VK"]) / p["C"]
    GV = p["phi"] / tau_w(V, p) * dw_inf(V, p)
    Gw = -p["phi"] / tau_w(V, p)
    return np.array([[FV, Fw], [GV, Gw]])


def equilibrium_branch(Vmin=-70.0, Vmax=60.0, n=13001):
    V = np.linspace(Vmin, Vmax, n)
    I = I_eq(V)
    tr, det, ev = [], [], []
    for v in V:
        J = jac_eq(v)
        tr.append(np.trace(J)); det.append(np.linalg.det(J))
        ev.append(np.linalg.eigvals(J))
    ev = np.array(ev)
    return dict(V=V, I=I, w=w_inf(V), tr=np.array(tr), det=np.array(det),
                ev_re=np.real(ev), ev_im=np.imag(ev))


def find_folds(br):
    d = dI_dV(br["V"])
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    folds = []
    for i in idx:
        Vf = brentq(dI_dV, br["V"][i], br["V"][i + 1], xtol=1e-14, rtol=1e-15)
        J = jac_eq(Vf)
        ev = np.linalg.eigvals(J)
        folds.append(dict(V=float(Vf), I=float(I_eq(Vf)), w=float(w_inf(Vf)), detJ=float(np.linalg.det(J)),
                          trJ=float(np.trace(J)), eig=[float(np.real(e)) for e in ev],
                          d2I_dV2=float(d2I_dV2(Vf)), dF_dI=1.0 / P["C"]))
    return folds


def find_hopf(br):
    tr = br["tr"]
    idx = np.where(np.sign(tr[:-1]) != np.sign(tr[1:]))[0]
    out = []
    for i in idx:
        f = lambda v: np.trace(jac_eq(v))
        Vh = brentq(f, br["V"][i], br["V"][i + 1], xtol=1e-13)
        J = jac_eq(Vh)
        out.append(dict(V=float(Vh), I=float(I_eq(Vh)), detJ=float(np.linalg.det(J)),
                        omega=float(np.sqrt(max(np.linalg.det(J), 0))), is_hopf=bool(np.linalg.det(J) > 0)))
    return out


# ---------------------------------------------------------------------------------------------
# periodic orbits by Poincare-section shooting
SEC_V = -10.0


def _return_map(w0, I, t_max=4e5, rtol=1e-11, atol=(1e-11, 1e-13)):
    """Start on section V=SEC_V (upward) with w=w0; integrate to next upward crossing."""
    def ev(t, y, I):
        return y[0] - SEC_V
    ev.direction = 1.0
    ev.terminal = False

    def aug(t, y, I):
        f = rhs(t, y[:2], I)
        J = jac_rhs(t, y[:2], I)
        return np.array([f[0], f[1], np.trace(J)])
    def ev3(t, y, I):
        return y[0] - SEC_V
    ev3.direction = 1.0
    ev3.terminal = True
    # leave the section first (integrate 5 ms), then detect the next crossing
    s0 = solve_ivp(aug, (0, 5.0), [SEC_V, w0, 0.0], args=(I,), method="DOP853", rtol=rtol, atol=list(atol) + [1e-10])
    s = solve_ivp(aug, (5.0, t_max), s0.y[:, -1], args=(I,), method="DOP853", rtol=rtol,
                  atol=list(atol) + [1e-10], events=ev3)
    if len(s.t_events[0]) == 0:
        return None
    yT = s.y_events[0][0]
    return dict(w=float(yT[1]), T=float(s.t_events[0][0]), int_div=float(yT[2]))


def periodic_orbit(I, w_guess, tol=1e-12, maxit=40):
    """Secant iteration on g(w) = P(w) - w."""
    def g(w):
        r = _return_map(w, I)
        return (np.nan if r is None else r["w"] - w), r
    w0, w1 = w_guess, w_guess * (1 + 1e-3) + 1e-6
    g0, r0 = g(w0); g1, r1 = g(w1)
    for _ in range(maxit):
        if not np.isfinite(g1) or not np.isfinite(g0) or g1 == g0:
            break
        w2 = w1 - g1 * (w1 - w0) / (g1 - g0)
        w0, g0, r0 = w1, g1, r1
        w1 = w2
        g1, r1 = g(w1)
        if np.isfinite(g1) and abs(g1) < tol:
            break
    if r1 is None or not np.isfinite(g1):
        return None
    # return-map derivative (finite difference) as second estimate of the nontrivial multiplier
    h = 1e-7
    rp = _return_map(w1 + h, I); rm = _return_map(w1 - h, I)
    dP = (rp["w"] - rm["w"]) / (2 * h) if (rp and rm) else np.nan
    # extrema along the orbit
    s = solve_ivp(rhs, (0, r1["T"]), [SEC_V, w1], args=(I,), method="DOP853", rtol=1e-11, atol=[1e-11, 1e-13],
                  dense_output=True)
    tt = np.linspace(0, r1["T"], 20001)
    Y = s.sol(tt)
    return dict(I=float(I), w_sec=float(w1), T=r1["T"], mult_liouville=float(np.exp(r1["int_div"])),
                mult_returnmap=float(dP), Vmin=float(Y[0].min()), Vmax=float(Y[0].max()), residual=float(g1),
                orbit=Y)


def continue_cycles(I_c_guess, deltas, w_start=None):
    out = []
    w_guess = w_start
    for d in deltas:
        I = I_c_guess + d
        if w_guess is None:
            sol = solve_ivp(rhs, (0, 5000), y0(), args=(I,), method="DOP853", rtol=1e-10, atol=[1e-10, 1e-12],
                            dense_output=True)
            tt = np.arange(3000, 5000, 0.05); Y = sol.sol(tt)
            k = np.where((Y[0][:-1] < SEC_V) & (Y[0][1:] >= SEC_V))[0][-1]
            w_guess = float(Y[1][k])
        po = periodic_orbit(I, w_guess)
        if po is None:
            out.append(dict(I=float(I), failed=True))
            continue
        w_guess = po["w_sec"]
        out.append(po)
    return out


def onset_bracket(I_lo, I_hi, T_sim=60000.0, n_bisect=14):
    """Direct simulation from the common IC: 'spiking' if >= 2 upward crossings of SEC_V occur in the
    last half of a T_sim record. Returns bracket (I_lo rest, I_hi spiking)."""
    def spikes(I):
        s = solve_ivp(rhs, (0, T_sim), y0(), args=(I,), method="DOP853", rtol=1e-10, atol=[1e-10, 1e-12],
                      dense_output=True)
        tt = np.arange(T_sim / 2, T_sim, 0.1); V = s.sol(tt)[0]
        return int(np.sum((V[:-1] < SEC_V) & (V[1:] >= SEC_V)))
    assert spikes(I_lo) < 2 and spikes(I_hi) >= 2
    for _ in range(n_bisect):
        m = 0.5 * (I_lo + I_hi)
        if spikes(m) >= 2:
            I_hi = m
        else:
            I_lo = m
    return I_lo, I_hi
