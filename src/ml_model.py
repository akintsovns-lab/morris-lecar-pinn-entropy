"""Morris-Lecar model (Revised Section 2): right-hand side, steady-state functions, Jacobian."""
from __future__ import annotations
import os
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG = os.path.join(_HERE, "..", "configs", "model.yaml")


def load_params(path: str = _CFG) -> dict:
    with open(path) as f:
        p = yaml.safe_load(f)
    p["phi"] = 1.0 / 15.0  # exact value of Table 1
    return p


P = load_params()


def m_inf(V, p=P):
    return 0.5 * (1.0 + np.tanh((V - p["V1"]) / p["V2"]))


def w_inf(V, p=P):
    return 0.5 * (1.0 + np.tanh((V - p["V3"]) / p["V4"]))


def tau_w(V, p=P):
    """Dimensionless tau_w = sech((V-V3)/(2V4)); physical time T_w = tau_w/phi."""
    return 1.0 / np.cosh((V - p["V3"]) / (2.0 * p["V4"]))


def dm_inf(V, p=P):
    return 0.5 / p["V2"] / np.cosh((V - p["V1"]) / p["V2"]) ** 2


def dw_inf(V, p=P):
    return 0.5 / p["V4"] / np.cosh((V - p["V3"]) / p["V4"]) ** 2


def currents(V, w, p=P):
    ICa = p["gCa"] * m_inf(V, p) * (V - p["VCa"])
    IK = p["gK"] * w * (V - p["VK"])
    IL = p["gL"] * (V - p["VL"])
    return ICa, IK, IL


def rhs(t, y, I, p=P):
    V, w = y[0], y[1]
    ICa, IK, IL = currents(V, w, p)
    dV = (-ICa - IK - IL + I) / p["C"]
    dw = p["phi"] * (w_inf(V, p) - w) / tau_w(V, p)
    return np.array([dV, dw]) if np.ndim(V) == 0 else np.vstack([dV, dw])


def jac_rhs(t, y, I, p=P):
    """Full Jacobian (valid off equilibrium, Eq. 24b of the corrections)."""
    V, w = y
    FV = -(p["gCa"] * (dm_inf(V, p) * (V - p["VCa"]) + m_inf(V, p)) + p["gK"] * w + p["gL"]) / p["C"]
    Fw = -p["gK"] * (V - p["VK"]) / p["C"]
    a = (V - p["V3"]) / (2 * p["V4"])
    inv_tau = np.cosh(a)
    d_inv_tau = np.sinh(a) / (2 * p["V4"])
    GV = p["phi"] * (dw_inf(V, p) * inv_tau + (w_inf(V, p) - w) * d_inv_tau)
    Gw = -p["phi"] * inv_tau
    return np.array([[FV, Fw], [GV, Gw]])


def I_eq(V, p=P):
    """Equilibrium current as a function of V* (Eq. 20)."""
    return (p["gCa"] * m_inf(V, p) * (V - p["VCa"]) + p["gK"] * w_inf(V, p) * (V - p["VK"])
            + p["gL"] * (V - p["VL"]))


def y0(p=P):
    return np.array([p["V0"], float(w_inf(p["V0"], p))])
