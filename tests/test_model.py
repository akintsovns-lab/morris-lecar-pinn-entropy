import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ml_model import P, rhs, jac_rhs, currents, m_inf, w_inf, tau_w, I_eq, y0


def test_parameters_table1():
    assert (P["C"], P["gCa"], P["gK"], P["gL"]) == (20.0, 4.0, 8.0, 2.0)
    assert (P["VCa"], P["VK"], P["VL"]) == (120.0, -84.0, -60.0)
    assert (P["V1"], P["V2"], P["V3"], P["V4"]) == (-1.2, 18.0, 12.0, 17.4)
    assert abs(P["phi"] - 1 / 15) < 1e-15


def test_V_nullcline():
    V = np.linspace(-70, 40, 200)
    for I in (30.0, 40.0, 50.0):
        ICa, _, IL = currents(V, 0.0)
        w = (I - ICa - IL) / (P["gK"] * (V - P["VK"]))
        d = rhs(0, np.vstack([V, w]), I)
        assert np.max(np.abs(d[0])) < 1e-12


def test_w_nullcline():
    V = np.linspace(-70, 40, 200)
    d = rhs(0, np.vstack([V, w_inf(V)]), 40.0)
    assert np.max(np.abs(d[1])) < 1e-15


def test_current_signs():
    V = np.linspace(-80, 100, 100)
    ICa, IK, IL = currents(V, 0.3)
    assert np.all(ICa < 0)                         # inward Ca current below V_Ca
    assert np.all(IK > 0)                          # outward K current above V_K
    assert np.all(np.sign(IL) == np.sign(V - P["VL"]) )
    # positive applied current depolarizes
    assert rhs(0, np.array([-60.0, 0.01]), 50.0)[0] > rhs(0, np.array([-60.0, 0.01]), 30.0)[0]


def test_gating_ranges():
    V = np.linspace(-100, 100, 500)
    for f in (m_inf, w_inf):
        assert np.all((f(V) > 0) & (f(V) < 1))
    assert np.all((tau_w(V) > 0) & (tau_w(V) <= 1))


def test_jacobian_fd():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = np.array([rng.uniform(-60, 30), rng.uniform(0, 0.5)]); I = rng.uniform(30, 50)
        J = jac_rhs(0, y, I)
        h = np.array([1e-6, 1e-8])
        for k in range(2):
            e = np.zeros(2); e[k] = h[k]
            fd = (rhs(0, y + e, I) - rhs(0, y - e, I)) / (2 * h[k])
            assert np.allclose(J[:, k], fd, rtol=1e-6, atol=1e-9)


def test_equilibrium_current():
    for V in (-40.0, -30.0, -20.0):
        I = I_eq(V)
        assert np.max(np.abs(rhs(0, np.array([V, w_inf(V)]), I))) < 1e-12


def test_initial_condition():
    v, w = y0()
    assert v == -60.0 and abs(w - w_inf(-60.0)) < 1e-15
