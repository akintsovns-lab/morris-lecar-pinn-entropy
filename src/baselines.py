"""Baselines: cubic-spline interpolation of sparse V and ODE least-squares fit of the hidden initial state."""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import observations, load_ref, OUT, tag_of  # noqa: E402
from grids import pinn_currents  # noqa: E402
from reference import integrate  # noqa: E402
from ml_model import w_inf  # noqa: E402


def spline_run(scenario, rho, eta, rep, kind="random"):
    tag = tag_of(scenario, rho, eta, rep, "spline", kind)
    fn = os.path.join(OUT, tag + ".npz")
    if os.path.exists(fn):
        return fn
    Is = pinn_currents(); tg = load_ref(Is[0])[0]
    V = np.zeros((len(Is), len(tg))); val = []
    t0 = time.time()
    for j, I in enumerate(Is):
        o = observations(I, j, rho, eta, rep, kind)
        V[j] = CubicSpline(o["t_obs"], o["V_obs"], bc_type="not-a-knot")(tg)
        idx = np.round(o["t_val"] / 0.1).astype(int)
        val.append(float(np.sqrt(np.mean((V[j, idx] - o["V_val"]) ** 2))) if len(idx) else np.nan)
    np.savez_compressed(fn, I=Is, t=tg, V=V.astype(np.float32), w=np.full(V.shape, np.nan, np.float32),
                        val_rmse=np.array(val))
    json.dump(dict(tag=tag, variant="spline", wall_s=time.time() - t0), open(fn[:-4] + ".json", "w"))
    return fn


def _fit_one(I, o, rtol=1e-7, atol=(1e-7, 1e-9)):
    t_obs, V_obs = o["t_obs"], o["V_obs"]

    def resid(th):
        s = integrate(I, (0.0, t_obs[-1]), th, "DOP853", rtol, atol, t_eval=t_obs)
        if s.y.shape[1] != len(t_obs):
            return np.full(len(t_obs), 1e3)
        return s.y[0] - V_obs
    V0 = V_obs[0]
    best = None
    for w0 in (float(w_inf(V0)), 0.02, 0.1, 0.2, 0.35):
        r = least_squares(resid, [V0, w0], bounds=([-90.0, 0.0], [60.0, 1.0]), x_scale=[10.0, 0.1],
                          max_nfev=60, xtol=1e-10, ftol=1e-10)
        if best is None or r.cost < best.cost:
            best = r
    return best


def odelsq_run(scenario, rho, eta, rep, kind="random"):
    tag = tag_of(scenario, rho, eta, rep, "odelsq", kind)
    fn = os.path.join(OUT, tag + ".npz")
    if os.path.exists(fn):
        return fn
    Is = pinn_currents(); tg = load_ref(Is[0])[0]
    V = np.zeros((len(Is), len(tg))); W = np.zeros_like(V); val = []; th = []
    t0 = time.time()
    for j, I in enumerate(Is):
        o = observations(I, j, rho, eta, rep, kind)
        r = _fit_one(float(I), o)
        s = integrate(float(I), (0.0, 1000.0), r.x, "DOP853", 1e-10, (1e-10, 1e-12), t_eval=tg)
        V[j], W[j] = s.y
        idx = np.round(o["t_val"] / 0.1).astype(int)
        val.append(float(np.sqrt(np.mean((V[j, idx] - o["V_val"]) ** 2))) if len(idx) else np.nan)
        th.append(list(map(float, r.x)) + [float(r.cost)])
    np.savez_compressed(fn, I=Is, t=tg, V=V.astype(np.float32), w=W.astype(np.float32), val_rmse=np.array(val),
                        theta=np.array(th))
    json.dump(dict(tag=tag, variant="odelsq", wall_s=time.time() - t0), open(fn[:-4] + ".json", "w"))
    return fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="B"); ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=0.0); ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--method", default="spline")
    a = ap.parse_args()
    f = spline_run if a.method == "spline" else odelsq_run
    print(f(a.scenario, a.rho, a.eta, a.rep))
