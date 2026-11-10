"""Sparse-data protocol: current-level split, observation masks, noise, blocked validation."""
from __future__ import annotations
import os, sys, json
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grids import pinn_currents

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def cfg():
    with open(os.path.join(ROOT, "configs", "sparse.yaml")) as f:
        return yaml.safe_load(f)


def current_split():
    I = pinn_currents()
    c = cfg()["split_rule"]
    idx = np.arange(len(I))
    test = idx % c["test_mod"][0] == c["test_mod"][1]
    val = (idx % c["val_mod"][0] == c["val_mod"][1]) & ~test
    train = ~(test | val)
    return dict(train=I[train], val=I[val], test=I[test], all=I)


def make_mask(n, rho, seed, kind="random"):
    """Binary mask on n grid points retaining round(rho*n) points; first and last always retained."""
    m = np.zeros(n, dtype=bool)
    if rho >= 1.0:
        m[:] = True
        return m
    if kind == "regular":
        q = int(round(1.0 / rho))
        m[::q] = True
    else:
        rng = np.random.default_rng(seed)
        k = int(round(rho * n)) - 2
        inner = rng.choice(np.arange(1, n - 1), size=k, replace=False)
        m[inner] = True
    m[0] = m[-1] = True
    return m


def val_blocks(t, block=5.0, period=100.0, offset=47.0):
    """Blocked temporal validation: points in [offset + k*period, offset + k*period + block)."""
    ph = (t - offset) % period
    return (ph >= 0) & (ph < block) & (t >= offset)


def gap_stats(t_obs):
    g = np.diff(t_obs)
    return dict(mean_gap=float(g.mean()), median_gap=float(np.median(g)), max_gap=float(g.max()))


def snr_db(v_ref, v_obs):
    return float(10 * np.log10(np.sum((v_ref - v_ref.mean()) ** 2) / np.sum((v_obs - v_ref) ** 2)))
