"""Current grids (single source of truth)."""
import os
import numpy as np, yaml
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _y(name):
    with open(os.path.join(ROOT, "configs", name)) as f:
        return yaml.safe_load(f)


def pinn_currents():
    b = _y("benchmark.yaml")
    f = b["pinn_currents_fine"]
    fine = np.round(np.arange(f["start"], f["stop"] + 1e-9, f["step"]), 4)
    return np.unique(np.round(np.r_[b["pinn_currents_coarse"], fine], 4))


def reference_currents():
    m = _y("model.yaml")
    a, b = m["I_range"]
    coarse = np.round(np.arange(a, b + 1e-9, m["coarse_step"]), 4)
    fa, fb = m["fine_range"]
    fine = np.round(np.arange(fa, fb + 1e-9, m["fine_step"]), 4)
    return np.unique(np.round(np.r_[coarse, fine, pinn_currents()], 4))
