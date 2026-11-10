"""EXPLORATORY (post hoc, D-23): sensitivity of the localization error and AUC to the quiescent SD floor.

The pre-registered floor is 0.1 mV (D-08). This script recomputes the entropies of reconstructions with floors
0.01 mV and 1.0 mV and reports the median eps and AUC per condition. It does not enter the hypothesis tests.
-> results/benchmark/explore_floor.json
"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from entropy import all_entropies, MEASURES  # noqa: E402
from benchmark import detect_derivative, auc_sep, IC  # noqa: E402
from grids import pinn_currents  # noqa: E402

FLOORS = (0.01, 0.1, 1.0)
CONDS = [("B", 0.05, 0.0), ("C", 0.05, 0.10), ("C", 0.20, 0.10), ("C", 0.05, 0.01)]
VARIANTS = ("pinn", "spline", "odelsq", "dataonly")


def curves(fn, floor):
    d = np.load(fn); P = pinn_currents(); H = {k: [] for k in MEASURES}
    for j, I in enumerate(P):
        k = int(np.argmin(np.abs(d["I"] - I)))
        e = all_entropies(d["V"][k].astype(np.float64)[::10], dict(sd_floor=floor))
        for m in MEASURES:
            H[m].append(e[m])
    return P, {m: np.array(v) for m, v in H.items()}


def main():
    out = dict(note="exploratory, post hoc; floors in mV; pre-registered floor 0.1", floors=list(FLOORS), conditions={})
    for sc, rho, eta in CONDS:
        for v in VARIANTS:
            fs = sorted(glob.glob(os.path.join(ROOT, "results", "pinn", f"{sc}_rho{rho:.2f}_eta{eta:.2f}_rep*_{v}.npz")))
            fs = [f for f in fs if os.path.exists(f[:-4] + ".json")]
            if not fs:
                continue
            key = f"{sc}|{rho}|{eta}|{v}"; out["conditions"][key] = dict(n=len(fs))
            for fl in FLOORS:
                E = {m: [] for m in MEASURES}; A = {m: [] for m in MEASURES}
                for fn in fs:
                    P, H = curves(fn, fl)
                    for m in MEASURES:
                        E[m].append(abs(detect_derivative(P, H[m])[0] - IC)); A[m].append(auc_sep(P, H[m]))
                out["conditions"][key][str(fl)] = {m: dict(eps_median=float(np.median(E[m])),
                                                           auc_median=float(np.nanmedian(A[m]))) for m in MEASURES}
            print(key, json.dumps(out["conditions"][key])[:300], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "benchmark", "explore_floor.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
