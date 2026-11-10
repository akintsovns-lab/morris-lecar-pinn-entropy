"""Spline baseline for every (scenario, rho, eta, rep) of the PINN queue; ODE-LSQ for a subset."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import yaml
from baselines import spline_run, odelsq_run
ROOT = os.path.abspath(os.path.join(HERE, ".."))
S = yaml.safe_load(open(os.path.join(ROOT, "configs", "sparse.yaml")))

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "spline"
    if which == "spline":
        spline_run("A", 1.0, 0.0, 0)
        for rep in range(S["n_masks"]):
            for rho in S["rho_levels"]:
                spline_run("B", rho, 0.0, rep)
            for eta in S["eta_levels"]:
                for rho in S["eta_rho_levels"]:
                    spline_run("C", rho, eta, rep)
    else:
        for rep in range(5):   # subset (D-16): 5 masks per condition
            for sc, rho, eta in (("B", 0.05, 0.0), ("C", 0.05, 0.10)):
                t0 = time.time(); fn = odelsq_run(sc, rho, eta, rep); print(fn, time.time() - t0, flush=True)
    print("done")
