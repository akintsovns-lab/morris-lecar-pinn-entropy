"""Summarize the stored reference trajectories -> results/reference_summary.json"""
import os, sys, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from grids import pinn_currents  # noqa: E402

rows = []
for fn in sorted(glob.glob(os.path.join(ROOT, "results", "reference", "ref_I*.npz"))):
    d = np.load(fn)
    r = dict(I=float(d["I"]), status=str(d["status"]), converged=bool(d["converged"]), t_conv=float(d["t_conv"]),
             n_ext=int(d["n_ext"]), n_spikes_20s=int(d["n_spikes"]), period=float(d["period"]),
             Vmax=float(d["Vmax"]), Vmin=float(d["Vmin"]), wall=float(d["wall"]))
    if "V_dense" in d.files:
        V = d["V_dense"]; r["n_spikes_1000ms"] = int(np.sum((V[:-1] < -10) & (V[1:] >= -10)))
        r["sd_1000ms"] = float(np.std(V)); r["sigma_V"] = float(np.std(V))
    rows.append(r)
P = set(np.round(pinn_currents(), 4))
out = dict(n_currents=len(rows), n_eq=sum(r["status"] == "eq" or r["status"].startswith("eq") for r in rows),
           n_cyc=sum(r["status"] == "cycle" for r in rows), all_converged=all(r["converged"] for r in rows),
           max_t_conv=max(r["t_conv"] for r in rows), n_extended=sum(r["n_ext"] > 0 for r in rows),
           pinn=[r for r in rows if round(r["I"], 4) in P], rows=rows)
json.dump(out, open(os.path.join(ROOT, "results", "reference_summary.json"), "w"), indent=0)
print({k: v for k, v in out.items() if not isinstance(v, list)})
print([(r["I"], r["n_spikes_1000ms"]) for r in out["pinn"]])
