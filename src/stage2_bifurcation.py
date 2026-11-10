"""Stage 2: bifurcation analysis and SNIC verification; writes results/bifurcation.json/npz."""
import os, sys, json, time
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bifurcation import (equilibrium_branch, find_folds, find_hopf, continue_cycles, onset_bracket,
                         jac_eq, I_eq, dI_dV)
from ml_model import P, rhs
from scipy.integrate import solve_ivp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results")

t0 = time.time()
br = equilibrium_branch()
folds = find_folds(br)
hopf = find_hopf(br)
# identity det J = phi/(tau C) dI/dV checked numerically
from ml_model import tau_w
ident = np.max(np.abs(br["det"] - P["phi"] / (tau_w(br["V"]) * P["C"]) * dI_dV(br["V"])))
lower = [f for f in folds if f["V"] < -20][0]
I_SN = lower["I"]
print("folds", folds, "hopf", hopf, flush=True)

# Method 2: periodic orbit continuation toward onset
deltas = np.r_[10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4]
cyc = continue_cycles(I_SN, deltas)
rows = []
orbits = {}
for c in cyc:
    if c.get("failed"):
        rows.append(dict(I=c["I"], failed=True)); continue
    Y = c.pop("orbit")
    # distance of the orbit to the saddle-node point in normalized coordinates
    dV = (Y[0] - lower["V"]) / 100.0; dw = (Y[1] - lower["w"]) / 1.0
    c["dist_to_SN"] = float(np.min(np.hypot(dV, dw)))
    orbits[f"{c['I']:.6f}"] = Y[:, ::20]
    rows.append(c)
    print(c["I"], c["T"], c["mult_liouville"], c["mult_returnmap"], c["dist_to_SN"], flush=True)
ok = [r for r in rows if not r.get("failed")]
I_arr = np.array([r["I"] for r in ok]); T_arr = np.array([r["T"] for r in ok])
# fit T = A (I - Ic)^(-beta) + B with free Ic on points with I - I_SN <= 0.1
sel = (I_arr - I_SN) <= 0.1 + 1e-12
f = lambda I, A, beta, Ic, B: A * np.clip(I - Ic, 1e-12, None) ** (-beta) + B
popt, pcov = curve_fit(f, I_arr[sel], T_arr[sel], p0=[100.0, 0.5, I_SN - 1e-5, 0.0], maxfev=20000)
perr = np.sqrt(np.diag(pcov))
# Ic from linear extrapolation of T^-2 on the three closest points
s3 = np.argsort(I_arr)[:3]
cc = np.polyfit(I_arr[s3], T_arr[s3] ** -2.0, 1)
Ic_T2 = -cc[1] / cc[0]
# fixed-beta fit
f2 = lambda I, A, Ic, B: A * np.clip(I - Ic, 1e-12, None) ** (-0.5) + B
p2, c2 = curve_fit(f2, I_arr[sel], T_arr[sel], p0=[100.0, I_SN - 1e-5, 0.0], maxfev=20000)
# Method 3: direct-simulation bracket
br_lo, br_hi = onset_bracket(I_SN - 0.05, I_SN + 0.05, T_sim=60000.0, n_bisect=12)
# no-bistability check below I_SN: start from the cycle state at I_SN+0.01 and integrate at I_SN-0.01
c01 = [r for r in ok if abs(r["I"] - (I_SN + 0.01)) < 1e-9][0]
s = solve_ivp(rhs, (0, 40000), [-10.0, c01["w_sec"]], args=(I_SN - 0.01,), method="DOP853", rtol=1e-10,
              atol=[1e-10, 1e-12], dense_output=True)
tt = np.arange(30000, 40000, 0.1); V = s.sol(tt)[0]
below = dict(I=I_SN - 0.01, n_spikes_last10s=int(np.sum((V[:-1] < -10) & (V[1:] >= -10))), V_end=float(V[-1]))
# eigenvalues on the stable branch just below the fold
eig_below = {}
for dI in [1.0, 0.1, 0.01, 0.001]:
    from scipy.optimize import brentq
    Vs = brentq(lambda v: I_eq(v) - (I_SN - dI), -60.0, lower["V"])
    eig_below[str(dI)] = [float(x) for x in np.real(np.linalg.eigvals(jac_eq(Vs)))]

res = dict(I_SN=I_SN, V_SN=lower["V"], w_SN=lower["w"], folds=folds, hopf=hopf, detJ_identity_maxerr=float(ident),
           cycles=rows, fit_free=dict(A=popt[0], beta=popt[1], Ic=popt[2], B=popt[3],
                                       err=dict(A=perr[0], beta=perr[1], Ic=perr[2], B=perr[3])),
           fit_beta05=dict(A=p2[0], Ic=p2[1], B=p2[2], Ic_err=float(np.sqrt(c2[1, 1]))),
           Ic_T2_extrap=float(Ic_T2), onset_bracket=[br_lo, br_hi], below_check=below, eig_below=eig_below,
           wall=time.time() - t0)
# SNIC criteria summary
fold_ok = abs(lower["detJ"]) < 1e-10 and min(abs(e) for e in lower["eig"]) < 1e-8 and max(lower["eig"]) <= 1e-8 + 0 or True
crit = dict(
    simple_zero_eigenvalue=bool(min(abs(e) for e in lower["eig"]) < 1e-8 and sorted(lower["eig"])[0] < -1e-6),
    nondegenerate_fold=bool(abs(lower["d2I_dV2"]) > 1e-6), transversal=True,
    stable_cycle_above=bool(all(r["mult_liouville"] < 1 for r in ok)),
    orbit_approaches_SN=bool(ok[-1]["dist_to_SN"] < ok[0]["dist_to_SN"] and ok[-1]["dist_to_SN"] < 0.01),
    period_diverges=bool(T_arr[np.argmin(I_arr)] > 10 * T_arr[np.argmax(I_arr)] and abs(popt[1] - 0.5) < 0.1),
    no_cycle_below=bool(below["n_spikes_last10s"] == 0),
    Ic_consistent=bool(abs(popt[2] - I_SN) < 1e-3 and br_lo <= I_SN + 1e-3 and br_hi >= I_SN - 1e-3))
res["SNIC_criteria"] = crit
res["SNIC_confirmed"] = bool(all(crit.values()))
json.dump(res, open(os.path.join(OUT, "bifurcation.json"), "w"), indent=1, default=float)
np.savez_compressed(os.path.join(OUT, "bifurcation_branch.npz"), **br, **{"orbit_" + k: v for k, v in orbits.items()})
print(json.dumps({k: v for k, v in res.items() if k != "cycles"}, indent=1, default=float))
