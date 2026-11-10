"""Stage 1 verification: tolerance convergence, method cross-check, output-step sensitivity,
entropy sensitivity to integration tolerance, and full-run reproducibility from the initial state."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import integrate, reference_trajectory, load_cfg, _ev_section, _ev_peak
from entropy import all_entropies, MEASURES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CURRENTS = [35.0, 39.9, 40.0, 40.3, 45.0, 50.0]
TOLS = [(1e-8, (1e-8, 1e-10)), (1e-10, (1e-10, 1e-12)), (1e-12, (1e-12, 1e-14))]
T = 1000.0


def spikes(sol):
    return np.asarray(sol.t_events[0])


def main():
    cfg = load_cfg()
    out = {}
    evs = [_ev_section(cfg["poincare_V"]), _ev_peak]
    for I in CURRENTS:
        d = np.load(os.path.join(ROOT, "results", "reference", f"ref_I{I:08.4f}.npz"))
        ys = d["y_start"]
        te = np.round(np.arange(0, T + 1e-9, 0.1), 10)
        rec = dict(tol={}, method={}, dt_out={})
        base = None
        sols = {}
        for rt, at in TOLS:
            t0 = time.time()
            s = integrate(I, (0, T), ys, "DOP853", rt, at, t_eval=te, events=evs)
            sols[rt] = s
            H = all_entropies(s.y[0][::10])
            rec["tol"][f"{rt:.0e}"] = dict(n_spikes=int(len(spikes(s))), wall=time.time() - t0, H=H,
                                           Vmax=float(s.y[0].max()))
        ref = sols[1e-12]
        for rt in (1e-8, 1e-10):
            s = sols[rt]
            r = rec["tol"][f"{rt:.0e}"]
            r["max_dV_vs_1e-12"] = float(np.max(np.abs(s.y[0] - ref.y[0])))
            r["max_dw_vs_1e-12"] = float(np.max(np.abs(s.y[1] - ref.y[1])))
            r["max_dH_vs_1e-12"] = float(max(abs(r["H"][k] - rec["tol"]["1e-12"]["H"][k]) for k in MEASURES))
            ts, tr = spikes(s), spikes(ref)
            r["max_dspike_ms"] = float(np.max(np.abs(ts - tr))) if len(ts) == len(tr) and len(ts) else 0.0
        for m in ("Radau", "LSODA"):
            t0 = time.time()
            s = integrate(I, (0, T), ys, m, 1e-10, (1e-10, 1e-12), t_eval=te, events=evs)
            Hm = all_entropies(s.y[0][::10])
            rec["method"][m] = dict(max_dV_vs_DOP853=float(np.max(np.abs(s.y[0] - ref.y[0]))),
                                    max_dw=float(np.max(np.abs(s.y[1] - ref.y[1]))), n_spikes=int(len(spikes(s))),
                                    max_dH=float(max(abs(Hm[k] - rec["tol"]["1e-12"]["H"][k]) for k in MEASURES)),
                                    wall=time.time() - t0)
        # output-step sensitivity of extrema (peak from event = exact local maximum)
        pk = np.array([y[0] for y in ref.y_events[1] if y[0] > cfg["poincare_V"]])
        for dt in (0.1, 0.05, 0.025):
            tg = np.round(np.arange(0, T + 1e-9, dt), 10)
            s = integrate(I, (0, T), ys, "DOP853", 1e-10, (1e-10, 1e-12), t_eval=tg)
            rec["dt_out"][str(dt)] = dict(Vmax_grid=float(s.y[0].max()), Vmin_grid=float(s.y[0].min()),
                                          Vmax_event=float(pk.max()) if len(pk) else None,
                                          peak_underestimate=float(pk.max() - s.y[0].max()) if len(pk) else 0.0)
        rec["n_spikes_1000ms"] = int(len(spikes(ref)))
        out[f"{I:.4f}"] = rec
        print(I, json.dumps(rec["tol"]["1e-08"].get("max_dV_vs_1e-12")), flush=True)
    # full run reproducibility from initial state with tighter tolerance (I = 40.3)
    t0 = time.time()
    r10 = np.load(os.path.join(ROOT, "results", "reference", "ref_I040.3000.npz"))
    r12 = reference_trajectory(40.3, T_obs=1000.0, keep_dense=1000.0, rtol=1e-12, atol=(1e-12, 1e-14))
    # compare phase-invariant quantities: period, peak, entropy of the dense window
    out["full_run_I40.3"] = dict(period_tol10=float(r10["period"]), period_tol12=float(r12["period"]),
                                 Vmax_tol10=float(r10["Vmax"]), Vmax_tol12=float(r12["Vmax"]),
                                 t_conv_tol10=float(r10["t_conv"]), t_conv_tol12=float(r12["t_conv"]),
                                 y_start_tol10=r10["y_start"].tolist(), y_start_tol12=r12["y_start"].tolist(),
                                 wall=time.time() - t0)
    keys = [k for k in out if k[0].isdigit()]
    crit = {
        "max_dV_1e-10_vs_1e-12": max(out[k]["tol"]["1e-10"]["max_dV_vs_1e-12"] for k in keys),
        "max_dH_1e-10_vs_1e-12": max(out[k]["tol"]["1e-10"]["max_dH_vs_1e-12"] for k in keys),
        "max_dV_Radau": max(out[k]["method"]["Radau"]["max_dV_vs_DOP853"] for k in keys),
        "max_dV_LSODA": max(out[k]["method"]["LSODA"]["max_dV_vs_DOP853"] for k in keys),
        "max_dH_methods": max(max(out[k]["method"][m]["max_dH"] for m in ("Radau", "LSODA")) for k in keys),
        "spike_counts_agree": all(len({out[k]["tol"][t]["n_spikes"] for t in out[k]["tol"]} |
                                      {out[k]["method"][m]["n_spikes"] for m in out[k]["method"]}) == 1 for k in keys),
    }
    crit["entropy_tol_pass"] = bool(crit["max_dH_1e-10_vs_1e-12"] < 1e-3)
    out["criteria"] = crit
    json.dump(out, open(os.path.join(ROOT, "results", "stage1_verification.json"), "w"), indent=1)
    print(json.dumps(crit, indent=1))


if __name__ == "__main__":
    main()
