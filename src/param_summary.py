"""Summary of the parametric (t, I) -> (V, w) generalization experiment -> results/param/param_summary.json"""
import os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
D = os.path.join(ROOT, "results", "param")
Q = ("relL2_V", "relL2_w", "rmse_w", "peak_time_err", "period_relerr", "frac_spikes_matched", "spectral_err")


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def main():
    fs = sorted(glob.glob(os.path.join(D, "param_seed*.json")))
    out = dict(n_seeds=len(fs), seeds=[], by_role={}, by_current={})
    if not fs:
        json.dump(out, open(os.path.join(D, "param_summary.json"), "w"), indent=1)
        print("no parametric runs"); return
    R = [json.load(open(f)) for f in fs]
    out["seeds"] = [r["seed"] for r in R]
    out["wall_s_median"] = med([r["wall_s"] for r in R])
    roles = sorted({v["role"] for r in R for v in r["per_current"].values()})
    for role in roles:
        d = {}
        for q in Q:
            per_seed = [med([v.get(q, np.nan) for v in r["per_current"].values() if v["role"] == role]) for r in R]
            d[q + "_median"] = med(per_seed)
            d[q + "_range"] = [float(np.nanmin(per_seed)), float(np.nanmax(per_seed))] if np.isfinite(per_seed).any() else [float("nan")] * 2
        sp_w = [med([v["relL2_w"] for v in r["per_current"].values() if v["role"] == role and v["n_spikes_ref"] > 0]) for r in R]
        d["relL2_w_spiking_median"] = med(sp_w)
        d["relL2_w_spiking_range"] = [float(np.nanmin(sp_w)), float(np.nanmax(sp_w))] if np.isfinite(sp_w).any() else [float("nan")] * 2
        sp_v = [med([v["relL2_V"] for v in r["per_current"].values() if v["role"] == role and v["n_spikes_ref"] > 0]) for r in R]
        d["relL2_V_spiking_median"] = med(sp_v)
        d["n_currents"] = int(sum(v["role"] == role for v in R[0]["per_current"].values()))
        fsp = [abs(v["first_spike_hat"] - v["first_spike_ref"]) for r in R for v in r["per_current"].values()
               if v["role"] == role and np.isfinite(v.get("first_spike_ref", np.nan))]
        d["first_spike_abs_err_median"] = med(fsp)
        miss = [1 for r in R for v in r["per_current"].values()
                if v["role"] == role and np.isfinite(v.get("first_spike_ref", np.nan)) and not np.isfinite(v.get("first_spike_hat", np.nan))]
        d["n_missed_first_spike"] = int(len(miss))
        d["n_spiking_cases"] = int(len(fsp) + len(miss))
        out["by_role"][role] = d
    for I in R[0]["per_current"]:
        out["by_current"][I] = dict(role=R[0]["per_current"][I]["role"],
                                    **{q + "_median": med([r["per_current"][I].get(q, np.nan) for r in R]) for q in ("relL2_V", "relL2_w")})
    json.dump(out, open(os.path.join(D, "param_summary.json"), "w"), indent=1)
    print(json.dumps(out["by_role"], indent=1))


if __name__ == "__main__":
    main()
