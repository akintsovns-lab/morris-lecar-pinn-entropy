"""Derived quantities quoted in the manuscript -> results/manuscript_numbers.json.

Only values computed here from other result files may appear in the text (audit.py checks this).
"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)


def J(p):
    return json.load(open(os.path.join(RES, p)))


def main():
    X = {}
    VER = J("stage1_verification.json"); BIF = J("bifurcation.json"); REF = J("reference_summary.json")
    BS = J("benchmark/benchmark_summary.json")
    # --- stage 1
    pu = [v["dt_out"]["0.1"]["peak_underestimate"] for k, v in VER.items() if isinstance(v, dict) and "dt_out" in v]
    X["peak_under_max"] = float(max(pu))
    rows = {round(r["I"], 4): r for r in REF["rows"]}
    X["period_40"] = rows[40.0]["period"]; X["period_40p1"] = rows[40.1]["period"]; X["period_50"] = rows[50.0]["period"]
    X["spikes_1000"] = {f"{r['I']:.2f}": r["n_spikes_1000ms"] for r in REF["pinn"]}
    X["max_spikes_1000"] = max(r["n_spikes_1000ms"] for r in REF["pinn"])
    # --- stage 2
    cyc = sorted(BIF["cycles"], key=lambda c: c["I"])
    X["T_closest"] = cyc[0]["T"]; X["dI_closest"] = cyc[0]["I"] - BIF["I_SN"]
    X["T_farthest"] = cyc[-1]["T"]; X["dI_farthest"] = cyc[-1]["I"] - BIF["I_SN"]
    X["onset_offset"] = [BIF["onset_bracket"][0] - BIF["I_SN"], BIF["onset_bracket"][1] - BIF["I_SN"]]
    X["Ic_diff_free"] = abs(BIF["fit_free"]["Ic"] - BIF["I_SN"])
    X["Ic_diff_b05"] = abs(BIF["fit_beta05"]["Ic"] - BIF["I_SN"])
    X["Ic_diff_T2"] = abs(BIF["Ic_T2_extrap"] - BIF["I_SN"])
    X["beta_dev_sigma"] = abs(BIF["fit_free"]["beta"] - 0.5) / BIF["fit_free"]["err"]["beta"]
    # --- reference entropy curves on the 34-current grid
    R = BS["reference"]; I34 = np.array(R["I_34"])
    above = I34 > BIF["I_SN"]
    X["ref34"] = {}
    for m, H in R["H_ref_34"].items():
        H = np.array(H)
        X["ref34"][m] = dict(first_above=float(H[above][0]), max_above=float(H[above].max()),
                             min_above=float(H[above].min()), at50=float(H[-1]), below_max=float(H[~above].max()))
    X["best_eps_34"] = float(abs(0.5 * (39.95 + 40.0) - BIF["I_SN"]))
    S = R["sensitivity"]; X["sens"] = {}
    for m in ["SampEn", "PE", "WPE", "SpEn"]:
        e = [v["eps"] for k, v in S.items() if k.startswith(m + "|")]
        e34 = [v["eps_34"] for k, v in S.items() if k.startswith(m + "|")]
        X["sens"][m] = dict(n=len(e), min=min(e), max=max(e), min34=min(e34), max34=max(e34),
                            n34_worse=int(sum(x > X["best_eps_34"] + 1e-9 for x in e34)))
    X["floquet_far"] = cyc[-1]["mult_liouville"]
    X.update(recon_numbers(BIF["I_SN"], I34))
    # --- noise scale in mV, fraction of realizations at the best achievable error, run statistics
    from train import sigma_V_train
    X["sigma_V_train"] = sigma_V_train()
    X["noise_sd_mV"] = {str(e): e * X["sigma_V_train"] for e in (0.01, 0.05, 0.1)}
    RL = J("benchmark/realizations.json")
    fb = {}
    for r in RL:
        key = f"{r['scenario']}|{r['rho']}|{r['eta']}|{r['variant']}"
        d = fb.setdefault(key, {m: [] for m in ["SampEn", "PE", "WPE", "SpEn"]})
        for m in d:
            ref = R["ref_34_1000ms"][m]["eps"]
            d[m].append(float(abs(r[m + "_eps"] - ref) < 1e-9))
    X["frac_ref_eps"] = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in fb.items()}
    pin = [r for r in RL if r["variant"] == "pinn"]
    X["n_pinn_runs"] = len(pin)
    X["n_pinn_runs_A"] = len([r for r in pin if r["scenario"] == "A"])
    noisefree = [k for k in X["frac_ref_eps"] if k.split("|")[2] == "0.0"]
    X["all_noisefree_reproduce_ref"] = bool(all(min(X["frac_ref_eps"][k].values()) == 1.0 for k in noisefree))
    X["min_frac_ref_noisefree"] = float(min(min(X["frac_ref_eps"][k].values()) for k in noisefree))
    import glob
    walls = [json.load(open(fn))["wall_s"] for fn in glob.glob(os.path.join(RES, "pinn", "[BC]_*_pinn.json"))]
    X["pinn_wall_median_s"] = float(np.median(walls)); X["n_pinn_BC"] = len(walls)
    json.dump(X, open(os.path.join(RES, "manuscript_numbers.json"), "w"), indent=1)
    print(json.dumps(X)[:1500])


def recon_numbers(Ic, I34):
    """Derived numbers for Results 3.3-3.5 (quiescent-window SD, location of w errors)."""
    import glob, sys
    sys.path.insert(0, HERE)
    out = {"quiet": {}, "w_err_spike_frac": {}}
    below = I34 < Ic - 0.005
    for fn in sorted(glob.glob(os.path.join(RES, "entropy", "rec", "*.json"))):
        tag = os.path.basename(fn)[:-5]; p = tag.split("_")
        key = "|".join([p[0], str(float(p[1][3:])), str(float(p[2][3:])), p[4]])
        e = json.load(open(fn))
        sd = np.array([e[f"{I:.4f}"]["sd"] for I in I34[below]])
        q = out["quiet"].setdefault(key, dict(sd_med=[], frac_nonquiet=[]))
        q["sd_med"].append(float(np.median(sd))); q["frac_nonquiet"].append(float(np.mean(sd >= 0.1)))
    for k, q in out["quiet"].items():
        out["quiet"][k] = dict(sd_median=float(np.median(q["sd_med"])), sd_min=float(np.min(q["sd_med"])),
                               sd_max=float(np.max(q["sd_med"])), frac_nonquiet_median=float(np.median(q["frac_nonquiet"])),
                               n=len(q["sd_med"]))
    # fraction of the squared w error inside +-10 ms of reference spikes (PINN, Scenario A and B rho=0.05)
    from train import load_ref
    from metrics import crossings
    for tag in ("A_rho1.00_eta0.00_rep00_pinn_init0", "B_rho0.05_eta0.00_rep00_pinn"):
        f = os.path.join(RES, "pinn", tag + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); num = den = 0.0; tfrac = []
        for j, I in enumerate(z["I"]):
            if I < Ic:
                continue
            t, V, w = load_ref(float(I)); e2 = (z["w"][j] - w) ** 2
            near = np.zeros(len(t), bool)
            for c in crossings(t, V):
                near |= np.abs(t - c) <= 10.0
            num += e2[near].sum(); den += e2.sum(); tfrac.append(near.mean())
        out["w_err_spike_frac"][tag] = dict(err_frac=float(num / den), time_frac_median=float(np.median(tfrac)))
    return out


if __name__ == "__main__":
    main()
