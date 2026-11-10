"""Stage 6: transition-detection benchmark, statistics and hypothesis evaluation -> results/benchmark/"""
import os, sys, json, glob, itertools
import numpy as np, yaml
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from benchmark import detect_derivative, detect_changepoint, auc_sep, holm, paired_wilcoxon, kendall_rank, IC, BCFG  # noqa
from entropy import MEASURES  # noqa
from grids import pinn_currents  # noqa
from sparse import current_split  # noqa

OUT = os.path.join(ROOT, "results", "benchmark"); os.makedirs(OUT, exist_ok=True)
S = yaml.safe_load(open(os.path.join(ROOT, "configs", "sparse.yaml")))
M = list(MEASURES)
RNG = np.random.default_rng(31415)


def boot_median_ci(x, n=2000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan")] * 2
    b = [np.median(RNG.choice(x, len(x))) for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def curve_stats(I, H):
    Is, slope = detect_derivative(I, H)
    cp = detect_changepoint(I, H)
    return dict(I_star=Is, eps=abs(Is - IC), eps_norm=abs(Is - IC) / IC, slope=slope, I_cp=cp, eps_cp=abs(cp - IC),
                auc=auc_sep(I, H))


def reference_block():
    r = json.load(open(os.path.join(ROOT, "results", "entropy", "reference_entropy.json")))
    I = np.array(r["I"]); P = pinn_currents()
    idx = [int(np.argmin(np.abs(I - p))) for p in P]
    out = {}
    for key, lab in (("short", "ref_292_1000ms"), ("long_mean", "ref_292_long"), ("full", "ref_292_full20s"),
                     ("short500", "ref_292_500ms")):
        out[lab] = {k: curve_stats(I, np.array(r[key][k])) for k in M}
    out["ref_34_1000ms"] = {k: curve_stats(P, np.array(r["short"][k])[idx]) for k in M}
    out["ref_34_long"] = {k: curve_stats(P, np.array(r["long_mean"][k])[idx]) for k in M}
    # sensitivity to entropy parameters (292 grid, 1000 ms)
    sens = {}
    for key, vals in r["sens"].items():
        k, p = key.split("|")
        sens[key] = dict(measure=k, params=json.loads(p), **curve_stats(I, np.array(vals)),
                         **{"eps_34": curve_stats(P, np.array(vals)[idx])["eps"]})
    out["sensitivity"] = sens
    out["H_ref_34"] = {k: np.array(r["short"][k])[idx].tolist() for k in M}
    out["I_34"] = P.tolist()
    # bandwidth sensitivity of the detection rule itself (reported, not used for selection)
    bw = {}
    for h in (0.02, 0.05, 0.10):
        bw[str(h)] = {k: abs(detect_derivative(P, np.array(r["short"][k])[idx], h=h)[0] - IC) for k in M}
    out["bandwidth_sensitivity_34"] = bw
    return out


def parse_tag(tag):
    p = tag.split("_")
    d = dict(scenario=p[0], rho=float(p[1][3:]), eta=float(p[2][3:]), rep=int(p[3][3:]), variant=p[4])
    d["init"] = int(p[5][4:]) if len(p) > 5 and p[5].startswith("init") else None
    return d


def realization_table(ref):
    Hr = {k: np.array(ref["H_ref_34"][k]) for k in M}
    P = np.array(ref["I_34"])
    split = current_split()
    test = np.isin(np.round(P, 4), np.round(split["test"], 4))
    keep = np.abs(P - IC) >= BCFG["I_crit_exclusion"]
    rows = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "results", "entropy", "rec", "*.json"))):
        tag = os.path.basename(fn)[:-5]
        e = json.load(open(fn)); d = parse_tag(tag)
        H = {k: np.array([e[f"{I:.4f}"][k] for I in P]) for k in M}
        row = dict(tag=tag, **d)
        for k in M:
            cs = curve_stats(P, H[k])
            row.update({f"{k}_{a}": b for a, b in cs.items()})
            row[f"{k}_bias"] = float(np.median(np.abs(H[k] - Hr[k])[keep]))
            row[f"{k}_bias_test"] = float(np.median(np.abs(H[k] - Hr[k])[test & keep]))
        mfn = os.path.join(ROOT, "results", "metrics", tag + ".json")
        if os.path.exists(mfn):
            mm = json.load(open(mfn))
            for q in ("relL2_V", "relL2_w", "rmse_V", "nrmse_V", "peak_time_err", "period_relerr", "spectral_err",
                      "amp_err", "frac_spikes_matched", "resV_rms", "resw_rms", "val_rmse"):
                vals = np.array([mm[f"{I:.4f}"].get(q, np.nan) for I in P], float)
                row[f"{q}_med_all"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")
                row[f"{q}_med_test"] = float(np.nanmedian(vals[test])) if np.isfinite(vals[test]).any() else float("nan")
                row[f"{q}_max_all"] = float(np.nanmax(vals)) if np.isfinite(vals).any() else float("nan")
        meta = os.path.join(ROOT, "results", "pinn", tag + ".json")
        if os.path.exists(meta):
            row["wall_s"] = json.load(open(meta)).get("wall_s", float("nan"))
        rows.append(row)
    return rows


def cond_key(r):
    return (r["scenario"], r["rho"], r["eta"], r["variant"])


def aggregate(rows):
    C = {}
    for r in rows:
        C.setdefault(cond_key(r), []).append(r)
    agg = {}
    for key, rs in sorted(C.items()):
        a = dict(scenario=key[0], rho=key[1], eta=key[2], variant=key[3], n=len(rs))
        for k in M:
            e = [r[f"{k}_eps"] for r in rs]
            a[f"{k}_eps_median"] = float(np.median(e)); a[f"{k}_eps_ci"] = boot_median_ci(e)
            a[f"{k}_eps_iqr"] = [float(np.percentile(e, 25)), float(np.percentile(e, 75))]
            a[f"{k}_eps_max"] = float(np.max(e))
            a[f"{k}_eps_cp_median"] = float(np.median([r[f"{k}_eps_cp"] for r in rs]))
            a[f"{k}_auc_median"] = float(np.median([r[f"{k}_auc"] for r in rs]))
            a[f"{k}_bias_median"] = float(np.median([r[f"{k}_bias"] for r in rs]))
            a[f"{k}_Istar_values"] = sorted(set(round(r[f"{k}_I_star"], 4) for r in rs))
        for q in ("relL2_V_med_test", "relL2_w_med_test", "relL2_V_med_all", "relL2_w_med_all", "relL2_V_max_all",
                  "relL2_w_max_all", "peak_time_err_med_all", "period_relerr_med_all", "spectral_err_med_all",
                  "frac_spikes_matched_med_all", "resV_rms_med_all", "val_rmse_med_all", "nrmse_V_med_all", "wall_s"):
            v = np.array([r.get(q, np.nan) for r in rs], float)
            a[q] = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
            a[q + "_iqr"] = ([float(np.nanpercentile(v, 25)), float(np.nanpercentile(v, 75))]
                             if np.isfinite(v).any() else [float("nan")] * 2)
        agg["|".join(map(str, key))] = a
    return agg, C


def hypotheses(ref, rows, C):
    out = {}
    R = ref["ref_34_1000ms"]
    pinnB = [("B", r, 0.0, "pinn") for r in S["rho_levels"]]
    pinnC = [("C", r, e, "pinn") for e in S["eta_levels"] for r in S["eta_rho_levels"]]
    conds = [c for c in pinnB + pinnC if c in C]
    # H1
    pv, info = [], []
    for c in conds:
        rs = sorted(C[c], key=lambda r: r["rep"])
        w = [r["WPE_eps"] for r in rs]; p_ = [r["PE_eps"] for r in rs]
        p, md = paired_wilcoxon(w, p_)
        pv.append(p); info.append(dict(cond=list(c), n=len(rs), med_WPE=float(np.median(w)), med_PE=float(np.median(p_)),
                                       p=p, median_diff=md))
    adj = holm(pv) if pv else []
    for i, a in zip(info, adj):
        i["p_holm"] = float(a)
    wb = sum(1 for i in info if i["med_WPE"] < i["med_PE"] and i["p_holm"] < 0.05)
    pb = sum(1 for i in info if i["med_PE"] < i["med_WPE"] and i["p_holm"] < 0.05)
    n = len(info)
    ref_ok = R["WPE"]["eps"] <= R["PE"]["eps"]
    verdict = ("CONFIRMED" if (ref_ok and n and wb >= 0.5 * n and pb == 0) else
               "REFUTED" if (n and pb >= 0.5 * n) else "INCONCLUSIVE")
    out["H1"] = dict(reference_eps_WPE=R["WPE"]["eps"], reference_eps_PE=R["PE"]["eps"], reference_ok=bool(ref_ok),
                     conditions=info, n_WPE_better=wb, n_PE_better=pb, n_conditions=n, verdict=verdict)
    # H2
    D = {}
    for rho in S["eta_rho_levels"]:
        c0, c1 = ("B", rho, 0.0, "pinn"), ("C", rho, max(S["eta_levels"]), "pinn")
        if c0 in C and c1 in C:
            D[str(rho)] = {k: float(np.median([r[f"{k}_eps"] for r in C[c1]]) - np.median([r[f"{k}_eps"] for r in C[c0]]))
                           for k in M}
    smallest = {rho: [k for k in M if v[k] == min(v.values())] for rho, v in D.items()}
    sp_small = [rho for rho, ks in smallest.items() if ks == ["SpEn"]]
    other_strict = [rho for rho, v in D.items() if any(v[k] < v["SpEn"] for k in M if k != "SpEn")]
    tests = []
    for rho in sp_small:
        rs = sorted(C[("C", float(rho), max(S["eta_levels"]), "pinn")], key=lambda r: r["rep"])
        for k in M:
            if k != "SpEn":
                p, md = paired_wilcoxon([r["SpEn_eps"] for r in rs], [r[f"{k}_eps"] for r in rs])
                tests.append(dict(rho=rho, other=k, p=p, median_diff_SpEn_minus_other=md))
    if tests:
        for t, a in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = float(a)
    worse = any(t["p_holm"] < 0.05 and t["median_diff_SpEn_minus_other"] > 0 for t in tests)
    nr = len(D)
    v2 = ("CONFIRMED" if nr and len(sp_small) >= 0.5 * nr and not worse else
          "REFUTED" if nr and len(other_strict) >= 0.5 * nr else "INCONCLUSIVE")
    out["H2"] = dict(Delta=D, smallest=smallest, tests=tests, verdict=v2)
    # H3
    ref_eps = [R[k]["eps"] for k in M]
    taus = {}
    for c in [("A", 1.0, 0.0, "pinn")] + pinnB + pinnC:
        if c in C:
            med = [float(np.median([r[f"{k}_eps"] for r in C[c]])) for k in M]
            taus["|".join(map(str, c))] = dict(median_eps=med, tau=kendall_rank(ref_eps, med))
    A_ok = taus.get("A|1.0|0.0|pinn", {}).get("tau", np.nan) >= 0.67
    pres = {r: taus.get(f"B|{r}|0.0|pinn", {}).get("tau", np.nan) >= 0.67 for r in S["rho_levels"]}
    rho_th = None
    for r in sorted(S["rho_levels"]):
        if all(pres[q] for q in S["rho_levels"] if q >= r):
            rho_th = r; break
    v3 = "REFUTED" if not A_ok else ("CONFIRMED" if rho_th is not None else "INCONCLUSIVE")
    out["H3"] = dict(reference_eps=dict(zip(M, ref_eps)), taus=taus, A_preserved=bool(A_ok), preserved_B=pres,
                     rho_th=rho_th, verdict=v3,
                     note="tau-b with ties; if the reference ranking is fully tied tau is undefined (nan)")
    # H4
    low = [c for c in conds if c[1] <= 0.10]
    t4 = []
    for c in low:
        cs = (c[0], c[1], c[2], "spline")
        if cs not in C:
            continue
        P = {r["rep"]: r for r in C[c]}; Q = {r["rep"]: r for r in C[cs]}
        reps = sorted(set(P) & set(Q))
        x = [P[i][f"{k}_eps"] for i in reps for k in M]; y = [Q[i][f"{k}_eps"] for i in reps for k in M]
        p, md = paired_wilcoxon(x, y)
        t4.append(dict(cond=list(c), n_pairs=len(x), med_pinn=float(np.median(x)), med_spline=float(np.median(y)), p=p,
                       median_diff=md))
    if t4:
        for t, a in zip(t4, holm([t["p"] for t in t4])):
            t["p_holm"] = float(a)
    pb4 = sum(1 for t in t4 if t["med_pinn"] < t["med_spline"] and t["p_holm"] < 0.05)
    sb4 = sum(1 for t in t4 if t["med_spline"] < t["med_pinn"] and t["p_holm"] < 0.05)
    v4 = ("CONFIRMED" if t4 and pb4 >= 0.5 * len(t4) else "REFUTED" if t4 and sb4 >= 0.5 * len(t4) else "INCONCLUSIVE")
    out["H4"] = dict(tests=t4, n_pinn_better=pb4, n_spline_better=sb4, verdict=v4)
    # H5
    def w_med(c):
        return float(np.median([r["relL2_w_med_test"] for r in C[c]])) if c in C else float("nan")
    A = w_med(("A", 1.0, 0.0, "pinn")); Bv = {r: w_med(("B", r, 0.0, "pinn")) for r in S["rho_levels"]}
    okB = all(Bv[r] < 0.10 for r in S["rho_levels"] if r >= 0.10)
    v5 = "REFUTED" if A > 0.10 else ("CONFIRMED" if (A < 0.10 and okB) else "INCONCLUSIVE")
    out["H5"] = dict(A_median_relL2_w_test=A, B_median_relL2_w_test=Bv, verdict=v5)
    return out


def main():
    ref = reference_block()
    rows = realization_table(ref)
    agg, C = aggregate(rows)
    hyp = hypotheses(ref, rows, C)
    sp = current_split()
    extra = dict(I_crit=IC, snr_db={str(e): float(-20 * np.log10(e)) for e in S["eta_levels"]},
                 split={k: [float(x) for x in v] for k, v in sp.items() if k != "all"},
                 n_realizations=len(rows))
    json.dump(dict(reference=ref, conditions=agg, hypotheses=hyp, extra=extra),
              open(os.path.join(OUT, "benchmark_summary.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(OUT, "realizations.json"), "w"), indent=0)
    import csv
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(os.path.join(OUT, "realizations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(json.dumps({h: hyp[h]["verdict"] for h in hyp}))


if __name__ == "__main__":
    main()
