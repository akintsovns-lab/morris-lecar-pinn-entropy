"""Supplementary material (Tables S1-S4, Figures S1-S3) generated from results/ only -> supplementary/"""
import os, sys, json, glob, ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
RES = os.path.join(ROOT, "results"); OUT = os.path.join(ROOT, "supplementary"); os.makedirs(OUT, exist_ok=True)
M = ["SampEn", "PE", "WPE", "SpEn"]
COL = dict(SampEn="#1b6ca8", PE="#d1495b", WPE="#edae49", SpEn="#00798c")
VCOL = dict(pinn="#1b6ca8", spline="#d1495b", odelsq="#00798c", dataonly="#edae49", noadapt="#6a4c93")
S = yaml.safe_load(open(os.path.join(ROOT, "configs", "sparse.yaml")))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def g(x, n=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{n}g}"


def table_s1(B):
    c = B["SNIC_criteria"]; f = B["folds"][0]; ff = B["fit_free"]
    ev = {
        "simple_zero_eigenvalue": f"eigenvalues at the fold: {f['eig'][0]:.1e}, {f['eig'][1]:.4f} ms^-1",
        "nondegenerate_fold": f"d2I/dV*2 = {f['d2I_dV2']:.4f}",
        "transversal": f"dF/dI = {f['dF_dI']:.3g}",
        "stable_cycle_above": "all nontrivial Floquet multipliers < 1 above I_SN (Poincare shooting)",
        "orbit_approaches_SN": "minimum distance of the cycle to (V_SN, w_SN) decreases as I -> I_SN+",
        "period_diverges": f"free fit beta = {ff['beta']:.5f} +/- {ff['err']['beta']:.1e}",
        "no_cycle_below": "no sustained oscillation below I_SN in direct simulation from multiple initial conditions",
        "Ic_consistent": f"|I_c(fit) - I_SN| = {abs(ff['Ic'] - B['I_SN']):.1e} uA/cm^2",
    }
    L = ["# Table S1. SNIC verification criteria", "", "| Criterion | Satisfied | Evidence |", "|---|---|---|"]
    for k, v in c.items():
        L.append(f"| {k.replace('_', ' ')} | {'yes' if v else 'no'} | {ev.get(k, '')} |")
    L += ["", f"SNIC confirmed: {B['SNIC_confirmed']}. Values from results/bifurcation.json."]
    open(os.path.join(OUT, "Table_S1_SNIC_criteria.md"), "w").write("\n".join(L) + "\n")


def table_s2():
    rows = []
    for fn in sorted(glob.glob(os.path.join(RES, "pinn", "*.json"))):
        d = json.load(open(fn))
        rows.append((d.get("tag", os.path.basename(fn)[:-5]), d.get("seed", ""), d.get("wall_s", "")))
    L = ["# Table S2. Seeds and realization indices", "",
         f"Mask seed = {S['mask_seed_base']} + rep; noise seed = {S['noise_seed_base']} + rep; "
         f"network initialization seed = {S['init_seed_base']} + rep (Scenario A: {S['init_seed_base']} + init). "
         "Parametric experiment: masks 77000 + 100 * seed + j (j = current index). "
         "All realizations that were run are listed; none was discarded.", "",
         "| Realization | Init seed | Wall time (s) |", "|---|---|---|"]
    for t, s, w in rows:
        try:
            w = f"{float(w):.0f}"
        except (TypeError, ValueError):
            w = ""
        L.append(f"| {t} | {s} | {w} |")
    open(os.path.join(OUT, "Table_S2_seeds.md"), "w").write("\n".join(L) + "\n")


def table_s3(BS):
    L = ["# Table S3. Per-condition summary statistics", "",
         "Median localization error eps_k (uA/cm^2) with bootstrap 95% CI of the median, median AUC, "
         "median relative L2 errors of V and w on test currents, n = number of realizations.", "",
         "| Scenario | rho | eta | Method | n | " + " | ".join(f"eps {k} [CI]" for k in M) +
         " | " + " | ".join(f"AUC {k}" for k in M) + " | relL2 V | relL2 w |",
         "|" + "---|" * (5 + 2 * len(M) + 2)]
    for key, a in sorted(BS["conditions"].items(), key=lambda kv: (kv[1]["scenario"], -kv[1]["rho"], kv[1]["eta"], kv[1]["variant"])):
        e = " | ".join(f"{g(a[k+'_eps_median'])} [{g(a[k+'_eps_ci'][0])}, {g(a[k+'_eps_ci'][1])}]" for k in M)
        u = " | ".join(g(a[k + "_auc_median"], 3) for k in M)
        L.append(f"| {a['scenario']} | {a['rho']} | {a['eta']} | {a['variant']} | {a['n']} | {e} | {u} | "
                 f"{g(a.get('relL2_V_med_test'), 3)} | {g(a.get('relL2_w_med_test'), 3)} |")
    open(os.path.join(OUT, "Table_S3_conditions.md"), "w").write("\n".join(L) + "\n")


def table_s4(BS):
    H = BS["hypotheses"]
    L = ["# Table S4. Paired tests (two-sided Wilcoxon signed-rank, Holm correction within each hypothesis)", ""]
    for h, d in H.items():
        L += [f"## {h}: verdict {d.get('verdict', 'n/a')}", ""]
        rows = d.get("conditions") or d.get("tests") or []
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            keys = [k for k in rows[0].keys()]
            L += ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
            for r in rows:
                L.append("| " + " | ".join(g(r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")
        else:
            L.append("```\n" + json.dumps({k: v for k, v in d.items() if k != "verdict"}, indent=1, default=str)[:4000] + "\n```")
        L.append("")
    open(os.path.join(OUT, "Table_S4_tests.md"), "w").write("\n".join(L) + "\n")


def fig_s1():
    """Reconstructions at all test currents, one realization per sparsity level (rep 0)."""
    from sparse import current_split
    from train import load_ref
    test = current_split()["test"]
    tags = [f"B_rho{r:.2f}_eta0.00_rep00_pinn" for r in S["rho_levels"]]
    tags = [t for t in tags if os.path.exists(os.path.join(RES, "pinn", t + ".npz"))]
    if not tags:
        return
    fig, axs = plt.subplots(len(tags), len(test), figsize=(9, 1.5 * len(tags) + 0.4), sharex=True, squeeze=False)
    for i, t in enumerate(tags):
        z = np.load(os.path.join(RES, "pinn", t + ".npz"))
        for j, I in enumerate(test):
            ax = axs[i, j]; k = int(np.argmin(np.abs(z["I"] - I)))
            tt, Vr, _ = load_ref(z["I"][k]); ax.plot(tt, Vr, color="k", lw=0.6)
            ax.plot(z["t"], z["V"][k], color="#1b6ca8", lw=0.5, alpha=0.85)
            if i == 0:
                ax.set_title(f"I = {I:.2f}", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"rho={t.split('_')[1][3:]}\nV (mV)", fontsize=7)
            ax.tick_params(labelsize=6)
    for ax in axs[-1]:
        ax.set_xlabel("t (ms)", fontsize=7)
    fig.suptitle("Figure S1. Reference (black) and PINN (blue) voltage at the test currents, Scenario B, realization 0", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_S1_reconstructions.png"), dpi=200); plt.close(fig)


def fig_s2():
    """Entropy curves of all methods (median over realizations) for the sparsest noisy condition."""
    BS = json.load(open(os.path.join(RES, "benchmark", "benchmark_summary.json")))
    I = np.array(BS["reference"]["I_34"]); Hr = BS["reference"]["H_ref_34"]
    rows = json.load(open(os.path.join(RES, "benchmark", "realizations.json")))
    conds = [("B", 0.05, 0.0), ("C", 0.05, 0.1)]
    fig, axs = plt.subplots(len(conds), 4, figsize=(9, 4.2), squeeze=False)
    for i, (sc, rho, eta) in enumerate(conds):
        for j, k in enumerate(M):
            ax = axs[i, j]; ax.plot(I, Hr[k], color="k", lw=1.2, label="reference")
            for v in ("pinn", "spline", "odelsq", "dataonly", "noadapt"):
                fs = sorted(glob.glob(os.path.join(RES, "entropy", "rec", f"{sc}_rho{rho:.2f}_eta{eta:.2f}_rep*_{v}.json")))
                if not fs:
                    continue
                Hs = np.array([[json.load(open(f))[f"{x:.4f}"][k] for x in I] for f in fs])
                ax.plot(I, np.median(Hs, 0), color=VCOL[v], lw=0.9, label=f"{v} (n={len(fs)})")
            ax.axvline(BS["extra"]["I_crit"], color="grey", lw=0.6, ls="--")
            ax.set_xlim(39.4, 41.2); ax.set_title(f"{k}: {sc}, rho={rho}, eta={eta}", fontsize=7)
            if i == len(conds) - 1:
                ax.set_xlabel("I (uA/cm2)")
    axs[0, 0].legend(fontsize=5.5, frameon=False)
    fig.suptitle("Figure S2. Median entropy curves near I_crit for all reconstruction methods", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_S2_entropy_methods.png"), dpi=200); plt.close(fig)


def fig_s3():
    """Training-loss histories of all PINN realizations (no selection)."""
    fs = sorted(glob.glob(os.path.join(RES, "pinn", "*_pinn*.json")))
    if not fs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    for fn in fs:
        d = json.load(open(fn)); h = d.get("history")
        if isinstance(h, str):
            try:
                h = ast.literal_eval(h)
            except Exception:
                continue
        if not h:
            continue
        it = np.cumsum([1] * len(h)); L = np.array([x.get("loss", np.nan) for x in h], float)
        dat = np.array([x.get("data", np.nan) for x in h], float)
        c = "#d1495b" if d.get("scenario") == "C" else ("#1b6ca8" if d.get("scenario") == "B" else "k")
        axs[0].semilogy(it, L, color=c, lw=0.4, alpha=0.5); axs[1].semilogy(it, dat, color=c, lw=0.4, alpha=0.5)
    axs[0].set_title("total loss (blue B, red C, black A)"); axs[1].set_title("data term")
    for ax in axs:
        ax.set_xlabel("logged checkpoint index")
    fig.suptitle(f"Figure S3. Training histories of all {len(fs)} PINN realizations", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_S3_training.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    B = json.load(open(os.path.join(RES, "bifurcation.json")))
    BS = json.load(open(os.path.join(RES, "benchmark", "benchmark_summary.json")))
    table_s1(B); table_s2(); table_s3(BS); table_s4(BS)
    fig_s1(); fig_s2(); fig_s3()
    print("supplementary written:", sorted(os.listdir(OUT)))
