"""All manuscript figures (PDF + PNG, 300 dpi) generated from results/ only."""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(ROOT, "results")
M = ["SampEn", "PE", "WPE", "SpEn"]
COL = dict(SampEn="#1b6ca8", PE="#d1495b", WPE="#edae49", SpEn="#00798c")
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def summary():
    return json.load(open(os.path.join(RES, "benchmark", "benchmark_summary.json")))


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.7)); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(-0.6, 3.2)
    top = [(0.1, "Morris–Lecar ODE\n(Table 1, class I)\nparameters known"),
           (2.1, "Reference $V(t)$, $w(t)$\nDOP853,\nrtol = $10^{-10}$,\n292 currents"),
           (4.1, "Sparse protocol\nmask $\\rho$, noise $\\eta$\n$w$ never observed"),
           (6.1, "PINN inverse\nreconstruction\n$\\hat{V}(t)$, $\\hat{w}(t)$"),
           (8.1, "Entropies $H_k(I)$\nSampEn, PE, WPE,\nSpEn (reference\nand reconstructions)")]
    bot = [(2.1, "Continuation\n$I_{\\mathrm{crit}}$ (SNIC)\n$T \\propto (I-I_{\\mathrm{c}})^{-1/2}$"),
           (4.1, "Baselines\ncubic spline,\nODE-LSQ,\ndata-only network"),
           (6.1, "Benchmark\n$\\varepsilon_k = |I^{*}_k - I_{\\mathrm{crit}}|$\nWilcoxon + Holm"),
           (8.1, "Fixed detection rule\n$I^{*} = \\arg\\max\\,|dH/dI|$\n+ change point")]
    for (x, t), y in [(b, 1.9) for b in top] + [(b, 0.2) for b in bot]:
        ax.add_patch(FancyBboxPatch((x, y), 1.8, 1.05, boxstyle="round,pad=0.04", fc="#eef3f8", ec="#33507a", lw=0.9))
        ax.text(x + 0.9, y + 0.52, t, ha="center", va="center", fontsize=7.0)
    arr = dict(arrowstyle="-|>", color="#33507a", lw=0.9)
    for x in (1.9, 3.9, 5.9, 7.9):
        ax.annotate("", (x + 0.2, 2.42), (x, 2.42), arrowprops=arr)
    ax.annotate("", (3.0, 1.29), (3.0, 1.86), arrowprops=arr)      # reference -> continuation
    ax.annotate("", (5.0, 1.29), (5.0, 1.86), arrowprops=arr)      # sparse -> baselines
    ax.annotate("", (8.25, 1.86), (5.95, 1.0), arrowprops=arr)     # baselines -> entropies
    ax.annotate("", (9.0, 1.29), (9.0, 1.86), arrowprops=arr)      # entropies -> detection
    ax.annotate("", (7.94, 0.72), (8.06, 0.72), arrowprops=arr)    # detection -> benchmark
    ax.annotate("", (6.6, 0.16), (3.0, 0.16), arrowprops=dict(arrowstyle="-|>", color="#33507a", lw=0.9,
                                                              connectionstyle="arc3,rad=0.25"))
    ax.text(4.8, -0.5, "$I_{\\mathrm{crit}}$ enters only the error $\\varepsilon$ (never the detection rule)", ha="center", fontsize=6.5, color="#33507a")
    save(fig, "fig1_pipeline")


def fig2_bifurcation():
    b = np.load(os.path.join(RES, "bifurcation_branch.npz")); r = json.load(open(os.path.join(RES, "bifurcation.json")))
    Ic = r["I_SN"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axs[0]
    st = (b["tr"] < 0) & (b["det"] > 0)
    sad = b["det"] < 0
    uns = (b["det"] > 0) & (b["tr"] > 0)
    for m, ls, lab in ((st, "-", "stable node/focus"), (sad, "--", "saddle"), (uns, ":", "unstable focus/node")):
        V = np.where(m, b["V"], np.nan); ax.plot(b["I"], V, ls, color="k", lw=1, label=lab)
    # periodic orbits from reference sweep
    Ir, Vmin, Vmax = [], [], []
    for fn in sorted(glob.glob(os.path.join(RES, "reference", "*.npz"))):
        d = np.load(fn)
        if str(d["status"]) == "cycle":
            Ir.append(float(d["I"])); Vmin.append(float(d["Vmin"])); Vmax.append(float(d["Vmax"]))
    o = np.argsort(Ir)
    ax.plot(np.array(Ir)[o], np.array(Vmax)[o], ".", ms=2.5, color="#d1495b", label="limit cycle max/min (DOP853)")
    ax.plot(np.array(Ir)[o], np.array(Vmin)[o], ".", ms=2.5, color="#d1495b")
    ax.axvline(Ic, color="#1b6ca8", lw=0.8); ax.text(Ic + 0.4, 36, f"$I_{{\\mathrm{{crit}}}}$ = {Ic:.5f}", color="#1b6ca8", fontsize=7.5)
    ax.set_xlim(25, 52); ax.set_ylim(-60, 40); ax.set_xlabel("$I$ (µA cm$^{-2}$)"); ax.set_ylabel("$V$ (mV)")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5); ax.set_title("(a) Equilibria and periodic orbits")
    ax = axs[1]
    cyc = r["cycles"]; d = np.array([c["I"] - Ic for c in cyc]); T = np.array([c["T"] for c in cyc])
    ax.loglog(d, T, "o", ms=3.5, color="#00798c", label="shooting (Poincaré)")
    f = r["fit_free"]; dd = np.logspace(np.log10(d.min()), np.log10(d.max()), 100)
    ax.loglog(dd, f["A"] * dd ** (-f["beta"]) + f["B"], "-", color="k", lw=0.8,
              label=f"fit $T = A\\,(I-I_{{\\mathrm{{c}}}})^{{-\\beta}} + B$, $\\beta$ = {f['beta']:.4f}")
    ax.set_xlabel("$I - I_{\\mathrm{crit}}$ (µA cm$^{-2}$)"); ax.set_ylabel("period $T$ (ms)"); ax.legend(frameon=False, loc="upper right")
    ax.set_title("(b) SNIC scaling of the period")
    fig.tight_layout(); save(fig, "fig2_bifurcation")


def fig3_reconstruction(tags=("B_rho0.05_eta0.00_rep00_pinn", "C_rho0.05_eta0.10_rep00_pinn"), currents=(39.75, 40.1, 46.0)):
    sys.path.insert(0, HERE)
    from train import observations, load_ref
    from grids import pinn_currents
    P = pinn_currents()
    fig, axs = plt.subplots(len(currents), 2 * len(tags), figsize=(7.4, 1.8 * len(currents) + 0.4), sharex=True)
    for c, tag in enumerate(tags):
        d = np.load(os.path.join(RES, "pinn", tag + ".npz"))
        sp = np.load(os.path.join(RES, "pinn", tag.replace("_pinn", "_spline") + ".npz"))
        rho = float(tag.split("_")[1][3:]); eta = float(tag.split("_")[2][3:])
        for i, I in enumerate(currents):
            j = int(np.argmin(np.abs(P - I))); t, Vr, wr = load_ref(P[j])
            o = observations(P[j], j, rho, eta, 0)
            ax = axs[i, 2 * c]
            ax.plot(t, Vr, color="k", lw=0.9, label="reference")
            ax.plot(t, sp["V"][j], color="#999999", lw=0.6, label="cubic spline")
            ax.plot(t, d["V"][j], "--", color="#d1495b", lw=0.8, label="PINN")
            ax.plot(o["t_obs"], o["V_obs"], ".", ms=1.3, color="#1b6ca8", label="observed")
            ax.set_ylabel(f"I = {P[j]:.2f} µA/cm²\nV (mV)", fontsize=7); ax.ticklabel_format(axis="y", useOffset=False)
            ax2 = axs[i, 2 * c + 1]
            ax2.plot(t, wr, color="k", lw=0.9); ax2.plot(t, d["w"][j], "--", color="#d1495b", lw=0.8)
            ax2.set_ylabel("w", fontsize=7); ax2.ticklabel_format(axis="y", useOffset=False)
            if i == 0:
                ax.set_title(f"V: ρ = {rho:g}, η = {eta:g}"); ax2.set_title(f"w (never observed)")
        axs[-1, 2 * c].set_xlabel("t (ms)"); axs[-1, 2 * c + 1].set_xlabel("t (ms)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, "fig3_reconstruction")


def _cond_H(tagpat):
    out = []
    for fn in sorted(glob.glob(os.path.join(RES, "entropy", "rec", tagpat + ".json"))):
        e = json.load(open(fn)); out.append(e)
    return out


def fig4_entropy_curves():
    s = summary(); R = json.load(open(os.path.join(RES, "entropy", "reference_entropy.json")))
    I = np.array(R["I"]); Ic = s["extra"]["I_crit"]; P = np.array(s["reference"]["I_34"])
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.8))
    for k, m in enumerate(M):
        for row, (xl, xr) in enumerate(((30, 50), (39.55, 40.65))):
            ax = axs[row, k]
            ax.plot(I, R["short"][m], "-", color="k", lw=0.9, label="reference, 1000 ms")
            ax.fill_between(I, R["long_ci_lo"][m], R["long_ci_hi"][m], color="#bbbbbb", alpha=0.6, lw=0,
                            label="reference, 20 s windows (95% CI)")
            for pat, col, lab in (("B_rho0.05_eta0.00_rep*_pinn", "#d1495b", "PINN ρ=0.05"),
                                  ("C_rho0.05_eta0.10_rep*_pinn", "#00798c", "PINN ρ=0.05, η=0.10"),
                                  ("C_rho0.05_eta0.10_rep*_spline", "#edae49", "spline ρ=0.05, η=0.10"),
                                  ("C_rho0.05_eta0.10_rep*_odelsq", "#6a4c93", "ODE-LSQ ρ=0.05, η=0.10")):
                Hs = _cond_H(pat)
                if not Hs:
                    continue
                A = np.array([[h[f"{x:.4f}"][m] for x in P] for h in Hs])
                med = np.median(A, 0); lo, hi = np.percentile(A, [25, 75], 0)
                ax.errorbar(P, med, yerr=[med - lo, hi - med], fmt="o", ms=2, lw=0.6, color=col, label=lab)
            ax.axvline(Ic, color="#1b6ca8", lw=0.6, ls=":")
            ax.set_xlim(xl, xr)
            if row == 0:
                ax.set_title(m)
            ax.set_xlabel("I (µA/cm²)")
        axs[0, 0].set_ylabel("$H$ (full range)"); axs[1, 0].set_ylabel("$H$ (near $I_{\\mathrm{crit}}$)")
    h, l = axs[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "fig4_entropy_curves")


def fig5_eps():
    from matplotlib.lines import Line2D
    s = summary(); C = s["conditions"]; Rr = s["reference"]["ref_34_1000ms"]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9))
    off = dict(SampEn=-0.24, PE=-0.08, WPE=0.08, SpEn=0.24)
    VM = dict(pinn=("o", True), spline=("s", False), odelsq=("D", False))
    def pt(ax, x, a, m, var, key="eps"):
        if not a:
            return
        mk, fill = VM[var]
        if key == "eps":
            y = a[f"{m}_eps_median"]; lo, hi = a[f"{m}_eps_iqr"]
            ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt=mk, ms=3.3, lw=0.7, color=COL[m],
                        mfc=COL[m] if fill else "white", mew=0.9)
        else:
            ax.plot([x], [a[f"{m}_auc_median"]], mk, ms=3.3, color=COL[m], mfc=COL[m] if fill else "white", mew=0.9)
    rhos = [0.5, 0.2, 0.1, 0.05]
    for m in M:
        for i, r in enumerate(rhos):
            for var, dx in (("pinn", -0.03), ("spline", 0.03), ("odelsq", 0.06)):
                pt(axs[0], i + off[m] + dx, C.get(f"B|{r}|0.0|{var}"), m, var)
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels([f"{r:g}" for r in rhos]); axs[0].set_xlabel("retained fraction ρ")
    axs[0].set_title("(a) Scenario B: sparsity")
    etas = [0.0, 0.01, 0.05, 0.10]
    for m in M:
        for i, e in enumerate(etas):
            for var, dx in (("pinn", -0.03), ("spline", 0.03), ("odelsq", 0.06)):
                a = C.get(f"B|0.05|0.0|{var}") if e == 0 else C.get(f"C|0.05|{e}|{var}")
                pt(axs[1], i + off[m] + dx, a, m, var)
                pt(axs[2], i + off[m] + dx, a, m, var, key="auc")
    for ax in axs[:2]:
        ax.set_yscale("log"); ax.set_ylim(0.007, 1.5)
        ax.axhline(Rr["PE"]["eps"], color="#888888", lw=0.5, ls=":")
    axs[0].set_ylabel("$\\varepsilon = |I^{*} - I_{\\mathrm{crit}}|$ (µA cm$^{-2}$)")
    axs[1].set_title("(b) Scenario C, ρ = 0.05: noise")
    axs[2].set_title("(c) regime separation, ρ = 0.05"); axs[2].set_ylabel("AUC (spiking > rest)")
    axs[2].axhline(0.5, color="#888888", lw=0.5, ls=":"); axs[2].set_ylim(-0.05, 1.05)
    for ax in axs[1:]:
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{e:g}" for e in etas]); ax.set_xlabel("noise level η")
    h = [Line2D([], [], color=COL[m], marker="o", ls="", ms=4, label=m) for m in M]
    h += [Line2D([], [], color="k", marker=VM[v][0], mfc="k" if VM[v][1] else "white", ls="", ms=4, label=l)
          for v, l in (("pinn", "PINN"), ("spline", "cubic spline"), ("odelsq", "ODE-LSQ"))]
    fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=6.8)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "fig5_eps_sparsity_noise")


def fig6_stability():
    rows = json.load(open(os.path.join(RES, "benchmark", "realizations.json")))
    best = summary()["reference"]["ref_34_1000ms"]["PE"]["eps"]
    conds = [("A", 1.0, 0.0), ("B", 0.5, 0.0), ("B", 0.2, 0.0), ("B", 0.1, 0.0), ("B", 0.05, 0.0),
             ("C", 0.2, 0.01), ("C", 0.2, 0.05), ("C", 0.2, 0.1), ("C", 0.05, 0.01), ("C", 0.05, 0.05), ("C", 0.05, 0.1)]
    lab = ["A" if c[0] == "A" else (f"B ρ={c[1]:g}" if c[0] == "B" else f"C ρ={c[1]:g} η={c[2]:g}") for c in conds]
    fig = plt.figure(figsize=(7.4, 5.4)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    sel = lambda c: [r for r in rows if (r["scenario"], r["rho"], r["eta"]) == c and r["variant"] == "pinn"]
    for ax, q, tt in ((axs[0], "relL2_V_med_all", "(a) rel. L2 error of V"),
                      (axs[1], "relL2_w_med_all", "(b) rel. L2 error of hidden w")):
        data = [[r[q] for r in sel(c) if q in r] for c in conds]
        ax.boxplot([d if d else [np.nan] for d in data], showfliers=False, widths=0.6,
                   medianprops=dict(color="k", lw=0.8))
        for i, d in enumerate(data):
            if d:
                ax.plot(np.full(len(d), i + 1) + np.linspace(-0.18, 0.18, len(d)), d, ".", ms=2.5, color="#d1495b")
        ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=6.3)
        ax.set_yscale("log"); ax.set_title(tt + " (median over 34 currents)", fontsize=8)
    ax = axs[2]; wdt = 0.2
    for k, m in enumerate(M):
        fr, nn = [], []
        for c in conds:
            rs = sel(c); nn.append(len(rs))
            fr.append(np.mean([r[f"{m}_eps"] <= best + 1e-9 for r in rs]) if rs else np.nan)
        ax.bar(np.arange(len(conds)) + (k - 1.5) * wdt, fr, wdt, color=COL[m], label=m)
    ax.set_xticks(range(len(conds))); lab2 = ["A" if c[0] == "A" else (f"B\nρ={c[1]:g}" if c[0] == "B" else f"C\nρ={c[1]:g}\nη={c[2]:g}") for c in conds]
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(lab2, nn)], fontsize=6.0)
    ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("fraction of realizations")
    ax.set_title("(c) fraction of PINN realizations with the smallest achievable error ε = %.4f µA/cm²" % best, fontsize=8)
    ax.legend(frameon=False, ncol=4, fontsize=6.5, loc="upper center")
    fig.tight_layout(); save(fig, "fig6_run_stability")


from matplotlib.colors import Normalize
import matplotlib.cm as cm


def fig7_sensitivity():
    s = summary(); sens = s["reference"]["sensitivity"]
    A = np.full((2, 4), np.nan)
    for v in sens.values():
        if v["measure"] == "SampEn":
            A[[2, 3].index(v["params"]["m"]), [0.1, 0.15, 0.2, 0.25].index(v["params"]["r_frac"])] = v["eps"]
    Bs = {}
    for m in ("PE", "WPE"):
        B = np.full((4, 4), np.nan)
        for v in sens.values():
            if v["measure"] == m:
                B[[3, 4, 5, 6].index(v["params"]["d"]), [1, 2, 4, 8].index(v["params"]["tau"])] = v["eps"]
        Bs[m] = B
    allv = np.log10(np.concatenate([A.ravel(), Bs["PE"].ravel(), Bs["WPE"].ravel()]))
    norm = Normalize(vmin=np.floor(allv.min() * 4) / 4, vmax=np.ceil(allv.max() * 4) / 4)
    cmap = "Blues"
    fig = plt.figure(figsize=(7.8, 2.7))
    pos = [(0.060, 0.19, 0.165, 0.64), (0.300, 0.19, 0.165, 0.64), (0.540, 0.19, 0.165, 0.64)]
    axs = [fig.add_axes(p) for p in pos]
    cax = fig.add_axes((0.725, 0.19, 0.012, 0.64))
    axs.append(fig.add_axes((0.885, 0.19, 0.105, 0.64)))

    def heat(ax, mat, xt, yt, title, xl, yl):
        ax.imshow(np.log10(mat), cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(xt)
        ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(yt)
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.tick_params(axis="x", labelsize=6.8)
        # grid separating the cells
        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True); ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.grid(which="minor", color="#222222", lw=1.0); ax.tick_params(which="minor", length=0)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#222222"); sp.set_linewidth(1.0)
        for (i, j), z in np.ndenumerate(mat):
            dark = norm(np.log10(z)) > 0.55
            ax.text(j, i, f"{z:.3f}", ha="center", va="center", fontsize=5.6, color="white" if dark else "black")
    heat(axs[0], A, [0.1, 0.15, 0.2, 0.25], [2, 3], "SampEn ($m \\times r$)", "$r$ / SD", "$m$")
    heat(axs[1], Bs["PE"], [1, 2, 4, 8], [3, 4, 5, 6], "PE ($d \\times \\tau$)", "$\\tau$ (samples)", "$d$")
    heat(axs[2], Bs["WPE"], [1, 2, 4, 8], [3, 4, 5, 6], "WPE ($d \\times \\tau$)", "$\\tau$ (samples)", "$d$")
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.ax.set_title("$\\log_{10}\\varepsilon$", fontsize=7.5, pad=6)
    cb.ax.tick_params(labelsize=6.5)
    ax = axs[3]
    pts = sorted((v["params"]["nperseg"], v["eps"]) for v in sens.values() if v["measure"] == "SpEn")
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "o-", color="#00798c"); ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs]); ax.set_xlim(100, 660); ax.set_ylim(0.0095, 0.036)
    ax.set_yticks([0.01, 0.02, 0.03]); ax.set_yticklabels(["0.01", "0.02", "0.03"]); ax.minorticks_off()
    ax.set_xlabel("Welch segment (samples)"); ax.set_title("SpEn"); ax.set_ylabel("$\\varepsilon$ (µA cm$^{-2}$)")
    fig.savefig(os.path.join(FIG, "fig7_sensitivity.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig7_sensitivity.png"), dpi=300, bbox_inches="tight"); plt.close(fig)



def fig8_param():
    fs = sorted(glob.glob(os.path.join(RES, "param", "param_seed*.json")))
    if not fs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for fn in fs:
        r = json.load(open(fn))
        for I, m in r["per_current"].items():
            c = dict(train="#999999", val="#edae49", test="#d1495b")[m["role"]]
            axs[0].plot(float(I), m["relL2_V"], "o", ms=3, color=c, alpha=0.7)
            if np.isfinite(m["first_spike_ref"]) or np.isfinite(m["first_spike_hat"]):
                fr = m["first_spike_ref"] if np.isfinite(m["first_spike_ref"]) else 520
                fh = m["first_spike_hat"] if np.isfinite(m["first_spike_hat"]) else 520
                axs[1].plot(fr, fh, "o", ms=3, color=c, alpha=0.7)
    for role, c in (("train", "#999999"), ("validation", "#edae49"), ("test", "#d1495b")):
        axs[0].plot([], [], "o", color=c, label=role)
    axs[0].set_yscale("log"); axs[0].set_xlabel("I (µA/cm²)"); axs[0].set_ylabel("rel. L2 error of V")
    axs[0].legend(frameon=False); axs[0].set_title("(a) Parametric PINN, 5 seeds")
    axs[1].plot([0, 520], [0, 520], "k:", lw=0.7); axs[1].set_xlabel("first-spike latency, reference (ms)")
    axs[1].set_ylabel("first-spike latency, PINN (ms)"); axs[1].set_title("(b) latency (top row: no spike within 500 ms)")
    fig.tight_layout(); save(fig, "fig8_parametric")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8"]
    F = {"1": fig1_pipeline, "2": fig2_bifurcation, "3": fig3_reconstruction, "4": fig4_entropy_curves, "5": fig5_eps,
         "6": fig6_stability, "7": fig7_sensitivity, "8": fig8_param}
    for w in which:
        F[w]()
        print("fig", w, "ok")
