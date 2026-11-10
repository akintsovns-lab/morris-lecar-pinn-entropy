"""Markdown tables and formatted strings for the manuscript; every value is read from results/ (CTX)."""
import math
CTX = {}
M4 = ["SampEn", "PE", "WPE", "SpEn"]


def _f(x, n=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{float(x):.{n}f}".replace("-", "−")


def table_reference():
    R = CTX["BS"]["reference"]
    rows = [("292 currents, 1000 ms (primary window)", "ref_292_1000ms"), ("292 currents, 500 ms", "ref_292_500ms"),
            ("292 currents, mean of 1000 ms windows over 20 s", "ref_292_long"),
            ("292 currents, full 20 s record", "ref_292_full20s"),
            ("34 currents, 1000 ms (benchmark grid)", "ref_34_1000ms"),
            ("34 currents, mean of 1000 ms windows over 20 s", "ref_34_long")]
    out = ["| Grid and window | SampEn | PE | WPE | SpEn |", "|---|---:|---:|---:|---:|"]
    for lab, k in rows:
        out.append(f"| {lab} | " + " | ".join(_f(R[k][m]["eps"], 4) for m in M4) + " |")
    return "\n".join(out)


def table_reference_auc():
    R = CTX["BS"]["reference"]
    return ", ".join(f"{m} {_f(R['ref_34_1000ms'][m]['auc'], 3)}" for m in M4)


def cond(sc, rho, eta, var):
    key = f"{sc}|{rho}|{eta}|{var}"
    return CTX["BS"]["conditions"].get(key)


def med(sc, rho, eta, var, m, what="eps_median", n=4):
    c = cond(sc, rho, eta, var)
    return _f(c[f"{m}_{what}"], n) if c else "n/a"


def nreal(sc, rho, eta, var):
    c = cond(sc, rho, eta, var)
    return c["n"] if c else 0


def _conds_sorted(variants=None):
    C = CTX["BS"]["conditions"]
    ks = [k for k, a in C.items() if variants is None or a["variant"] in variants]
    order = dict(pinn=0, dataonly=1, noadapt=2, spline=3, odelsq=4)
    return sorted(ks, key=lambda k: (C[k]["scenario"], -C[k]["rho"], C[k]["eta"], order.get(C[k]["variant"], 9)))


VNAME = dict(pinn="PINN", spline="Cubic spline", odelsq="ODE-LSQ", dataonly="Data-only network", noadapt="PINN, fixed weights")
VSHORT = dict(pinn="PINN", spline="Spline", odelsq="ODE-LSQ", dataonly="Data-only", noadapt="PINN-fw")


def _iqr(a, q, n=4):
    v = a.get(q); i = a.get(q + "_iqr")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{_f(v, n)} [{_f(i[0], n)}, {_f(i[1], n)}]"


def table_recon():
    """Reconstruction accuracy on the test currents (median over realizations, IQR)."""
    C = CTX["BS"]["conditions"]
    out = ["| Scenario | ρ | η | Method | n | rel. L2(V) | rel. L2(w) |", "|---|---:|---:|---|---:|---|---|"]
    for k in _conds_sorted():
        a = C[k]
        out.append(f"| {a['scenario']} | {a['rho']:g} | {a['eta']:g} | {VNAME[a['variant']]} | {a['n']} | "
                   f"{_iqr(a, 'relL2_V_med_test')} | {_iqr(a, 'relL2_w_med_test')} |")
    return "\n".join(out)


def table_eps():
    """Median localization error and median AUC per measure and condition."""
    C = CTX["BS"]["conditions"]
    out = ["| Sc. | ρ | η | Method | n | " + " | ".join(f"ε {m}" for m in M4) + " | " + " | ".join(f"AUC {m}" for m in M4) + " |",
           "|---|---:|---:|---|---:|" + "---:|" * 8]
    for k in _conds_sorted():
        a = C[k]
        out.append(f"| {a['scenario']} | {a['rho']:g} | {a['eta']:g} | {VSHORT[a['variant']]} | {a['n']} | "
                   + " | ".join(_f(a[f'{m}_eps_median'], 4) for m in M4) + " | "
                   + " | ".join(_f(a[f'{m}_auc_median'], 2) for m in M4) + " |")
    return "\n".join(out)


def verdict(h):
    return CTX["BS"]["hypotheses"][h]["verdict"].lower()


def cval(sc, rho, eta, var, q, n=4):
    c = cond(sc, rho, eta, var)
    return _f(c[q], n) if c and q in c else "n/a"


def table_hyp():
    H = CTX["BS"]["hypotheses"]
    d1, d2, d3, d4, d5 = (H[k] for k in ("H1", "H2", "H3", "H4", "H5"))
    def r(x, n=4):
        return _f(x, n)
    e1 = (f"reference: ε(WPE) = {r(d1['reference_eps_WPE'])}, ε(PE) = {r(d1['reference_eps_PE'])}; "
          f"WPE significantly better in {d1['n_WPE_better']} and PE in {d1['n_PE_better']} of {d1['n_conditions']} conditions")
    dl = "; ".join(f"ρ = {rho}: " + ", ".join(f"{m} {r(v[m], 3)}" for m in M4) for rho, v in d2["Delta"].items())
    sm = "; ".join(f"ρ = {rho}: {', '.join(v)}" for rho, v in d2["smallest"].items())
    e2 = f"Δ_k (µA cm⁻²): {dl}; smallest Δ: {sm}"
    tA = d3["taus"].get("A|1.0|0.0|pinn", {}).get("tau", float("nan"))
    tB = ", ".join(f"ρ = {k}: {'yes' if v else 'no'}" for k, v in d3["preserved_B"].items())
    e3 = f"τ_b(A) = {r(tA, 2)}; ranking preserved in B at {tB}; ρ_th = {d3['rho_th']}"
    e4 = (f"PINN significantly better in {d4['n_pinn_better']}, spline in {d4['n_spline_better']} of "
          f"{len(d4['tests'])} conditions")
    e5 = (f"median rel. L2(ŵ): A {r(d5['A_median_relL2_w_test'], 3)}; B " +
          ", ".join(f"ρ = {k}: {r(v, 3)}" for k, v in d5["B_median_relL2_w_test"].items()))
    rows = [("H1", "WPE at least as accurate as PE", e1, d1), ("H2", "SpEn most noise-robust", e2, d2),
            ("H3", "ranking preserved after reconstruction", e3, d3), ("H4", "PINN better than spline at ρ ≤ 0.10", e4, d4),
            ("H5", "hidden w recovered", e5, d5)]
    out = ["| ID | Hypothesis | Evidence | Verdict |", "|---|---|---|---|"]
    for i, s, e, d in rows:
        out.append(f"| {i} | {s} | {e} | {d['verdict'].capitalize()} |")
    return "\n".join(out)
