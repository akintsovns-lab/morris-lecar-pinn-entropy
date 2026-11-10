# Pre-registered hypotheses (written before any computation)

Date of registration: 2026-10-30 (Stage 0, before the first ODE integration).
Hypotheses are evaluated only once, after Stage 6, with the fixed detection rule
defined in `configs/benchmark.yaml`. They are not modified after results are seen.

Notation: for entropy measure k in {SampEn, PE, WPE, SpEn}, eps_k = |I_k* - I_crit|,
where I_k* is obtained by the pre-fixed smoothed-derivative rule; I_crit comes from
continuation (Stage 2). "Realization" = one mask/noise/initialization draw.
Paired tests: two-sided Wilcoxon signed-rank on per-realization eps values,
Holm correction across all pairwise comparisons inside one hypothesis, alpha = 0.05.
When all paired differences are zero (ties), the comparison is recorded as "no difference".

## H1 — amplitude weighting (WPE vs PE)
Statement: WPE localizes the transition at least as accurately as PE.
Criterion:
- Reference (ODE) signals: eps_WPE <= eps_PE (single deterministic value).
- Sparse/noisy PINN reconstructions (Scenarios B and C): over all (rho, eta) conditions,
  count conditions where median eps_WPE < median eps_PE with Holm-adjusted p < 0.05.
- CONFIRMED if this holds in >= 50% of the conditions and PE is never significantly better;
  REFUTED if PE is significantly better in >= 50% of conditions;
  INCONCLUSIVE otherwise.

## H2 — noise robustness of spectral entropy
Statement: SpEn is the measure whose transition estimate is most robust to measurement noise.
Robustness index per measure and sparsity: Delta_k(rho) = median eps_k(rho, eta_max) - median eps_k(rho, eta=0),
with eta_max = 0.10.
Criterion:
- CONFIRMED if SpEn has the smallest Delta_k at >= 50% of rho levels and, at eta_max,
  eps_SpEn is not significantly worse (Holm) than any other measure at those rho levels;
- REFUTED if another measure has a strictly smaller Delta_k at >= 50% of rho levels;
- INCONCLUSIVE otherwise (including ties).

## H3 — preservation of the ranking after PINN reconstruction
Statement: The ranking of the four measures by eps on the reference signals is preserved after
PINN reconstruction when the retained fraction rho is above a threshold.
Criterion: Kendall tau between the reference ranking and the ranking by median eps in a condition.
"Preserved" means tau >= 0.67 (at most one discordant pair among four measures).
- CONFIRMED if the ranking is preserved in Scenario A and there is a threshold rho_th in
  {0.05, 0.10, 0.20, 0.50} such that it is preserved for all rho >= rho_th (Scenario B);
- REFUTED if it is not preserved already in Scenario A;
- INCONCLUSIVE if preserved in A but no monotone threshold exists.
If the reference ranking contains ties, tau-b is used.

## H4 — physics constraints versus unconstrained interpolation
Statement: At low retained fractions (rho <= 0.10) the PINN preserves transition-relevant
structure better than cubic-spline interpolation of the same sparse observations.
Criterion: pooled over the four measures, median eps_PINN < median eps_spline with
Holm-adjusted p < 0.05 in >= 50% of the (rho <= 0.10, eta) conditions -> CONFIRMED;
spline significantly better in >= 50% -> REFUTED; otherwise INCONCLUSIVE.

## H5 — hidden-state recovery
Statement: The voltage-only PINN recovers the unobserved recovery variable w(t).
Criterion: median relative L2 error of w_hat over test realizations < 0.10 in Scenario A and in
Scenario B for rho >= 0.10 -> CONFIRMED; > 0.10 in Scenario A -> REFUTED; otherwise INCONCLUSIVE.

## Fixed analysis rules registered together with the hypotheses
- The entropy value at I_crit itself is never used; grid currents within 0.005 uA/cm^2 of
  I_crit are excluded from detection and ranking.
- The detection rule (smoothing, derivative, change point) is fixed in `configs/benchmark.yaml`
  before Stage 5 and is applied identically to all signals.
- No seed selection: all realizations that were run are reported; failed trainings are
  reported as failures and included in the statistics with their actual (possibly large) error.
