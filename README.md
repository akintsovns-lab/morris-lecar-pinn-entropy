# Entropy-based detection of the Morris–Lecar SNIC transition from PINN-reconstructed sparse voltage data

Code, configurations, unit tests and processed results for the manuscript
"Entropy-Based Detection of the Rest-to-Firing Transition in Morris–Lecar Dynamics Reconstructed
from Sparse Voltage Data by Physics-Informed Neural Networks" (submitted to MDPI *Entropy*).

Every number, table and figure in the manuscript is produced by the scripts below from the files in
`results/`.

## Requirements

CPU only (the reported runs used 2 cores, no GPU). Python 3.14 and the pinned packages:

```
pip install -r requirements.txt
```

## Layout

| Path | Content |
|------|---------|
| `configs/` | model parameters, reference integration, sparse-data protocol, PINN, benchmark rule; `hypotheses.md` (pre-registered H1–H5, fixed before any computation) |
| `src/` | model, reference solver, continuation, sparse masks, PINN, baselines, entropy estimators, benchmark, figures |
| `tests/` | unit tests (model, entropy estimators vs. independent implementations, sparse protocol, benchmark rule) |
| `results/` | all processed results (JSON/CSV), logs, run configuration and environment snapshot |
| `figures/` | manuscript figures 1–8 (PDF + PNG, 300 dpi), written by `src/figures.py` |
| `supplementary/` | Tables S1–S4 and Figures S1–S3 |
| `DECISIONS.md`, `LIMITATIONS.md` | decision log (all deviations from the plan) and limitations (Russian) |
| `STATE.json` | stage status, configuration and output hashes |

## Reproduction

```
python3 run_all.py --quick      # unit tests, bifurcation analysis, benchmark, figures
python3 run_all.py              # everything, including reference sweep and all PINN trainings (many CPU hours)
python3 run_all.py --from 5     # recompute entropies/benchmark from existing reconstructions
```

Individual stages:

1. `src/stage1_reference.py`, `src/stage1_verify.py` – DOP853 reference trajectories (rtol 1e-10) for 292 currents and
   tolerance/method/output-grid checks.
2. `src/stage2_bifurcation.py` – equilibrium continuation, fold and Hopf points, periodic orbits by Poincaré shooting,
   period-scaling fit, SNIC criteria → `results/bifurcation.json`.
3. `src/verify_refs.py` – DOI verification of all references against Crossref/DataCite.
4. `src/run_baselines.py spline|odelsq`, `src/run_queue.py` (PINN queue, resumable, several workers allowed:
   `PINN_THREADS=1 python3 src/run_queue.py`), `src/run_param.py` (parametric network, generalization across currents).
5. `src/stage5_entropy_ref.py`, `src/evaluate.py`, `src/stage5_entropy_rec.py` – entropies and reconstruction metrics.
6. `src/stage6_benchmark.py` – localization errors, statistics, hypothesis verdicts → `results/benchmark/`.
7. `src/figures.py`, `src/supplementary.py`.

Seeds are deterministic functions of the condition and realization index (Supplementary Table S2). All runs that were
performed are included; none was discarded.

## Role of the PINN

The PINN solves an inverse problem (reconstruction of V(t) between sparse samples and of the never-observed recovery
variable w(t) with known parameters and current). It is not proposed as a replacement for an ODE integrator.

## License

MIT (see `LICENSE`). Please cite using `CITATION.cff`. A Zenodo DOI will be added upon archiving.

## Figure file names

`src/figures.py` writes `fig1_pipeline` (Figure 1), `fig2_bifurcation` (Figure 2), `fig4_entropy_curves` (Figure 3),
`fig7_sensitivity` (Figure 4), `fig3_reconstruction` (Figure 5), `fig5_eps_sparsity_noise` (Figure 6),
`fig6_run_stability` (Figure 7) and `fig8_parametric` (Figure 8 ).
