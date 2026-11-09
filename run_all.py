"""Regenerate every result and figure from scratch.

    python3 run_all.py            # full pipeline (CPU; several hours, PINN queue dominates)
    python3 run_all.py --from 5   # restart at stage 5 (uses existing results/)
    python3 run_all.py --quick    # unit tests + stages that take minutes (no PINN training)

Every stage is resumable: finished outputs are skipped. Progress is written to STATE.json and results/logs/.
"""
import os, sys, subprocess, time
ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STAGES = [
    (0, "unit tests", [[PY, "-m", "pytest", "-q", "tests"]]),
    (1, "reference solutions + verification", [[PY, "src/stage1_reference.py"], [PY, "src/stage1_verify.py"],
                                                 [PY, "src/reference_summary.py"]]),
    (2, "bifurcation analysis", [[PY, "src/stage2_bifurcation.py"]]),
    (3, "verification of literature DOIs", [[PY, "src/verify_refs.py"]]),
    (4, "sparse data, PINN and baselines", [[PY, "src/run_baselines.py", "spline"], [PY, "src/run_baselines.py", "odelsq"],
                                             [PY, "src/run_queue.py"], [PY, "src/run_param.py"]]),
    (5, "entropies and reconstruction metrics", [[PY, "src/stage5_entropy_ref.py"], [PY, "src/evaluate.py"],
                                                  [PY, "src/stage5_entropy_rec.py"]]),
    (6, "benchmark and hypotheses", [[PY, "src/stage6_benchmark.py"], [PY, "src/param_summary.py"],
                                      [PY, "src/snapshot_config.py"], [PY, "src/manuscript_numbers.py"]]),
    (7, "figures and supplementary", [[PY, "src/figures.py"], [PY, "src/supplementary.py"]]),
]
QUICK = {0, 2, 3, 6, 7, 8}


def main():
    start = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
    quick = "--quick" in sys.argv
    for k, name, cmds in STAGES:
        if k < start or (quick and k not in QUICK):
            continue
        print(f"=== stage {k}: {name}", flush=True)
        for c in cmds:
            t0 = time.time()
            r = subprocess.run(c, cwd=ROOT)
            print(f"    {' '.join(c[1:])}: exit {r.returncode} ({time.time() - t0:.0f} s)", flush=True)
            if r.returncode != 0:
                sys.exit(f"stage {k} failed: {' '.join(c)}")


if __name__ == "__main__":
    main()
