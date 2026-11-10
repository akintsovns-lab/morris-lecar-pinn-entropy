"""Stage 1: reference sweep over all currents (parallel), stored in results/reference/."""
import os, sys, json, time
import numpy as np, yaml
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import reference_trajectory, load_cfg
from grids import reference_currents, pinn_currents

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "reference")


def job(I):
    fn = os.path.join(OUT, f"ref_I{I:08.4f}.npz")
    if os.path.exists(fn):
        return fn
    t = time.time()
    keep = 1000.0 if any(abs(I - J) < 1e-9 for J in pinn_currents()) else 0.0
    r = reference_trajectory(I, keep_dense=keep)
    r["wall"] = time.time() - t
    info = r.pop("info")
    np.savez_compressed(fn, **{k: v for k, v in r.items()}, info=json.dumps(info))
    return fn


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    Is = reference_currents()
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    t0 = time.time()
    with Pool(nproc) as pool:
        for k, fn in enumerate(pool.imap_unordered(job, Is[::-1])):
            if k % 20 == 0:
                print(k, len(Is), fn, round(time.time() - t0, 1), flush=True)
    print("done", time.time() - t0)
