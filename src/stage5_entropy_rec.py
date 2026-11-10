"""Stage 5b: entropies of every reconstruction (1000-ms window, 1-ms sampling) -> results/entropy/rec/<tag>.json"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from entropy import all_entropies, MEASURES  # noqa: E402

OUT = os.path.join(ROOT, "results", "entropy", "rec"); os.makedirs(OUT, exist_ok=True)


def entropies_of(fn, floors=(0.1,)):
    tag = os.path.basename(fn)[:-4]
    out = os.path.join(OUT, tag + ".json")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(fn):
        return json.load(open(out))
    d = np.load(fn)
    res = {}
    for j, I in enumerate(d["I"]):
        x = d["V"][j].astype(np.float64)[::10]
        e = all_entropies(x)
        res[f"{I:.4f}"] = {k: e[k] for k in MEASURES} | dict(sd=e["sd"], quiescent=e["quiescent"],
                                                              sampen_bound=e["sampen_bound"])
    json.dump(res, open(out, "w"))
    return res


if __name__ == "__main__":
    for fn in sorted(glob.glob(os.path.join(ROOT, "results", "pinn", "*.npz"))):
        if not os.path.exists(fn[:-4] + ".json"):   # still being written by a running worker
            continue
        entropies_of(fn)
