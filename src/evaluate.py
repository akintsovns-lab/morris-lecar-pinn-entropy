"""Compute reconstruction metrics for every stored realization -> results/metrics/<tag>.json"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from metrics import metrics  # noqa: E402

REF = os.path.join(ROOT, "results", "reference"); OUT = os.path.join(ROOT, "results", "metrics")
_cache = {}


def ref(I):
    k = round(float(I), 4)
    if k not in _cache:
        d = np.load(os.path.join(REF, f"ref_I{k:08.4f}.npz")); _cache[k] = (d["V_dense"], d["w_dense"])
    return _cache[k]


def evaluate(fn):
    tag = os.path.basename(fn)[:-4]
    out = os.path.join(OUT, tag + ".json")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(fn):
        return json.load(open(out))
    d = np.load(fn)
    res = {}
    for j, I in enumerate(d["I"]):
        Vr, wr = ref(I)
        Vh = d["V"][j].astype(float); wh = d["w"][j].astype(float)
        m = metrics(d["t"], Vh, Vr, wh if np.all(np.isfinite(wh)) else None, wr)
        m["val_rmse"] = float(d["val_rmse"][j])
        if "resV_rms" in d:
            m["resV_rms"] = float(d["resV_rms"][j]); m["resw_rms"] = float(d["resw_rms"][j])
        res[f"{I:.4f}"] = m
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(out, "w"), indent=0)
    return res


if __name__ == "__main__":
    for fn in sorted(glob.glob(os.path.join(ROOT, "results", "pinn", "*.npz"))):
        if not os.path.exists(fn[:-4] + ".json"):   # still being written by a running worker
            continue
        evaluate(fn)
