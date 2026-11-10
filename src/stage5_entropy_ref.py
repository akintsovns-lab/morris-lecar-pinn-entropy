"""Stage 5a: entropies of the reference signals (short 1000-ms window and long 20 000-ms record),
bootstrap CIs over windows, and parameter-sensitivity grid."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from entropy import all_entropies, windowed, MEASURES  # noqa: E402
from grids import reference_currents  # noqa: E402

OUT = os.path.join(ROOT, "results", "entropy"); os.makedirs(OUT, exist_ok=True)
SENS = ([("SampEn", dict(m=m, r_frac=r)) for m in (2, 3) for r in (0.10, 0.15, 0.20, 0.25)] +
        [("PE", dict(d=d, tau=t)) for d in (3, 4, 5, 6) for t in (1, 2, 4, 8)] +
        [("WPE", dict(d=d, tau=t)) for d in (3, 4, 5, 6) for t in (1, 2, 4, 8)] +
        [("SpEn", dict(nperseg=n, noverlap=n // 2)) for n in (128, 256, 512)])


def main():
    Is = reference_currents()
    rng = np.random.default_rng(2024)
    res = dict(I=Is.tolist(), short={k: [] for k in MEASURES}, long_mean={k: [] for k in MEASURES},
               long_ci_lo={k: [] for k in MEASURES}, long_ci_hi={k: [] for k in MEASURES},
               full={k: [] for k in MEASURES}, short500={k: [] for k in MEASURES}, sd_short=[], quiescent=[],
               sampen_bound=[], sens={f"{k}|{json.dumps(p, sort_keys=True)}": [] for k, p in SENS})
    t0 = time.time()
    for I in Is:
        V = np.load(os.path.join(ROOT, "results", "reference", f"ref_I{I:08.4f}.npz"))["V"]
        s = all_entropies(V[:1001]); s5 = all_entropies(V[:501])
        W = windowed(V[:20000], 1000, 500)
        Hw = {k: np.array([w[k] for w in W]) for k in MEASURES}
        f = all_entropies(V)
        for k in MEASURES:
            res["short"][k].append(s[k]); res["short500"][k].append(s5[k]); res["full"][k].append(f[k])
            res["long_mean"][k].append(float(Hw[k].mean()))
            bs = [rng.choice(Hw[k], len(Hw[k]), replace=True).mean() for _ in range(1000)]
            res["long_ci_lo"][k].append(float(np.percentile(bs, 2.5))); res["long_ci_hi"][k].append(float(np.percentile(bs, 97.5)))
        res["sd_short"].append(s["sd"]); res["quiescent"].append(bool(s["quiescent"])); res["sampen_bound"].append(bool(s["sampen_bound"]))
        for k, p in SENS:
            res["sens"][f"{k}|{json.dumps(p, sort_keys=True)}"].append(all_entropies(V[:1001], p)[k])
    res["wall_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "reference_entropy.json"), "w"))
    print("done", res["wall_s"])


if __name__ == "__main__":
    main()
