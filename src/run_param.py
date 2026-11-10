"""Current-generalization experiment with the parametric PINN (t, I) -> (V, w) from the holding IC."""
import os, sys, json, time
import numpy as np, yaml
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
os.environ.setdefault("PINN_NET_DTYPE", "float32")
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("PINN_THREADS", 2)))
from sparse import current_split, make_mask  # noqa: E402
from reference import integrate  # noqa: E402
from ml_model import y0  # noqa: E402
from pinn_param import ParamPINN  # noqa: E402
from metrics import metrics, crossings  # noqa: E402

T = 500.0; RHO = 0.2
SEEDS = [int(x) for x in os.environ.get("PARAM_SEEDS", "0,1,2,3,4").split(",")]   # subsets allow parallel workers
OUT = os.path.join(ROOT, "results", "param"); os.makedirs(OUT, exist_ok=True)


def ref_from_ic(I, t):
    s = integrate(float(I), (0, T), y0(), "DOP853", 1e-10, (1e-10, 1e-12), t_eval=t)
    return s.y


def main():
    sp = current_split()
    t = np.round(np.arange(0, T + 1e-9, 0.1), 10)
    allI = np.sort(np.r_[sp["train"], sp["val"], sp["test"]])
    refs = {round(float(I), 4): ref_from_ic(I, t) for I in allI}
    for seed in SEEDS:
        fn = os.path.join(OUT, f"param_seed{seed}.json")
        if os.path.exists(fn):
            continue
        t_obs, V_obs = [], []
        for j, I in enumerate(sp["train"]):
            m = make_mask(len(t), RHO, 77000 + 100 * seed + j)
            t_obs.append(t[m]); V_obs.append(refs[round(float(I), 4)][0][m])
        Vall = np.concatenate(V_obs)
        torch.manual_seed(seed)
        net = ParamPINN(sp["train"], t_obs, V_obs, T=T, Vc=float(Vall.mean()), Vs=float(Vall.std()), seed=seed,
                        V0=float(y0()[0]), w0=float(y0()[1]))
        net.train(n_warmup=600, n_wfit=300, n_adam=1500, n_lbfgs=100, verbose=True)  # D-22
        res = dict(seed=seed, wall_s=net.train_time, history=net.history[-4:], per_current={})
        pred = {}
        for I in allI:
            Vh, wh = net.predict(float(I), t)
            Vr, wr = refs[round(float(I), 4)]
            m = metrics(t, Vh, Vr, wh, wr)
            cr, ch = crossings(t, Vr), crossings(t, Vh)
            m["first_spike_ref"] = float(cr[0]) if len(cr) else float("nan")
            m["first_spike_hat"] = float(ch[0]) if len(ch) else float("nan")
            role = "train" if I in sp["train"] else ("val" if I in sp["val"] else "test")
            res["per_current"][f"{I:.4f}"] = dict(role=role, **m)
            pred[f"{I:.4f}"] = np.vstack([Vh, wh]).astype(np.float32)
        np.savez_compressed(fn[:-5] + ".npz", t=t, **pred)
        json.dump(res, open(fn, "w"), indent=1)
        print("seed", seed, "done", net.train_time, flush=True)


if __name__ == "__main__":
    main()
