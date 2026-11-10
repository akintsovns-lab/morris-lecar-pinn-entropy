"""Run one realization (scenario A/B/C, mask/noise/initialization seeds) for all PINN currents.

Usage: python src/train.py --scenario B --rho 0.05 --rep 0 [--eta 0.0] [--variant pinn|dataonly|noadapt]
Outputs results/pinn/<tag>.npz (reconstructions on the 0.1 ms grid, float32) and <tag>.json (metadata).
"""
from __future__ import annotations
import os, sys, json, time, argparse, hashlib
import numpy as np, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
PC = yaml.safe_load(open(os.path.join(ROOT, "configs", "pinn.yaml")))
os.environ.setdefault("PINN_NET_DTYPE", PC["net_dtype"])
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("PINN_THREADS", PC["threads"])))
from grids import pinn_currents  # noqa: E402
from sparse import cfg as scfg, current_split, make_mask, val_blocks  # noqa: E402
from pinn import SegmentPINN  # noqa: E402

REF = os.path.join(ROOT, "results", "reference")
OUT = os.path.join(ROOT, "results", "pinn")


def load_ref(I):
    d = np.load(os.path.join(REF, f"ref_I{I:08.4f}.npz"))
    return d["t_dense"], d["V_dense"], d["w_dense"]


def sigma_V_train():
    s = current_split()
    return float(np.std(np.concatenate([load_ref(I)[1] for I in s["train"]])))


def observations(I, j, rho, eta, rep, kind="random"):
    """Observed (t, V) for current index j: mask -> noise -> blocked-validation split."""
    c = scfg()
    t, V, w = load_ref(I)
    mseed = c["mask_seed_base"] + 100000 * int(round(rho * 100)) + 100 * rep + j
    m = make_mask(len(t), rho, mseed, kind)
    Vo = V.copy()
    if eta > 0:
        nseed = c["noise_seed_base"] + 100000 * int(round(eta * 100)) + 1000 * int(round(rho * 100)) + 100 * rep + j
        Vo = V + eta * sigma_V_train() * np.random.default_rng(nseed).standard_normal(len(t))
    vb = val_blocks(t, c["val_block_ms"], c["val_block_period_ms"])
    tr = m & ~vb; tr[0] = True
    va = m & vb
    return dict(t_obs=t[tr], V_obs=Vo[tr], t_val=t[va], V_val=Vo[va], mask=m, Vnoisy=Vo)


def tag_of(scenario, rho, eta, rep, variant, kind="random", init=None):
    s = f"{scenario}_rho{rho:.2f}_eta{eta:.2f}_rep{rep:02d}_{variant}"
    if kind != "random":
        s += f"_{kind}"
    if init is not None:
        s += f"_init{init}"
    return s


def run(scenario, rho, eta, rep, variant="pinn", kind="random", init=None, verbose=False):
    os.makedirs(OUT, exist_ok=True)
    tag = tag_of(scenario, rho, eta, rep, variant, kind, init)
    fn = os.path.join(OUT, tag + ".npz")
    if os.path.exists(fn):
        return fn
    c = scfg()
    Is = pinn_currents()
    split = current_split()
    trajs, obs = [], []
    for j, I in enumerate(Is):
        o = observations(I, j, rho, eta, rep, kind)
        obs.append(o)
        trajs.append(dict(t0=0.0, t1=1000.0, I=float(I), t_obs=o["t_obs"], V_obs=o["V_obs"]))
    trainset = set(np.round(split["train"], 4))
    Vtr = np.concatenate([o["V_obs"] for I, o in zip(Is, obs) if round(I, 4) in trainset])
    Vc, Vs = float(Vtr.mean()), float(Vtr.std())
    seed = c["init_seed_base"] + (init if init is not None else rep)
    torch.manual_seed(seed)
    net = SegmentPINN(trajs, seg_len=PC["seg_len"], width=PC["width"], depth=PC["depth"],
                      n_fourier=PC["n_fourier"], n_col_per_ms=PC["n_col_per_ms"], Vc=Vc, Vs=Vs, seed=seed,
                      lam_reg=PC["lam_reg"], adaptive=(variant != "noadapt"), lam_data_min=PC["lam_data_min"],
                      overlap=PC["overlap"])
    t0 = time.time()
    if variant == "dataonly":
        net.train_data_only(n_iter=PC["n_warmup"] + PC["n_wfit"] + PC["n_adam"], lr=PC["lr"], verbose=verbose)
    else:
        net.train(n_adam=PC["n_adam"], lr=PC["lr"], n_lbfgs=PC["n_lbfgs"], resample_every=PC["resample_every"],
                  weight_every=PC["weight_every"], verbose=verbose, n_warmup=PC["n_warmup"], n_wfit=PC["n_wfit"])
    wall = time.time() - t0
    tg = load_ref(Is[0])[0]
    Vh, wh = net.predict(tg)
    if variant == "dataonly":
        wh[:] = np.nan
        RV = RW = np.full_like(Vh, np.nan)
    else:
        RV, RW = net.residuals_on(tg)
    val_rmse = []
    for j, o in enumerate(obs):
        if len(o["t_val"]):
            idx = np.round(o["t_val"] / 0.1).astype(int)
            val_rmse.append(float(np.sqrt(np.mean((Vh[j, idx] - o["V_val"]) ** 2))))
        else:
            val_rmse.append(float("nan"))
    np.savez_compressed(fn, I=Is, t=tg, V=Vh.astype(np.float32), w=wh.astype(np.float32),
                        resV_rms=np.sqrt(np.nanmean(RV ** 2, 1)), resw_rms=np.sqrt(np.nanmean(RW ** 2, 1)),
                        val_rmse=np.array(val_rmse), n_obs=np.array([len(o["t_obs"]) for o in obs]))
    meta = dict(tag=tag, scenario=scenario, rho=rho, eta=eta, rep=rep, variant=variant, kind=kind, init=init,
                seed=seed, Vc=Vc, Vs=Vs, wall_s=wall, n_segments=net.B, pinn_cfg=PC,
                history=net.history[-12:], sha256=hashlib.sha256(open(fn, "rb").read()).hexdigest())
    json.dump(meta, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    return fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="B"); ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=0.0); ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--variant", default="pinn"); ap.add_argument("--kind", default="random")
    ap.add_argument("--init", type=int, default=None); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    print(run(a.scenario, a.rho, a.eta, a.rep, a.variant, a.kind, a.init, a.verbose))
