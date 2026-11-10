"""Batched temporal-domain-decomposition PINN for the Morris-Lecar inverse problem.

Many small, independent tanh MLPs (one per time segment of each trajectory) are trained
simultaneously with batched matrix products. Each network maps the local normalized time
tau in [-1, 1] of its segment to (V_hat, w_hat). Time derivatives are obtained by exact
forward-mode differentiation through the network (equivalent to automatic differentiation in t).
Adjacent segments of one trajectory are coupled through interface-continuity losses.
All computations are carried out in float64.

Loss per network b:  L = lam_data*L_data + lam_V*L_V + lam_w*L_w + lam_IC*L_IC + lam_reg*L_reg
(+ interface continuity, counted in L_IC as the internal "initial condition" of each segment).
Adaptive weights: gradient-norm balancing (learning-rate annealing type), per network.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch

import os as _os
# Network dtype: float64 by default; PINN_NET_DTYPE=float32 runs the MLP in float32 while the
# Morris-Lecar residuals are always evaluated in float64 (see _ml_terms casts in loss_terms).
NET_DTYPE = torch.float32 if _os.environ.get("PINN_NET_DTYPE", "float64") == "float32" else torch.float64
torch.set_default_dtype(torch.float64)

PAR = dict(C=20.0, gCa=4.0, gK=8.0, gL=2.0, VCa=120.0, VK=-84.0, VL=-60.0,
           V1=-1.2, V2=18.0, V3=12.0, V4=17.4, phi=1.0 / 15.0)


def _ml_terms(V, w, I, p=PAR):
    minf = 0.5 * (1 + torch.tanh((V - p["V1"]) / p["V2"]))
    winf = 0.5 * (1 + torch.tanh((V - p["V3"]) / p["V4"]))
    inv_tau = torch.cosh((V - p["V3"]) / (2 * p["V4"]))
    Iion = p["gCa"] * minf * (V - p["VCa"]) + p["gK"] * w * (V - p["VK"]) + p["gL"] * (V - p["VL"])
    return Iion, winf, inv_tau


class BatchedMLP(torch.nn.Module):
    def __init__(self, B, width=32, depth=2, n_fourier=0, seed=0, dout=2):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.nf = n_fourier
        din = 1 + 2 * n_fourier
        dims = [din] + [width] * depth + [dout]
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for a, c in zip(dims[:-1], dims[1:]):
            std = math.sqrt(2.0 / (a + c))  # Glorot normal
            self.W.append(torch.nn.Parameter((torch.randn(B, a, c, generator=g, dtype=torch.float64) * std).to(NET_DTYPE)))
            self.b.append(torch.nn.Parameter(torch.zeros(B, 1, c, dtype=NET_DTYPE)))

    def forward(self, tau, need_grad=True):
        """tau: (B, N, 1). Returns out (B,N,dout) and d out / d tau, cast to float64."""
        tau = tau.to(self.W[0].dtype)
        if self.nf > 0:
            k = torch.arange(1, self.nf + 1, dtype=tau.dtype) * math.pi
            s, c = torch.sin(tau * k), torch.cos(tau * k)
            x = torch.cat([tau, s, c], dim=-1)
            dx = torch.cat([torch.ones_like(tau), c * k, -s * k], dim=-1)
        else:
            x = tau
            dx = torch.ones_like(tau)
        h, dh = x, dx
        nL = len(self.W)
        for i in range(nL):
            z = torch.baddbmm(self.b[i], h, self.W[i])
            dz = torch.bmm(dh, self.W[i]) if need_grad else None
            if i < nL - 1:
                h = torch.tanh(z)
                if need_grad:
                    dh = (1 - h * h) * dz
            else:
                h, dh = z, dz
        return h.double(), (dh.double() if dh is not None else None)


class SegmentPINN:
    """Container that sets up segments for a list of trajectories and trains them jointly."""

    def __init__(self, trajs, seg_len=50.0, width=32, depth=2, n_fourier=0, n_col_per_ms=2.0,
                 Vc=-20.0, Vs=25.0, seed=0, lam_reg=1e-8, adaptive=True, lam_data_min=1.0, overlap=0.2,
                 p=PAR):
        """trajs: list of dicts with keys t0, t1, I, t_obs (1D), V_obs (1D), t_val, V_val (optional)."""
        self.p = p
        self.Vc, self.Vs = float(Vc), float(Vs)
        self.adaptive = adaptive
        self.lam_data_min = float(lam_data_min)
        self.lam_reg = lam_reg
        segs = []
        for j, tr in enumerate(trajs):
            n = max(1, int(round((tr["t1"] - tr["t0"]) / seg_len)))
            edges = np.linspace(tr["t0"], tr["t1"], n + 1)
            d = overlap * (edges[1] - edges[0])
            for k in range(n):
                segs.append(dict(traj=j, k=k, n=n, ta=edges[k], tb=edges[k + 1], I=tr["I"],
                                 xa=max(tr["t0"], edges[k] - d), xb=min(tr["t1"], edges[k + 1] + d)))
        self.segs = segs
        self.trajs = trajs
        B = len(segs)
        self.B = B
        self.netV = BatchedMLP(B, width, depth, n_fourier, seed, dout=1)
        self.netW = BatchedMLP(B, width, depth, n_fourier, seed + 7919, dout=1)
        # network domain = nominal segment extended by the overlap (overlapping domain decomposition)
        self.ta = torch.tensor([s["xa"] for s in segs])[:, None, None]
        self.tb = torch.tensor([s["xb"] for s in segs])[:, None, None]
        self.tau_lo = (2 * (torch.tensor([s["ta"] for s in segs]) - self.ta[:, 0, 0]) / (self.tb - self.ta)[:, 0, 0] - 1).view(-1, 1, 1)
        self.tau_hi = (2 * (torch.tensor([s["tb"] for s in segs]) - self.ta[:, 0, 0]) / (self.tb - self.ta)[:, 0, 0] - 1).view(-1, 1, 1)
        self.I = torch.tensor([s["I"] for s in segs])[:, None, None]
        self.L = self.tb - self.ta
        # observations per segment (padded)
        obs_t, obs_v = [], []
        for s in segs:
            tr = trajs[s["traj"]]
            sel = (tr["t_obs"] >= s["xa"]) & (tr["t_obs"] <= s["xb"])
            obs_t.append(tr["t_obs"][sel]); obs_v.append(tr["V_obs"][sel])
        nmax = max(1, max(len(a) for a in obs_t))
        T = np.zeros((B, nmax)); Vv = np.zeros((B, nmax)); M = np.zeros((B, nmax))
        for b in range(B):
            m = len(obs_t[b])
            T[b, :m] = obs_t[b]; Vv[b, :m] = obs_v[b]; M[b, :m] = 1.0
            if m < nmax:
                T[b, m:] = segs[b]["xa"]
        self.tau_obs = (2 * (torch.tensor(T)[..., None] - self.ta) / self.L - 1)
        self.V_obs = torch.tensor(Vv)
        self.M_obs = torch.tensor(M)
        self.n_obs = self.M_obs.sum(1).clamp(min=1.0)
        # interface pairs: segment b and b+1 of the same trajectory
        nxt = [b for b in range(B - 1) if segs[b]["traj"] == segs[b + 1]["traj"]]
        self.if_left = torch.tensor(nxt, dtype=torch.long)
        self.if_right = self.if_left + 1
        first = [b for b in range(B) if segs[b]["k"] == 0]
        self.first = torch.tensor(first, dtype=torch.long)
        # IC target: first observation of each trajectory (retained by protocol)
        self.ic_V = torch.tensor([trajs[segs[b]["traj"]]["V_obs"][0] for b in first])
        self.ic_tau = torch.tensor([2 * (trajs[segs[b]["traj"]]["t_obs"][0] - segs[b]["xa"]) /
                                    (segs[b]["xb"] - segs[b]["xa"]) - 1 for b in first])
        self.n_col = max(8, int(round(n_col_per_ms * seg_len * (1 + 2 * overlap))))
        self.rng = np.random.default_rng(int(seed) + 12345)
        self.lam = torch.ones(B, 4)  # data, V, w, IC/interface
        self.history = []
        self.resample()

    # ------------------------------------------------------------------
    def resample(self):
        """Stratified (jittered) collocation sampling in each segment, regenerated periodically."""
        n = self.n_col
        u = (np.arange(n)[None, :] + self.rng.random((self.B, n))) / n
        self.tau_col = torch.tensor(2 * u - 1)[..., None]

    def _state(self, tau, need_grad=True):
        oV, dV = self.netV(tau, need_grad)
        oW, dW = self.netW(tau, need_grad)
        V = self.Vc + self.Vs * oV[..., 0]
        w = torch.sigmoid(oW[..., 0])
        if not need_grad:
            return V, w, None, None
        dVdt = self.Vs * dV[..., 0] * (2.0 / self.L[..., 0])
        dwdt = w * (1 - w) * dW[..., 0] * (2.0 / self.L[..., 0])
        return V, w, dVdt, dwdt

    def parameters(self):
        return list(self.netV.parameters()) + list(self.netW.parameters())

    def loss_terms(self):
        p = self.p
        V, w, dV, dw = self._state(self.tau_col)
        Iion, winf, inv_tau = _ml_terms(V, w, self.I[..., 0], p)
        rV = (p["C"] * dV + Iion - self.I[..., 0]) / (p["C"] * self.Vs / 10.0)
        rw = (dw - p["phi"] * (winf - w) * inv_tau) / p["phi"]
        LV = (rV ** 2).mean(1)
        Lw = (rw ** 2).mean(1)
        Vo, _, _, _ = self._state(self.tau_obs, need_grad=False)
        Ld = (((Vo - self.V_obs) / self.Vs) ** 2 * self.M_obs).sum(1) / self.n_obs
        # interface continuity + IC
        Lic = torch.zeros(self.B)
        if len(self.if_left) > 0:
            Vr, wr, dVr, dwr = self._state(self.tau_hi)   # value at right nominal edge
            Vl, wl, dVl, dwl = self._state(self.tau_lo)   # value at left nominal edge
            a, c = self.if_left, self.if_right
            jump = (((Vr[a, 0] - Vl[c, 0]) / self.Vs) ** 2 + ((wr[a, 0] - wl[c, 0]) / 0.1) ** 2
                    + ((dVr[a, 0] - dVl[c, 0]) / (self.Vs / 10.0)) ** 2)
            Lic = Lic.index_add(0, a, 0.5 * jump).index_add(0, c, 0.5 * jump)
        tau_ic = self.ic_tau.view(-1, 1, 1)
        # evaluate first segments at IC time
        out, _ = self.netV(torch.zeros(self.B, 1, 1).index_copy(0, self.first, tau_ic), need_grad=False)
        Vic = self.Vc + self.Vs * out[self.first, 0, 0]
        Lic = Lic.index_add(0, self.first, ((Vic - self.ic_V) / self.Vs) ** 2)
        return torch.stack([Ld, LV, Lw, Lic], dim=1)

    def data_ic_terms(self):
        Vo, _, _, _ = self._state_V(self.tau_obs)
        Ld = (((Vo - self.V_obs) / self.Vs) ** 2 * self.M_obs).sum(1) / self.n_obs
        tau_ic = self.ic_tau.view(-1, 1, 1)
        out, _ = self.netV(torch.zeros(self.B, 1, 1).index_copy(0, self.first, tau_ic), need_grad=False)
        Vic = self.Vc + self.Vs * out[self.first, 0, 0]
        Lic = torch.zeros(self.B).index_add(0, self.first, ((Vic - self.ic_V) / self.Vs) ** 2)
        return Ld, Lic

    def _state_V(self, tau):
        oV, _ = self.netV(tau, need_grad=False)
        return self.Vc + self.Vs * oV[..., 0], None, None, None

    def reg(self):
        return sum((W ** 2).sum(dim=(1, 2)) for W in list(self.netV.W) + list(self.netW.W))

    def _update_weights(self, alpha=0.9):
        """Gradient-norm balancing per network: lam_i = sum_j |g_j| / |g_i| (moving average)."""
        terms = self.loss_terms()
        params = self.parameters()
        norms = []
        for i in range(4):
            g = torch.autograd.grad(terms[:, i].sum(), params, retain_graph=(i < 3), allow_unused=True)
            sq = torch.zeros(self.B)
            for gi in g:
                if gi is not None:
                    sq = sq + (gi ** 2).sum(dim=tuple(range(1, gi.dim())))
            norms.append(torch.sqrt(sq) + 1e-12)
        N = torch.stack(norms, 1)
        new = N.sum(1, keepdim=True) / N
        new = new / new.mean(1, keepdim=True)
        new[:, 0] = new[:, 0].clamp(min=self.lam_data_min)  # never let the data weight fall below the floor
        if not getattr(self, "_lam_init", False):
            self.lam = new.detach(); self._lam_init = True
        else:
            self.lam = alpha * self.lam + (1 - alpha) * new.detach()

    def total(self):
        terms = self.loss_terms()
        return (self.lam * terms).sum() + self.lam_reg * self.reg().sum(), terms

    def _log(self, stage, it, loss, terms, t0, verbose):
        rec = dict(stage=stage, it=it, loss=float(loss), t=time.time() - t0,
                   **{k: float(terms[:, i].mean()) for i, k in enumerate(["data", "resV", "resw", "ic"])})
        self.history.append(rec)
        if verbose:
            print(rec, flush=True)

    def train(self, n_adam=3000, lr=3e-3, n_lbfgs=300, resample_every=500, weight_every=100,
              log_every=500, verbose=False, time_limit=None, n_warmup=1000, n_wfit=500):
        """Three-phase curriculum:
        (1) data warm-up of the V-network (data + IC terms only);
        (2) w-network fitted to the physics residuals with the V-network frozen;
        (3) joint Adam training of all terms with adaptive weights, followed by L-BFGS."""
        t0 = time.time()
        pV, pW = list(self.netV.parameters()), list(self.netW.parameters())
        opt = torch.optim.Adam(pV, lr=lr)
        for it in range(n_warmup):
            opt.zero_grad()
            Ld, Lic = self.data_ic_terms()
            loss = (Ld + Lic).sum() + self.lam_reg * self.reg().sum()
            loss.backward(); opt.step()
            if it % log_every == 0 and verbose:
                print(dict(stage="warmup", it=it, data=float(Ld.mean()), t=time.time() - t0), flush=True)
        opt = torch.optim.Adam(pW, lr=lr)
        for it in range(n_wfit):
            if it % resample_every == 0 and it > 0:
                self.resample()
            opt.zero_grad()
            terms = self.loss_terms()
            loss = (terms[:, 1] + terms[:, 2]).sum()
            loss.backward(); opt.step()
            if it % log_every == 0:
                self._log("wfit", it, loss.detach(), terms.detach(), t0, verbose)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(0.1) ** (1.0 / max(1, n_adam)))
        for it in range(n_adam):
            if it % resample_every == 0 and it > 0:
                self.resample()
            if self.adaptive and it % weight_every == 0:
                self._update_weights()
            opt.zero_grad()
            loss, terms = self.total()
            loss.backward()
            opt.step(); sched.step()
            if it % log_every == 0 or it == n_adam - 1:
                self._log("adam", it, loss.detach(), terms.detach(), t0, verbose)
            if time_limit and time.time() - t0 > time_limit:
                break
        if n_lbfgs > 0:
            self.resample()
            lb = torch.optim.LBFGS(self.parameters(), lr=1.0, max_iter=n_lbfgs, history_size=50,
                                   line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)

            def closure():
                lb.zero_grad()
                l, _ = self.total()
                l.backward()
                return l
            try:
                lb.step(closure)
            except Exception as e:  # pragma: no cover
                self.history.append(dict(stage="lbfgs_error", msg=str(e)))
            loss, terms = self.total()
            self._log("lbfgs", n_lbfgs, loss.detach(), terms.detach(), t0, verbose)
        self.train_time = time.time() - t0
        return self.history

    def train_data_only(self, n_iter=2400, lr=3e-3, log_every=500, verbose=False):
        """Ablation without physics: V-network fitted to data + IC + interface continuity of V and dV/dt
        (no residual terms, same architecture, matched number of first-order iterations)."""
        t0 = time.time()
        opt = torch.optim.Adam(list(self.netV.parameters()), lr=lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(0.1) ** (1.0 / max(1, n_iter)))
        a, c = self.if_left, self.if_right
        for it in range(n_iter):
            opt.zero_grad()
            Ld, Lic = self.data_ic_terms()
            oR, dR = self.netV(self.tau_hi); oL, dL = self.netV(self.tau_lo)
            VR = self.Vs * oR[..., 0]; VL = self.Vs * oL[..., 0]
            dVR = self.Vs * dR[..., 0] * (2.0 / self.L[..., 0]); dVL = self.Vs * dL[..., 0] * (2.0 / self.L[..., 0])
            jump = ((VR[a, 0] - VL[c, 0]) / self.Vs) ** 2 + ((dVR[a, 0] - dVL[c, 0]) / (self.Vs / 10.0)) ** 2
            loss = (Ld + Lic).sum() + jump.sum() + self.lam_reg * self.reg().sum()
            loss.backward(); opt.step(); sched.step()
            if it % log_every == 0 or it == n_iter - 1:
                self.history.append(dict(stage="dataonly", it=it, data=float(Ld.mean()), t=time.time() - t0))
                if verbose:
                    print(self.history[-1], flush=True)
        self.train_time = time.time() - t0
        return self.history

    @torch.no_grad()
    def predict(self, t_grid):
        """Evaluate every trajectory on the dense grid t_grid (absolute times); returns V, w arrays (n_traj, N)."""
        nT = len(self.trajs)
        Vout = np.full((nT, len(t_grid)), np.nan); wout = np.full((nT, len(t_grid)), np.nan)
        for b, s in enumerate(self.segs):
            last = s["k"] == s["n"] - 1
            tr = self.trajs[s["traj"]]
            sel = (t_grid >= s["ta"] - 1e-9) & ((t_grid < s["tb"] - 1e-9) | (last & (t_grid <= s["tb"] + 1e-9)))
            if not sel.any():
                continue
            tau = torch.tensor(2 * (t_grid[sel] - s["xa"]) / (s["xb"] - s["xa"]) - 1)[None, :, None]
            oV = self._eval_single(self.netV, b, tau); oW = self._eval_single(self.netW, b, tau)
            Vout[s["traj"], sel] = self.Vc + self.Vs * oV[0, :, 0].numpy()
            wout[s["traj"], sel] = torch.sigmoid(oW[0, :, 0]).numpy()
        return Vout, wout

    @staticmethod
    def _eval_single(net, b, tau):
        tau = tau.to(net.W[0].dtype)
        x = tau
        if net.nf > 0:
            k = torch.arange(1, net.nf + 1, dtype=tau.dtype) * math.pi
            x = torch.cat([tau, torch.sin(tau * k), torch.cos(tau * k)], -1)
        h = x
        for i in range(len(net.W)):
            h = torch.baddbmm(net.b[i][b:b + 1], h, net.W[i][b:b + 1])
            if i < len(net.W) - 1:
                h = torch.tanh(h)
        return h.double()

    @torch.no_grad()
    def residuals_on(self, t_grid):
        """Physics residuals in physical units (uA/cm^2 and 1/ms) on t_grid, per trajectory."""
        nT = len(self.trajs)
        RV = np.full((nT, len(t_grid)), np.nan); RW = np.full((nT, len(t_grid)), np.nan)
        for b, s in enumerate(self.segs):
            last = s["k"] == s["n"] - 1
            sel = (t_grid >= s["ta"] - 1e-9) & ((t_grid < s["tb"] - 1e-9) | (last & (t_grid <= s["tb"] + 1e-9)))
            if not sel.any():
                continue
            tau = torch.tensor(2 * (t_grid[sel] - s["xa"]) / (s["xb"] - s["xa"]) - 1)[None, :, None]
            L = s["xb"] - s["xa"]
            res = []
            for net in (self.netV, self.netW):
                sub = BatchedMLP(1, 2, 1, 0, 0, dout=1)
                sub.nf = net.nf
                sub.W = torch.nn.ParameterList([torch.nn.Parameter(W[b:b + 1].clone()) for W in net.W])
                sub.b = torch.nn.ParameterList([torch.nn.Parameter(c[b:b + 1].clone()) for c in net.b])
                res.append(sub(tau))
            (oV, dV), (oW, dW) = res
            V = self.Vc + self.Vs * oV[..., 0]; w = torch.sigmoid(oW[..., 0])
            dVt = self.Vs * dV[..., 0] * 2 / L; dwt = w * (1 - w) * dW[..., 0] * 2 / L
            Iion, winf, inv_tau = _ml_terms(V, w, torch.tensor(s["I"]), self.p)
            RV[s["traj"], sel] = (self.p["C"] * dVt + Iion - s["I"]).numpy()[0]
            RW[s["traj"], sel] = (dwt - self.p["phi"] * (winf - w) * inv_tau).numpy()[0]
        return RV, RW
