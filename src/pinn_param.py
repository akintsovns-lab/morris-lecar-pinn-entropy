"""Parametric PINN (t, I) -> (V, w): current-generalization experiment (separate from the main benchmark).

The window [0, T] starts from the common holding initial condition (V0 = -60 mV, w0 = w_inf(V0)).
Time is decomposed into overlapping segments; the network of segment k maps (tau, iota) with
iota = (I - 40)/10 to (V, w). Data (sparse V) and collocation points exist only at TRAINING currents;
test currents are never seen in any loss term.
"""
from __future__ import annotations
import math, time
import numpy as np
import torch
from pinn import BatchedMLP, _ml_terms, PAR, NET_DTYPE


class ParamMLP(BatchedMLP):
    def __init__(self, B, width, depth, seed):
        super().__init__(B, width, depth, 0, seed, dout=1)
        # replace first layer by a 2-input layer
        g = torch.Generator().manual_seed(int(seed) + 1)
        std = math.sqrt(2.0 / (2 + width))
        self.W[0] = torch.nn.Parameter((torch.randn(B, 2, width, generator=g, dtype=torch.float64) * std).to(NET_DTYPE))

    def forward(self, tau, iota, need_grad=True):
        x = torch.cat([tau, iota], -1).to(self.W[0].dtype)
        dx = torch.cat([torch.ones_like(tau), torch.zeros_like(iota)], -1).to(self.W[0].dtype)
        h, dh = x, dx
        n = len(self.W)
        for i in range(n):
            z = torch.baddbmm(self.b[i], h, self.W[i])
            dz = torch.bmm(dh, self.W[i]) if need_grad else None
            if i < n - 1:
                h = torch.tanh(z)
                if need_grad:
                    dh = (1 - h * h) * dz
            else:
                h, dh = z, dz
        return h.double(), (dh.double() if dh is not None else None)


class ParamPINN:
    def __init__(self, I_train, t_obs, V_obs, T=500.0, seg_len=50.0, overlap=0.2, width=32, depth=3,
                 n_col_per_ms=1.0, V0=-60.0, w0=None, Vc=-30.0, Vs=25.0, seed=0, p=PAR):
        self.p, self.Vc, self.Vs = p, Vc, Vs
        self.I_train = np.asarray(I_train, float)
        nS = int(round(T / seg_len)); edges = np.linspace(0, T, nS + 1); d = overlap * seg_len
        self.edges = edges
        self.xa = np.maximum(0, edges[:-1] - d); self.xb = np.minimum(T, edges[1:] + d)
        self.B = nS
        self.netV = ParamMLP(nS, width, depth, seed); self.netW = ParamMLP(nS, width, depth, seed + 7919)
        self.L = torch.tensor(self.xb - self.xa)[:, None]
        nI = len(I_train)
        self.iota_of = lambda I: (np.asarray(I) - 40.0) / 10.0
        # data per segment (padded)
        TT, VV, II = [], [], []
        for k in range(nS):
            ts, vs, is_ = [], [], []
            for j, I in enumerate(I_train):
                sel = (t_obs[j] >= self.xa[k]) & (t_obs[j] <= self.xb[k])
                ts.append(t_obs[j][sel]); vs.append(V_obs[j][sel]); is_.append(np.full(sel.sum(), I))
            TT.append(np.concatenate(ts)); VV.append(np.concatenate(vs)); II.append(np.concatenate(is_))
        nmax = max(len(a) for a in TT)
        def pad(arrs, fill):
            out = np.full((nS, nmax), fill, float); M = np.zeros((nS, nmax))
            for k, a in enumerate(arrs):
                out[k, :len(a)] = a; M[k, :len(a)] = 1
            return out, M
        T_, M = pad(TT, 0.0); V_, _ = pad(VV, 0.0); I_, _ = pad(II, 40.0)
        for k in range(nS):
            T_[k, M[k] == 0] = self.xa[k]
        self.tau_obs = torch.tensor(2 * (T_ - self.xa[:, None]) / (self.xb - self.xa)[:, None] - 1)[..., None]
        self.iota_obs = torch.tensor(self.iota_of(I_))[..., None]
        self.V_obs = torch.tensor(V_); self.M = torch.tensor(M); self.nobs = self.M.sum(1).clamp(min=1)
        self.n_col = int(round(n_col_per_ms * (self.xb[0] - self.xa[0] + 2 * d) if False else n_col_per_ms * seg_len * (1 + 2 * overlap)))
        self.Icol = np.repeat(self.I_train, self.n_col)
        self.iota_col = torch.tensor(self.iota_of(self.Icol))[None, :, None].expand(nS, -1, -1)
        self.I_col = torch.tensor(self.Icol)[None, :]
        self.rng = np.random.default_rng(seed + 99)
        # interfaces: right nominal edge of k vs left nominal edge of k+1, at every training current
        self.tau_hi = torch.tensor(2 * (edges[1:] - self.xa) / (self.xb - self.xa) - 1)[:, None, None].expand(-1, nI, -1)
        self.tau_lo = torch.tensor(2 * (edges[:-1] - self.xa) / (self.xb - self.xa) - 1)[:, None, None].expand(-1, nI, -1)
        self.iota_tr = torch.tensor(self.iota_of(self.I_train))[None, :, None].expand(nS, -1, -1)
        self.V0 = V0; self.w0 = w0
        self.history = []
        self.resample()

    def resample(self):
        nI = len(self.I_train); n = self.n_col
        u = (np.tile(np.arange(n), nI)[None, :] + self.rng.random((self.B, n * nI))) / n
        self.tau_col = torch.tensor(2 * u - 1)[..., None]

    def _state(self, tau, iota, need_grad=True):
        oV, dV = self.netV(tau, iota, need_grad); oW, dW = self.netW(tau, iota, need_grad)
        V = self.Vc + self.Vs * oV[..., 0]; w = torch.sigmoid(oW[..., 0])
        if not need_grad:
            return V, w, None, None
        return V, w, self.Vs * dV[..., 0] * 2 / self.L, w * (1 - w) * dW[..., 0] * 2 / self.L

    def terms(self):
        p = self.p
        V, w, dV, dw = self._state(self.tau_col, self.iota_col)
        Iion, winf, inv_tau = _ml_terms(V, w, self.I_col)
        rV = (p["C"] * dV + Iion - self.I_col) / (p["C"] * self.Vs / 10.0)
        rw = (dw - p["phi"] * (winf - w) * inv_tau) / p["phi"]
        Vo, _, _, _ = self._state(self.tau_obs, self.iota_obs, need_grad=False)
        Ld = (((Vo - self.V_obs) / self.Vs) ** 2 * self.M).sum(1) / self.nobs
        Vr, wr, dVr, _ = self._state(self.tau_hi, self.iota_tr); Vl, wl, dVl, _ = self._state(self.tau_lo, self.iota_tr)
        jump = (((Vr[:-1] - Vl[1:]) / self.Vs) ** 2 + ((wr[:-1] - wl[1:]) / 0.1) ** 2
                + ((dVr[:-1] - dVl[1:]) / (self.Vs / 10)) ** 2).mean(1)
        Lic = torch.zeros(self.B); Lic[:-1] += 0.5 * jump; Lic[1:] += 0.5 * jump
        # known initial condition at t = 0 (segment 0, tau = -1), all training currents
        Vi, wi, _, _ = self._state(torch.full((1, len(self.I_train), 1), -1.0).expand(self.B, -1, -1), self.iota_tr, False)
        Lic[0] += (((Vi[0] - self.V0) / self.Vs) ** 2 + ((wi[0] - self.w0) / 0.01) ** 2).mean()
        return torch.stack([Ld, (rV ** 2).mean(1), (rw ** 2).mean(1), Lic], 1)

    def train(self, n_warmup=1000, n_wfit=500, n_adam=3000, n_lbfgs=200, lr=3e-3, resample_every=500,
              weight_every=100, verbose=False):
        t0 = time.time()
        pV, pW = list(self.netV.parameters()), list(self.netW.parameters())
        opt = torch.optim.Adam(pV, lr=lr)
        for it in range(n_warmup):
            opt.zero_grad(); T = self.terms(); (T[:, 0] + T[:, 3]).sum().backward(); opt.step()
        opt = torch.optim.Adam(pW, lr=lr)
        for it in range(n_wfit):
            opt.zero_grad(); T = self.terms(); (T[:, 1] + T[:, 2] + T[:, 3]).sum().backward(); opt.step()
        lam = torch.ones(self.B, 4)
        opt = torch.optim.Adam(pV + pW, lr=lr)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1 / n_adam))
        params = pV + pW
        for it in range(n_adam):
            if it % resample_every == 0 and it > 0:
                self.resample()
            if it % weight_every == 0:
                T = self.terms(); N = []
                for i in range(4):
                    g = torch.autograd.grad(T[:, i].sum(), params, retain_graph=i < 3, allow_unused=True)
                    sq = torch.zeros(self.B)
                    for gi in g:
                        if gi is not None:
                            sq = sq + (gi ** 2).sum(dim=tuple(range(1, gi.dim())))
                    N.append(sq.sqrt() + 1e-12)
                N = torch.stack(N, 1); new = N.sum(1, keepdim=True) / N; new = new / new.mean(1, keepdim=True)
                new[:, 0] = new[:, 0].clamp(min=1.0)
                lam = new.detach() if it == 0 else 0.9 * lam + 0.1 * new.detach()
            opt.zero_grad(); T = self.terms(); (lam * T).sum().backward(); opt.step(); sch.step()
            if it % 500 == 0 or it == n_adam - 1:
                rec = dict(stage="adam", it=it, t=time.time() - t0, **{k: float(T[:, i].mean()) for i, k in
                                                                         enumerate(["data", "resV", "resw", "ic"])})
                self.history.append(rec)
                if verbose:
                    print(rec, flush=True)
        if n_lbfgs:
            self.resample()
            lb = torch.optim.LBFGS(params, lr=1.0, max_iter=n_lbfgs, history_size=50, line_search_fn="strong_wolfe",
                                   tolerance_grad=1e-12, tolerance_change=1e-14)
            def closure():
                lb.zero_grad(); l = (lam * self.terms()).sum(); l.backward(); return l
            try:
                lb.step(closure)
            except Exception as e:  # pragma: no cover
                self.history.append(dict(stage="lbfgs_error", msg=str(e)))
            T = self.terms()
            self.history.append(dict(stage="lbfgs", t=time.time() - t0, **{k: float(T[:, i].mean()) for i, k in
                                                                            enumerate(["data", "resV", "resw", "ic"])}))
        self.train_time = time.time() - t0

    @torch.no_grad()
    def predict(self, I, t):
        V = np.full(len(t), np.nan); w = np.full(len(t), np.nan)
        for k in range(self.B):
            last = k == self.B - 1
            sel = (t >= self.edges[k] - 1e-9) & ((t < self.edges[k + 1] - 1e-9) | (last & (t <= self.edges[k + 1] + 1e-9)))
            tau = torch.tensor(2 * (t[sel] - self.xa[k]) / (self.xb[k] - self.xa[k]) - 1)[None, :, None]
            io = torch.full_like(tau, float(self.iota_of(I)))
            # evaluate only network k
            sub = lambda net: _eval_k(net, k, tau, io)
            V[sel] = self.Vc + self.Vs * sub(self.netV)
            w[sel] = 1 / (1 + np.exp(-sub(self.netW)))
        return V, w


def _eval_k(net, k, tau, io):
    h = torch.cat([tau, io], -1).to(net.W[0].dtype)
    for i in range(len(net.W)):
        h = torch.baddbmm(net.b[i][k:k + 1], h, net.W[i][k:k + 1])
        if i < len(net.W) - 1:
            h = torch.tanh(h)
    return h.double()[0, :, 0].numpy()
