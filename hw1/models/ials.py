"""Implicit ALS (Hu, Koren, Volinsky 2008) with frequency-scaled regularisation (Rendle et al. 2022)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .common import get_device


class _Side:

    def __init__(self, P: sp.csr_matrix, C: sp.csr_matrix, device):
        P = P.tocsr(); C = C.tocsr()
        assert (P.indptr == C.indptr).all() and (P.indices == C.indices).all()
        coo = P.tocoo()
        self.n = P.shape[0]
        self.rows = torch.tensor(coo.row, dtype=torch.long, device=device)
        self.cols = torch.tensor(coo.col, dtype=torch.long, device=device)
        self.p = torch.tensor(P.data, dtype=torch.float32, device=device)
        self.c = torch.tensor(C.data, dtype=torch.float32, device=device)
        self.count = torch.tensor(np.diff(P.indptr), dtype=torch.float32, device=device)


class WeightedALS:

    def __init__(self, factors: int = 128, reg: float = 1e-2, reg_exp: float = 0.0, w0: float = 1.0,
                 iterations: int = 15, cg_steps: int = 3, init_std: float = 0.01, chunk: int = 500_000,
                 seed: int = 42, device: str | None = None, verbose: bool = False):
        self.factors, self.reg, self.reg_exp, self.w0 = int(factors), reg, reg_exp, w0
        self.iterations, self.cg_steps, self.init_std, self.chunk = int(iterations), int(cg_steps), init_std, chunk
        self.seed, self.device, self.verbose = seed, get_device(device), verbose

    def _matvec(self, V, side: _Side, other, gram, lam):
        out = self.w0 * (V @ gram) + lam[:, None] * V
        for s in range(0, len(side.rows), self.chunk):
            r = side.rows[s:s + self.chunk]; c = side.cols[s:s + self.chunk]
            Yc = other[c]
            dots = (V[r] * Yc).sum(1)
            out.index_add_(0, r, ((side.c[s:s + self.chunk] - self.w0) * dots)[:, None] * Yc)
        return out

    def _solve(self, X, side: _Side, other, n_other: int):
        gram = other.T @ other
        lam = self.reg * (side.count + self.w0 * n_other) ** self.reg_exp
        b = torch.zeros_like(X)
        for s in range(0, len(side.rows), self.chunk):
            r = side.rows[s:s + self.chunk]; c = side.cols[s:s + self.chunk]
            b.index_add_(0, r, (side.c[s:s + self.chunk] * side.p[s:s + self.chunk])[:, None] * other[c])
        r = b - self._matvec(X, side, other, gram, lam)
        p = r.clone()
        rs = (r * r).sum(1)
        for _ in range(self.cg_steps):
            Ap = self._matvec(p, side, other, gram, lam)
            a = rs / (p * Ap).sum(1).clamp_min(1e-20)
            X = X + a[:, None] * p
            r = r - a[:, None] * Ap
            rs_new = (r * r).sum(1)
            p = r + (rs_new / rs.clamp_min(1e-20))[:, None] * p
            rs = rs_new
        return X

    def fit_weighted(self, P: sp.csr_matrix, C: sp.csr_matrix):
        dev = self.device
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        n_users, n_items = P.shape
        users = _Side(P, C, dev)
        items = _Side(P.T.tocsr(), C.T.tocsr(), dev)
        X = (torch.randn(n_users, self.factors, generator=g) * self.init_std).to(dev)
        Y = (torch.randn(n_items, self.factors, generator=g) * self.init_std).to(dev)
        with torch.no_grad():
            for it in range(self.iterations):
                X = self._solve(X, users, Y, n_items)
                Y = self._solve(Y, items, X, n_users)
                if self.verbose:
                    print(f"iter {it + 1}/{self.iterations}", flush=True)
        self.user_factors, self.item_factors = X, Y
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        u = torch.as_tensor(np.asarray(users), dtype=torch.long, device=self.device)
        return (self.user_factors[u] @ self.item_factors.T).cpu().numpy()


class IALS(WeightedALS):

    def __init__(self, alpha: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def fit(self, R: sp.csr_matrix):
        R = R.tocsr().astype(np.float32)
        P = R.copy(); P.data[:] = 1.0
        C = R.copy(); C.data = 1.0 + self.alpha * R.data
        return self.fit_weighted(P, C)
