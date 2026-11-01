"""Item-based collaborative filtering (neighbourhood model)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .common import get_device


class ItemKNN:
    def __init__(self, k: int = 100, shrink: float = 10.0, alpha: float = 0.5, pow: float = 1.0,
                 iuf: bool = False, block_size: int = 2048, device: str | None = None):
        self.k, self.shrink, self.alpha, self.pow, self.iuf = int(k), shrink, alpha, pow, bool(iuf)
        self.block_size = block_size
        self.device = get_device(device)

    def fit(self, X: sp.csr_matrix, profile: sp.csr_matrix | None = None):
        X = X.tocsr().astype(np.float32)
        self.X = X if profile is None else profile.tocsr().astype(np.float32)
        n_users, n_items = X.shape
        Xw = X
        if self.iuf:
            n_u = np.diff(X.indptr).astype(np.float32)
            Xw = sp.diags(1.0 / np.log1p(np.maximum(n_u, 1.0)) ** 0.5) @ X
            Xw = Xw.tocsr()
        sq_norm = np.asarray(Xw.multiply(Xw).sum(axis=0)).ravel()
        dev = self.device
        Xt = Xw.T.tocoo()
        Xt_t = torch.sparse_coo_tensor(np.vstack([Xt.row, Xt.col]), Xt.data, size=Xt.shape,
                                       device=dev).coalesce()
        Xc = Xw.tocsc()
        n_i = torch.tensor(sq_norm, device=dev)
        left = n_i.clamp_min(1e-12) ** self.alpha
        k = min(self.k, n_items - 1)
        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.block_size):
            stop = min(start + self.block_size, n_items)
            block = torch.tensor(Xc[:, start:stop].toarray(), device=dev)
            co = torch.sparse.mm(Xt_t, block) 
            right = n_i[start:stop].clamp_min(1e-12) ** (1.0 - self.alpha)
            sim = co / (left[:, None] * right[None, :] + self.shrink)
            idx = torch.arange(start, stop, device=dev)
            sim[idx, idx - start] = 0.0 
            if self.pow != 1.0:
                sim = sim.clamp_min(0) ** self.pow
            top_v, top_i = torch.topk(sim, k, dim=0) 
            rows.append(top_i.T.reshape(-1).cpu().numpy())
            cols.append(np.repeat(np.arange(start, stop), k))
            vals.append(top_v.T.reshape(-1).cpu().numpy())
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        keep = vals > 0
        self.S = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n_items, n_items),
                               dtype=np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.X[users] @ self.S).toarray()
