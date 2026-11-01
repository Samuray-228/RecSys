"""EASE^R (Steck 2019): embarrassingly shallow linear item-item autoencoder."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .common import get_device


class EASE:
    def __init__(self, reg: float = 500.0, device: str | None = None, dtype=torch.float32):
        self.reg = reg
        self.device = get_device(device)
        self.dtype = dtype

    def fit(self, X: sp.csr_matrix, profile: sp.csr_matrix | None = None):
        X = X.tocsr().astype(np.float32)
        self.X = X if profile is None else profile.tocsr().astype(np.float32)
        G = torch.tensor((X.T @ X).toarray(), device=self.device, dtype=self.dtype)
        G.diagonal().add_(self.reg)
        P = torch.linalg.inv(G)
        del G
        B = P / (-P.diagonal()[None, :])
        B.diagonal().zero_()
        self.B = B.to(torch.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        xu = self.X[users].tocoo()
        xu_t = torch.sparse_coo_tensor(np.vstack([xu.row, xu.col]), xu.data, size=xu.shape,
                                       device=self.device, dtype=torch.float32)
        return torch.sparse.mm(xu_t, self.B).cpu().numpy()
