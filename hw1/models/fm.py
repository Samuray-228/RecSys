"""Factorization Machine (Rendle 2010) for top-K ranking, trained with sampled BPR / softmax."""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .common import get_device


def _padded(m: sp.csr_matrix, offset: int, pad_index: int):
    m = m.tocsr()
    lengths = np.diff(m.indptr)
    L = max(int(lengths.max()), 1)
    idx = np.full((m.shape[0], L), pad_index, dtype=np.int64)
    val = np.zeros((m.shape[0], L), dtype=np.float32)
    pos = np.arange(m.nnz) - np.repeat(m.indptr[:-1], lengths)
    rows = np.repeat(np.arange(m.shape[0]), lengths)
    idx[rows, pos] = m.indices + offset
    val[rows, pos] = m.data
    return idx, val


class FM:
    def __init__(self, factors: int = 64, lr: float = 2e-3, reg: float = 1e-5, epochs: int = 10,
                 batch_size: int = 4096, n_neg: int = 1024, loss: str = "softmax", init_std: float = 0.01,
                 seed: int = 42, device: str | None = None, verbose: bool = False):
        self.factors, self.lr, self.reg, self.epochs = int(factors), lr, reg, int(epochs)
        self.batch_size, self.n_neg, self.loss, self.init_std = int(batch_size), int(n_neg), loss, init_std
        self.seed, self.device, self.verbose = seed, get_device(device), verbose

    def _side(self, idx_pad, val_pad, rows):
        idx = idx_pad[rows]; val = val_pad[rows]
        v = self.V(idx) * val[..., None] 
        s = v.sum(1)
        q = (v * v).sum(1) 
        lin = (self.w(idx).squeeze(-1) * val).sum(1)
        return s, q, lin

    def _item_terms(self, items):
        s, q, lin = self._side(self.i_idx, self.i_val, items)
        return s, lin + 0.5 * (s * s - q).sum(-1)

    def _user_vec(self, users, exclude_items=None):
        idx = self.u_idx[users]; val = self.u_val[users]
        v = self.V(idx) * val[..., None]
        s = v.sum(1)
        if exclude_items is not None and self.hist_offset is not None:
            hit = (idx == (self.hist_offset + exclude_items)[:, None]).float() * val
            h = hit.sum(1)
            s = s - h[:, None] * self.V(self.hist_offset + exclude_items)
        return s

    def fit(self, X: sp.csr_matrix, user_feats: sp.csr_matrix, item_feats: sp.csr_matrix,
            hist: sp.csr_matrix | None = None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = self.device
        n_users, n_items = X.shape
        user_side = user_feats if hist is None else sp.hstack([user_feats, hist]).tocsr()
        n_u_feat, n_i_feat = user_side.shape[1], item_feats.shape[1]
        self.hist_offset = None if hist is None else user_feats.shape[1]
        n_total = n_u_feat + n_i_feat
        pad = n_total
        ui, uv = _padded(user_side, 0, pad)
        ii, iv = _padded(item_feats, n_u_feat, pad)
        self.u_idx, self.u_val = torch.tensor(ui, device=dev), torch.tensor(uv, device=dev)
        self.i_idx, self.i_val = torch.tensor(ii, device=dev), torch.tensor(iv, device=dev)
        self.V = torch.nn.Embedding(n_total + 1, self.factors, padding_idx=pad).to(dev)
        self.w = torch.nn.Embedding(n_total + 1, 1, padding_idx=pad).to(dev)
        torch.nn.init.normal_(self.V.weight, std=self.init_std)
        torch.nn.init.zeros_(self.w.weight)
        with torch.no_grad():
            self.V.weight[pad].zero_()
        params = list(self.V.parameters()) + list(self.w.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)

        coo = X.tocoo()
        pu, pi = coo.row.astype(np.int64), coo.col.astype(np.int64)
        n = len(pu)
        self.history = []
        for ep in range(self.epochs):
            t0 = time.time()
            perm = rng.permutation(n)
            total = 0.0
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                u = torch.tensor(pu[b], device=dev)
                ip = torch.tensor(pi[b], device=dev)
                ineg = torch.randint(0, n_items, (self.n_neg,), device=dev)
                su = self._user_vec(u, exclude_items=ip)
                sp_, bp = self._item_terms(ip)
                sn, bn = self._item_terms(ineg)
                y_pos = (su * sp_).sum(-1) + bp
                y_neg = su @ sn.T + bn[None, :]
                hit = ineg[None, :] == ip[:, None]
                if self.loss == "bpr":
                    loss = -(F.logsigmoid(y_pos[:, None] - y_neg) * (~hit)).sum() / (~hit).sum()
                else:
                    y_neg = y_neg.masked_fill(hit, float("-inf"))
                    logits = torch.cat([y_pos[:, None], y_neg], 1)
                    loss = F.cross_entropy(logits, torch.zeros(len(b), dtype=torch.long, device=dev))
                reg = (su.pow(2).sum(-1).mean() + sp_.pow(2).sum(-1).mean() + sn.pow(2).sum(-1).mean())
                loss_total = loss + self.reg * reg
                opt.zero_grad(set_to_none=True)
                loss_total.backward()
                opt.step()
                total += loss.item() * len(b)
            self.history.append(total / n)
            if self.verbose:
                print(f"epoch {ep + 1}/{self.epochs} loss={total / n:.4f} ({time.time() - t0:.1f}s)", flush=True)
        self._cache_items()
        return self

    @torch.no_grad()
    def _cache_items(self, chunk: int = 8192):
        n_items = self.i_idx.shape[0]
        S, B = [], []
        for s in range(0, n_items, chunk):
            items = torch.arange(s, min(s + chunk, n_items), device=self.device)
            si, bi = self._item_terms(items)
            S.append(si); B.append(bi)
        self.item_vecs, self.item_bias = torch.cat(S), torch.cat(B)

    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        u = torch.as_tensor(np.asarray(users), dtype=torch.long, device=self.device)
        su = self._user_vec(u)
        return (su @ self.item_vecs.T + self.item_bias[None, :]).cpu().numpy()
