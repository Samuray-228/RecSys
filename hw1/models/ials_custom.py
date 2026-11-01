"""Model 5: iALS with a hand-designed target signal (recency- and rating-aware preference)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .ials import WeightedALS


def custom_target(rating: np.ndarray, ts: np.ndarray, t_end: float, gamma: float = 0.5,
                  half_life_days: float = 180.0, beta: float = 0.3, dislike_zero: bool = True) -> np.ndarray:
    rating = np.asarray(rating, dtype=np.float32)
    g = np.ones_like(rating)
    rated = rating > 0
    g[rated] = np.maximum(0.0, 1.0 + gamma * (rating[rated] - 3.5) / 1.5)
    if dislike_zero:
        g[(rating >= 1) & (rating <= 2)] = 0.0
    if half_life_days is None or np.isinf(half_life_days):
        d = np.ones_like(g)
    else:
        dt = np.maximum(t_end - np.asarray(ts, dtype=np.float64), 0.0) / 86400.0
        d = (beta + (1.0 - beta) * np.exp2(-dt / half_life_days)).astype(np.float32)
    return (g * d).astype(np.float32)


class CustomTargetIALS(WeightedALS):
    def __init__(self, alpha: float = 10.0, gamma: float = 0.5, half_life_days: float = 180.0,
                 beta: float = 0.3, dislike_zero: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.alpha, self.gamma, self.half_life_days = alpha, gamma, half_life_days
        self.beta, self.dislike_zero = beta, dislike_zero

    def fit(self, df: pd.DataFrame, n_users: int, n_items: int, t_end: float | None = None):
        t_end = float(df["ts"].max()) if t_end is None else t_end
        p = custom_target(df["rating"].to_numpy(), df["ts"].to_numpy(), t_end, self.gamma,
                          self.half_life_days, self.beta, self.dislike_zero)
        u, i = df["user"].to_numpy(), df["item"].to_numpy()
        P = sp.csr_matrix((p, (u, i)), shape=(n_users, n_items), dtype=np.float32)
        C = sp.csr_matrix((np.full(len(p), 1.0 + self.alpha, np.float32), (u, i)),
                          shape=(n_users, n_items), dtype=np.float32)
        P.sort_indices(); C.sort_indices()
        if not ((P.indptr == C.indptr).all() and (P.indices == C.indices).all()):
            raise ValueError("duplicate (user, item) pairs in df")
        self.target_values = p
        return self.fit_weighted(P, C)
