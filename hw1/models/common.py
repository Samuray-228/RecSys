"""Shared utilities: reproducibility, sparse matrices, top-K retrieval and ranking metrics."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import torch
except ImportError: 
    torch = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_device(prefer: Optional[str] = None) -> "torch.device":
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_csr(users: np.ndarray, items: np.ndarray, values: Optional[np.ndarray],
           n_users: int, n_items: int, dtype=np.float32) -> sp.csr_matrix:
    if values is None:
        values = np.ones(len(users), dtype=dtype)
    df = pd.DataFrame({"u": users, "i": items, "v": values})
    df = df.groupby(["u", "i"], sort=False)["v"].max().reset_index()
    m = sp.csr_matrix((df["v"].to_numpy(dtype), (df["u"].to_numpy(), df["i"].to_numpy())),
                      shape=(n_users, n_items), dtype=dtype)
    m.sort_indices()
    return m


def recency_profile(users: np.ndarray, items: np.ndarray, ts: np.ndarray, t_end: float,
                    half_life_days: Optional[float], n_users: int, n_items: int) -> sp.csr_matrix:
    if half_life_days is None or np.isinf(half_life_days):
        w = np.ones(len(users), dtype=np.float32)
    else:
        dt = np.maximum(t_end - np.asarray(ts, dtype=np.float64), 0.0) / 86400.0
        w = np.exp2(-dt / half_life_days).astype(np.float32)
    return to_csr(users, items, w, n_users, n_items)


def binarize(m: sp.csr_matrix) -> sp.csr_matrix:
    b = m.copy().astype(np.float32)
    b.data[:] = 1.0
    return b


def topk_from_scores(scores: np.ndarray, seen: sp.csr_matrix, users: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).copy()
    sub = seen[users]
    rows = np.repeat(np.arange(len(users)), np.diff(sub.indptr))
    scores[rows, sub.indices] = -np.inf
    part = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def recommend(score_fn: Callable[[np.ndarray], np.ndarray], users: np.ndarray, seen: sp.csr_matrix,
              k: int = 10, batch_size: int = 2048) -> np.ndarray:
    out = np.empty((len(users), k), dtype=np.int64)
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        out[start:start + len(batch)] = topk_from_scores(score_fn(batch), seen, batch, k)
    return out


def ranking_metrics(recs: np.ndarray, truth: List[np.ndarray], k: int = 10,
                    n_items: Optional[int] = None,
                    item_pop: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Binary-relevance metrics averaged over users.

    NDCG@k   - position-aware quality of the list (primary metric).
    Recall@k - |hits| / min(|relevant|, k): share of what the user will read that we surface.
    HR@k     - at least one hit in the list.
    MAP@k    - mean average precision (normalised by min(|relevant|, k)).
    Coverage@k - share of the catalog that appears in at least one list.
    Novelty@k  - mean self-information -log2(pop share) of recommended items.
    """
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    idcg_cum = np.cumsum(discounts)
    ndcg = np.empty(len(truth)); recall = np.empty(len(truth)); hr = np.empty(len(truth)); ap = np.empty(len(truth))
    for n, (rec, rel) in enumerate(zip(recs, truth)):
        hits = np.isin(rec[:k], rel)
        n_rel = min(len(rel), k)
        dcg = float((hits * discounts).sum())
        ndcg[n] = dcg / idcg_cum[n_rel - 1]
        recall[n] = hits.sum() / n_rel
        hr[n] = float(hits.any())
        prec_at = np.cumsum(hits) / np.arange(1, k + 1)
        ap[n] = float((prec_at * hits).sum() / n_rel)
    res = {f"NDCG@{k}": ndcg.mean(), f"Recall@{k}": recall.mean(), f"HR@{k}": hr.mean(), f"MAP@{k}": ap.mean()}
    if n_items is not None:
        res[f"Coverage@{k}"] = len(np.unique(recs[:, :k])) / n_items
    if item_pop is not None:
        p = item_pop / item_pop.sum()
        res[f"Novelty@{k}"] = float(np.mean(-np.log2(np.maximum(p[recs[:, :k]], 1e-12))))
    return res


def per_user_ndcg(recs: np.ndarray, truth: List[np.ndarray], k: int = 10) -> np.ndarray:
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    idcg_cum = np.cumsum(discounts)
    return np.array([(np.isin(r[:k], t) * discounts).sum() / idcg_cum[min(len(t), k) - 1]
                     for r, t in zip(recs, truth)])


@dataclass
class EvalSet:
    users: np.ndarray
    truth: List[np.ndarray]
    seen: sp.csr_matrix
    item_pop: Optional[np.ndarray] = None

    def subsample(self, n: int, seed: int = 0) -> "EvalSet":
        if n >= len(self.users):
            return self
        idx = np.sort(np.random.default_rng(seed).choice(len(self.users), n, replace=False))
        return EvalSet(self.users[idx], [self.truth[j] for j in idx], self.seen, self.item_pop)


def evaluate(model, ev: EvalSet, k: int = 10, batch_size: int = 2048, return_recs: bool = False):
    t0 = time.time()
    recs = recommend(model.score, ev.users, ev.seen, k=k, batch_size=batch_size)
    res = ranking_metrics(recs, ev.truth, k=k, n_items=ev.seen.shape[1], item_pop=ev.item_pop)
    res["eval_sec"] = time.time() - t0
    return (res, recs) if return_recs else res


class Popularity:

    def __init__(self, window_days: Optional[float] = None):
        self.window_days = window_days

    def fit(self, users: np.ndarray, items: np.ndarray, ts: np.ndarray, n_items: int):
        mask = np.ones(len(items), bool)
        if self.window_days is not None:
            mask = ts >= ts.max() - self.window_days * 86400
        self.pop = np.bincount(items[mask], minlength=n_items).astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.pop, (len(users), len(self.pop)))


def grid_search(build_and_fit: Callable[[dict], object], grid: Iterable[dict], ev: EvalSet,
                metric: str = "NDCG@10", k: int = 10, verbose: bool = True) -> pd.DataFrame:
    rows = []
    for params in grid:
        t0 = time.time()
        model = build_and_fit(params)
        fit_sec = time.time() - t0
        res = evaluate(model, ev, k=k)
        rows.append({**params, **res, "fit_sec": fit_sec})
        if verbose:
            print({**params, metric: round(res[metric], 5), f"Recall@{k}": round(res[f'Recall@{k}'], 5),
                   "fit_sec": round(fit_sec, 1)}, flush=True)
        del model
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


def expand_grid(**axes) -> List[dict]:
    keys = list(axes)
    out = [{}]
    for key in keys:
        out = [{**o, key: v} for o in out for v in axes[key]]
    return out
