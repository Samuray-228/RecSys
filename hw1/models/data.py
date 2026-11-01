"""Loading, cleaning, temporal splitting and feature construction for Goodreads Comics & Graphic.

Source: UCSD Book Graph (Wan & McAuley, RecSys 2018; Wan et al., ACL 2019),
https://mengtingwan.github.io/data/goodreads.html  (files of the "Comics & Graphic" genre).
"""

from __future__ import annotations

import gzip
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import EvalSet, to_csr

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"
FILES = {
    "interactions": "goodreads_interactions_comics_graphic.json.gz",
    "books": "goodreads_books_comics_graphic.json.gz",
}


def download(data_dir: str = "data") -> Dict[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for key, name in FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(BASE_URL + name, path)
        paths[key] = path
    return paths


def load_interactions(path: str, cache: Optional[str] = None) -> pd.DataFrame:
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    cols = {"user_id": [], "book_id": [], "is_read": [], "rating": [], "has_review": [], "date_added": []}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cols["user_id"].append(d["user_id"])
            cols["book_id"].append(d["book_id"])
            cols["is_read"].append(bool(d["is_read"]))
            cols["rating"].append(int(d["rating"]))
            cols["has_review"].append(len(d.get("review_text_incomplete") or "") > 0)
            cols["date_added"].append(d["date_added"])
    df = pd.DataFrame(cols)
    df["user_id"] = df["user_id"].astype("category")
    df["book_id"] = df["book_id"].astype(np.int64)
    df["rating"] = df["rating"].astype(np.int8)
    df["date_added"] = pd.to_datetime(df["date_added"], format="%a %b %d %H:%M:%S %z %Y",
                                      errors="coerce", utc=True)
    if cache:
        df.to_parquet(cache, index=False)
    return df


def _int_or_nan(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


_SERIES_PAREN = re.compile(r"\(([^()]+?),?\s*#\s*[\d.\-]+\)\s*$")
_VOLUME = re.compile(r"[,:]?\s*(vol\.?|volume|tome|tomo|band|book|issue|part|no\.?|#)\s*[\divxlc]+\b.*$", re.I)


def series_key(title: str) -> str:
    m = _SERIES_PAREN.search(title or "")
    if m:
        return m.group(1).strip().lower()
    base = _VOLUME.sub("", title or "").strip().lower()
    return base if base and base != (title or "").strip().lower() else ""


def load_books(path: str, cache: Optional[str] = None) -> pd.DataFrame:
    if cache and os.path.exists(cache):
        return pd.read_pickle(cache)
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "book_id": int(d["book_id"]),
                "work_id": int(d["work_id"]) if d.get("work_id") else -int(d["book_id"]),
                "title": d.get("title_without_series") or d.get("title") or "",
                "authors": [a["author_id"] for a in d.get("authors", []) if not a.get("role")]
                           or [a["author_id"] for a in d.get("authors", [])],
                "publisher": (d.get("publisher") or "").strip(),
                "format": (d.get("format") or "").strip().lower(),
                "language": (d.get("language_code") or "").strip().lower(),
                "is_ebook": d.get("is_ebook") == "true",
                "num_pages": _int_or_nan(d.get("num_pages")),
                "year": _int_or_nan(d.get("publication_year")),
                "series": series_key(d.get("title") or ""),
                "shelves": [(s["name"], int(s["count"])) for s in d.get("popular_shelves", [])],
                "description_len": len(d.get("description") or ""),
            })
    books = pd.DataFrame(rows)
    if cache:
        books.to_pickle(cache)
    return books


@dataclass
class PrepConfig:
    start: str = "2013-01-01"
    end: str = "2017-11-04"
    min_item: int = 20 
    min_user: int = 10
    max_interactions: int = 2_000_000
    seed: int = 42


def k_core(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    while True:
        n = len(df)
        ic = df["item"].map(df["item"].value_counts())
        df = df[ic >= min_item]
        uc = df["user"].map(df["user"].value_counts())
        df = df[uc >= min_user]
        if len(df) == n:
            return df


def prepare(inter: pd.DataFrame, books: pd.DataFrame, cfg: PrepConfig, log: Optional[list] = None) -> pd.DataFrame:
    log = log if log is not None else []
    df = inter
    log.append(("raw", len(df), df["user_id"].nunique(), df["book_id"].nunique()))
    df = df.dropna(subset=["date_added"])
    df = df[(df["date_added"] >= pd.Timestamp(cfg.start, tz="UTC")) &
            (df["date_added"] < pd.Timestamp(cfg.end, tz="UTC"))]
    log.append(("time window", len(df), df["user_id"].nunique(), df["book_id"].nunique()))
    df = df.merge(books[["book_id", "work_id"]], on="book_id", how="inner")
    df = (df.sort_values("date_added")
            .groupby(["user_id", "work_id"], observed=True, sort=False)
            .agg(date_added=("date_added", "first"), rating=("rating", "max"),
                 is_read=("is_read", "max"), has_review=("has_review", "max"), book_id=("book_id", "first"))
            .reset_index())
    log.append(("editions -> works, dedup", len(df), df["user_id"].nunique(), df["work_id"].nunique()))
    df = df.rename(columns={"user_id": "user_raw", "work_id": "item_raw"})
    df["user"] = df["user_raw"].astype(str)
    df["item"] = df["item_raw"]
    df = k_core(df, cfg.min_user, cfg.min_item)
    log.append((f"k-core (user>={cfg.min_user}, item>={cfg.min_item})", len(df), df["user"].nunique(), df["item"].nunique()))
    if len(df) > cfg.max_interactions:
        rng = np.random.default_rng(cfg.seed)
        users = np.sort(df["user"].unique())
        rng.shuffle(users)
        cnt = df.groupby("user").size().reindex(users).to_numpy()
        keep = users[: int(np.searchsorted(np.cumsum(cnt), cfg.max_interactions * 0.97))]
        df = df[df["user"].isin(set(keep))]
        df = k_core(df, cfg.min_user, cfg.min_item)
        log.append(("seeded user sample + k-core", len(df), df["user"].nunique(), df["item"].nunique()))
    df["ts"] = df["date_added"].astype("int64") // 10**9
    df = df.sort_values(["ts", "user"]).reset_index(drop=True)
    return df[["user", "item", "book_id", "rating", "is_read", "has_review", "ts", "date_added"]]


def is_relevant(df: pd.DataFrame) -> np.ndarray:
    return ~df["rating"].isin([1, 2]).to_numpy()


@dataclass
class Split:
    train: pd.DataFrame 
    target: pd.DataFrame
    n_users: int
    n_items: int
    user_ids: np.ndarray
    item_ids: np.ndarray
    t_end: float
    stats: Dict[str, float] = field(default_factory=dict)

    def matrix(self, values: Optional[np.ndarray] = None) -> sp.csr_matrix:
        return to_csr(self.train["user"].to_numpy(), self.train["item"].to_numpy(), values,
                      self.n_users, self.n_items)

    def eval_set(self, item_pop: Optional[np.ndarray] = None) -> EvalSet:
        seen = to_csr(self.train["user"].to_numpy(), self.train["item"].to_numpy(), None,
                      self.n_users, self.n_items)
        tgt = self.target[self.target["relevant"]]
        truth = tgt.groupby("user")["item"].apply(lambda s: np.unique(s.to_numpy()))
        users = truth.index.to_numpy()
        if item_pop is None:
            item_pop = np.bincount(self.train["item"].to_numpy(), minlength=self.n_items).astype(np.float64)
        return EvalSet(users=users, truth=list(truth.to_numpy()), seen=seen, item_pop=item_pop)


def make_split(df: pd.DataFrame, history_end: pd.Timestamp, target_end: pd.Timestamp) -> Split:
    hist = df[df["date_added"] < history_end].copy()
    tgt = df[(df["date_added"] >= history_end) & (df["date_added"] < target_end)].copy()
    user_ids = np.sort(hist["user"].unique())
    item_ids = np.sort(hist["item"].unique())
    u_map = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i_map = pd.Series(np.arange(len(item_ids)), index=item_ids)
    stats = {"target_interactions": len(tgt)}
    tgt["relevant"] = is_relevant(tgt)
    stats["target_relevant"] = int(tgt["relevant"].sum())
    warm_u = tgt["user"].isin(u_map.index)
    warm_i = tgt["item"].isin(i_map.index)
    stats["share_cold_user_rows"] = float((~warm_u).mean())
    stats["share_new_item_rows"] = float((warm_u & ~warm_i).sum() / max(warm_u.sum(), 1))
    tgt = tgt[warm_u & warm_i]
    hist["user"] = u_map.loc[hist["user"]].to_numpy()
    hist["item"] = i_map.loc[hist["item"]].to_numpy()
    tgt["user"] = u_map.loc[tgt["user"]].to_numpy()
    tgt["item"] = i_map.loc[tgt["item"]].to_numpy()
    sp_ = Split(hist.reset_index(drop=True), tgt.reset_index(drop=True), len(user_ids), len(item_ids),
                user_ids, item_ids, float(hist["ts"].max()), stats)
    sp_.stats["eval_users"] = int(tgt.loc[tgt["relevant"], "user"].nunique())
    return sp_


SHELF_STOP = re.compile(
    r"(to-?read|currently|reading|read-?in|^read$|^read-|owned|own|wish|tbr|default|favou?rite|fav|"
    r"library|kindle|ebook|e-book|audio|buy|borrow|maybe|dnf|did-not-finish|re-?read|abandon|"
    r"20\d\d|19\d\d|challenge|shelf|book-?club|my-|i-own|have|series|next|later|giveaway|netgalley|"
    r"arc|review|calibre|amazon|kobo|nook|scribd|hoopla|overdrive|purchased|bought|on-hold|unread|"
    r"finished|books?$|stars?|^\d)")


def _bucket(x: np.ndarray, edges: List[float]) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=-1.0)
    b = np.digitize(x, edges)
    b[x < 0] = len(edges) + 1
    return b


class FeatureBuilder:

    def __init__(self, n_tags: int = 300, tags_per_book: int = 10, min_author_items: int = 3,
                 n_publishers: int = 300, hist_len: int = 200):
        self.n_tags, self.tags_per_book = n_tags, tags_per_book
        self.min_author_items, self.n_publishers, self.hist_len = min_author_items, n_publishers, hist_len

    @staticmethod
    def _block(rows, cols, n_rows, n_cols, norm=True):
        rows, cols = np.asarray(rows), np.asarray(cols)
        m = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_rows, n_cols))
        m.data[:] = 1.0
        if norm:
            cnt = np.diff(m.indptr).astype(np.float32)
            m = sp.diags(1.0 / np.sqrt(np.maximum(cnt, 1.0))) @ m
        return m.tocsr()

    def _onehot(self, codes, n_rows):
        codes = np.asarray(codes)
        return self._block(np.arange(n_rows), codes, n_rows, int(codes.max()) + 1, norm=False)

    def item_blocks(self, split: Split, books: pd.DataFrame) -> Dict[str, sp.csr_matrix]:
        n = split.n_items
        rep = (split.train.groupby(["item", "book_id"]).size().reset_index(name="c")
               .sort_values(["item", "c"], ascending=[True, False]).drop_duplicates("item"))
        meta = rep.merge(books, on="book_id", how="left").set_index("item").reindex(np.arange(n))
        blocks = {"item_id": self._block(np.arange(n), np.arange(n), n, n, norm=False)}
        ex = meta["authors"].apply(lambda a: a[:3] if isinstance(a, list) else []).explode().dropna()
        vc = ex.value_counts(); keep = vc[vc >= self.min_author_items].index
        ex = ex[ex.isin(keep)]
        a_map = {a: j for j, a in enumerate(keep)}
        blocks["author"] = self._block(ex.index.to_numpy(), ex.map(a_map).to_numpy(), n, max(len(keep), 1))
        ser = meta["series"].fillna("")
        vc = ser[ser != ""].value_counts(); keep = vc[vc >= 2].index
        ex = ser[ser.isin(keep)]
        s_map = {a: j for j, a in enumerate(keep)}
        blocks["series"] = self._block(ex.index.to_numpy(), ex.map(s_map).to_numpy(), n, max(len(keep), 1))
        in_series = (ser != "").to_numpy().astype(int)
        def tags_of(sh):
            if not isinstance(sh, list):
                return []
            good = [t for t, c in sh if not SHELF_STOP.search(t)]
            return good[: self.tags_per_book]
        tags = meta["shelves"].apply(tags_of)
        ex = tags.explode().dropna()
        vocab = ex.value_counts().index[: self.n_tags]
        ex = ex[ex.isin(vocab)]
        t_map = {t: j for j, t in enumerate(vocab)}
        self.tag_vocab = list(vocab)
        blocks["tags"] = self._block(ex.index.to_numpy(), ex.map(t_map).to_numpy(), n, len(vocab))
        pub = meta["publisher"].fillna("").str.lower()
        top_pub = pub[pub != ""].value_counts().index[: self.n_publishers]
        pub_code = pub.map({p: j for j, p in enumerate(top_pub)}).fillna(len(top_pub)).astype(int)
        blocks["publisher"] = self._onehot(pub_code.to_numpy(), n)
        fmt = meta["format"].fillna("").replace("", "unknown")
        top_fmt = fmt.value_counts().index[:15]
        blocks["format"] = self._onehot(fmt.map({p: j for j, p in enumerate(top_fmt)}).fillna(len(top_fmt)).astype(int).to_numpy(), n)
        lang = meta["language"].fillna("").replace("", "unknown")
        top_lang = lang.value_counts().index[:15]
        blocks["language"] = self._onehot(lang.map({p: j for j, p in enumerate(top_lang)}).fillna(len(top_lang)).astype(int).to_numpy(), n)
        blocks["year"] = self._onehot(_bucket(meta["year"], [1980, 1990, 2000, 2005, 2008, 2010, 2012, 2013, 2014, 2015, 2016, 2017]), n)
        blocks["pages"] = self._onehot(_bucket(meta["num_pages"], [1, 50, 100, 150, 200, 300, 500, 1000]), n)
        blocks["flags"] = self._onehot(meta["is_ebook"].fillna(False).astype(int).to_numpy() * 2 + in_series, n)
        tr = split.train
        pop = np.bincount(tr["item"], minlength=n)
        recent = np.bincount(tr.loc[tr["ts"] > split.t_end - 30 * 86400, "item"], minlength=n)
        blocks["pop"] = self._onehot(_bucket(pop, [25, 50, 100, 200, 400, 800, 1600]), n)
        blocks["recent_pop"] = self._onehot(_bucket(recent, [1, 2, 4, 8, 16, 32, 64]), n)
        return blocks

    def user_blocks(self, split: Split) -> Dict[str, sp.csr_matrix]:
        n = split.n_users
        tr = split.train
        cnt = np.bincount(tr["user"], minlength=n)
        rated = np.bincount(tr["user"], weights=(tr["rating"] > 0).astype(float), minlength=n)
        rsum = np.bincount(tr["user"], weights=tr["rating"].astype(float), minlength=n)
        mean_r = np.where(rated > 0, rsum / np.maximum(rated, 1), np.nan)
        last = pd.Series(tr["ts"].to_numpy()).groupby(tr["user"].to_numpy()).max().reindex(np.arange(n)).to_numpy()
        days_since = (split.t_end - last) / 86400
        return {
            "user_id": self._block(np.arange(n), np.arange(n), n, n, norm=False),
            "activity": self._onehot(_bucket(cnt, [15, 25, 40, 70, 120, 250, 500]), n),
            "rated_share": self._onehot(_bucket(rated / np.maximum(cnt, 1), [0.1, 0.3, 0.5, 0.7, 0.9]), n),
            "mean_rating": self._onehot(_bucket(mean_r, [3.0, 3.5, 4.0, 4.5]), n),
            "recency": self._onehot(_bucket(days_since, [7, 30, 90, 180, 365]), n),
        }

    def history_bag(self, split: Split) -> sp.csr_matrix:
        tr = split.train.sort_values("ts")
        tr = tr.groupby("user").tail(self.hist_len)
        return self._block(tr["user"].to_numpy(), tr["item"].to_numpy(), split.n_users, split.n_items)


def stack_blocks(blocks: Dict[str, sp.csr_matrix], names: List[str]) -> sp.csr_matrix:
    return sp.hstack([blocks[k] for k in names]).tocsr().astype(np.float32)
