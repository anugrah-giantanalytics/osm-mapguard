"""Optional side tables that sit next to the feature store (v3, 2026-10-11).

Both are keyed by changeset id and joined onto the rows of a prepared store. A missing file
is not an error: callers get None and skip the variant, so older stores keep working.

    object_context.parquet   what each modified/deleted object WAS before the edit (built on
                             the laptop from the raw stores: scripts/probes/object_context_extract.py)
                             ctx_tokens - bag like "@historic m@name @tourism=museum"
                             ctx_*      - numeric: share of touched objects with wikidata, name, ...
    text_embeddings.npz      pretrained multilingual sentence embeddings of the comment and of the
                             tag text (built on the cluster: train/08_text_embeddings)
    where_features.parquet   distance from the author's usual area (planet changeset dump).
                             Tested 2026-10-10: no gain on the ovid hybrid; kept for completeness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CONTEXT_FILE = "object_context.parquet"
EMBED_FILE = "text_embeddings.npz"
EMBED_META = "text_embeddings.json"
WHERE_FILE = "where_features.parquet"


def _root(store) -> Path:
    return Path(store.root)


def object_context(store, cids) -> tuple[list[str], np.ndarray, list[str]] | None:
    """(tokens per row, numeric matrix (N, k), column names) aligned to `cids`, or None."""
    p = _root(store) / CONTEXT_FILE
    if not p.exists():
        return None
    import pandas as pd

    df = pd.read_parquet(p).drop_duplicates("cid").set_index("cid")
    df = df.reindex(np.asarray(cids).astype("int64"))
    cols = [c for c in df.columns if c.startswith("ctx_") and c != "ctx_tokens"]
    num = df[cols].to_numpy(dtype="float64")
    num[:, [cols.index(c) for c in ("ctx_n_modified", "ctx_n_known") if c in cols]] = np.log1p(
        num[:, [cols.index(c) for c in ("ctx_n_modified", "ctx_n_known") if c in cols]])
    tokens = df["ctx_tokens"].fillna("").astype(str).tolist()
    return tokens, np.nan_to_num(num, nan=0.0), cols


def has_text_embeddings(store) -> bool:
    """Complete only: the meta file is written after the npz, and a --limit run never counts."""
    import json

    p, meta = _root(store) / EMBED_FILE, _root(store) / EMBED_META
    if not (p.exists() and meta.exists()):
        return False
    return json.loads(meta.read_text()).get("limit") is None


def has_object_context(store) -> bool:
    return (_root(store) / CONTEXT_FILE).exists()


def text_embeddings(store, cids) -> dict[str, np.ndarray] | None:
    """{'comment': (N, d), 'tags': (N, d)} float32 aligned to `cids`, zeros where missing."""
    p = _root(store) / EMBED_FILE
    if not has_text_embeddings(store):
        return None
    z = np.load(p)
    pos = {int(c): i for i, c in enumerate(z["cids"])}
    idx = np.array([pos.get(int(c), -1) for c in cids])
    out = {}
    for k in ("comment", "tags"):
        m = z[k].astype("float32")
        a = np.zeros((len(idx), m.shape[1]), dtype="float32")
        a[idx >= 0] = m[idx[idx >= 0]]
        out[k] = a
    return out


def where_features(store, cids) -> tuple[np.ndarray, list[str]] | None:
    p = _root(store) / WHERE_FILE
    if not p.exists():
        return None
    import pandas as pd

    df = pd.read_parquet(p).drop_duplicates("cid").set_index("cid").reindex(np.asarray(cids).astype("int64"))
    cols = [c for c in df.columns if c.startswith("where_")]
    a = df[cols].to_numpy(dtype="float64")
    for j, c in enumerate(cols):
        if c.endswith("_km"):
            a[:, j] = np.log1p(a[:, j])
    return np.nan_to_num(a, nan=-1.0), cols
