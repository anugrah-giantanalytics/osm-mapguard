"""Output 3 - which signal fired - by occlusion.

The plan promises three outputs: a score, which edit, and which signal. The first two come
out of the model. The third is measured here by removing one branch's input at a time and
watching the score move:

    B1  geometry columns set to their training mean (0 after normalisation)
    B2  text replaced with padding, text-derived counts set to the mean
    B3  real edits shuffled within each changeset - removes order, keeps content
    B4  user history emptied
    B5  tabular features set to the mean
    B6  edit tokens and per-edit old>new text replaced with padding

Occlusion rather than gradients because it answers the reviewer's question directly -
"how much would this score change without that evidence?" - and because it behaves the same
under Keras 2 and Keras 3 with no custom gradient code.

It is an attribution, not a causal claim. Branches interact, so the per-branch deltas do not
sum to the score, and the paper should say so wherever these numbers appear.
"""

from __future__ import annotations

import numpy as np

from .preprocess import GEOMETRY_EDIT_COLUMNS, TEXT_TABULAR_COLUMNS, is_geometry_tabular


def _occlude(arrays: dict, branch: str, edit_columns, tabular_columns, rng) -> dict | None:
    a = {k: v.copy() for k, v in arrays.items()}
    if branch == "B1":
        e_idx = [i for i, c in enumerate(edit_columns) if c in GEOMETRY_EDIT_COLUMNS]
        t_idx = [i for i, c in enumerate(tabular_columns) if is_geometry_tabular(c)]
        if not e_idx and not t_idx:
            return None
        a["edit_features"][:, :, e_idx] = 0.0
        a["tabular"][:, t_idx] = 0.0
    elif branch == "B2":
        a["text"][:] = 0
        t_idx = [i for i, c in enumerate(tabular_columns) if c in TEXT_TABULAR_COLUMNS]
        a["tabular"][:, t_idx] = 0.0
    elif branch == "B3":
        for r in range(len(a["edit_mask"])):
            n = int(a["edit_mask"][r].sum())
            if n > 1:
                order = rng.permutation(n)
                a["edit_features"][r, :n] = a["edit_features"][r, order]
    elif branch == "B4":
        a["user_sequence"][:] = 0.0
        a["user_mask"][:] = 0.0
    elif branch == "B5":
        a["tabular"][:] = 0.0
    elif branch == "B6":
        if "edit_tokens" not in a:
            return None
        a["edit_tokens"][:] = 0
        a["edit_text"][:] = 0
    else:
        return None
    return a


def branch_attribution(model, arrays: dict, *, branches, edit_columns, tabular_columns,
                       y: np.ndarray | None = None, batch_size: int = 256,
                       seed: int = 0) -> dict:
    from mapguard.models.stage2 import pack   # lazy: keeps TensorFlow out of CPU-only imports

    rng = np.random.default_rng(seed)
    base = model.predict(pack(model, arrays), batch_size=batch_size, verbose=0)[0].ravel()
    out = {"base_mean_score": round(float(base.mean()), 4), "branches": {}}
    per_row = {}

    for b in branches:
        occ = _occlude(arrays, b, edit_columns, tabular_columns, rng)
        if occ is None:
            continue
        p = model.predict(pack(model, occ), batch_size=batch_size, verbose=0)[0].ravel()
        delta = base - p
        per_row[b] = delta
        entry = {"mean_abs_delta": round(float(np.abs(delta).mean()), 4),
                 "mean_delta": round(float(delta.mean()), 4)}
        if y is not None:
            yy = np.asarray(y).astype(int)
            for cls, name in ((1, "positives"), (0, "negatives")):
                m = yy == cls
                if m.any():
                    entry[f"mean_delta_{name}"] = round(float(delta[m].mean()), 4)
        out["branches"][b] = entry

    out["note"] = ("occlusion deltas; branches interact, so deltas do not sum to the score "
                   "and are not causal effects")
    return out, per_row, base


def top_examples(cids, y, scores, edit_weights, edit_mask, per_row_delta, k: int = 20) -> list:
    """The k highest-scoring changesets with the evidence a reviewer would open first."""
    order = np.argsort(-np.asarray(scores).ravel())[:k]
    rows = []
    for i in order:
        n_real = int(edit_mask[i].sum())
        w = np.asarray(edit_weights[i][:n_real])
        top_edits = np.argsort(-w)[:3].tolist() if n_real else []
        rows.append({
            "cid": int(cids[i]),
            "label": int(y[i]) if y is not None else None,
            "score": round(float(np.ravel(scores)[i]), 4),
            "top_edit_slots": top_edits,
            "top_edit_weights": [round(float(w[j]), 4) for j in top_edits],
            "signal_deltas": {b: round(float(d[i]), 4) for b, d in per_row_delta.items()},
        })
    return rows
