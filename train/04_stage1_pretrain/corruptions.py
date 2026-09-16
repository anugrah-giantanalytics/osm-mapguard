"""Pretext C - synthetic damage applied to real edits.

Stage 1 learns what a normal edit looks like by learning to tell real edits from corrupted
ones. The corruptions are chosen to mimic real vandalism modes, because a pretext task only
transfers if the difference it teaches is the difference that matters downstream.

Every corruption operates on the **raw** per-edit feature vector - metres, ratios, counts -
before the log and z-score transform, so a "400 km drag" is a 400 km drag and not a number
that happens to look large after scaling. Columns are addressed by name from the data
contract, never by position.

| Corruption       | Changes                                 | Real vandalism it mimics                       |
|------------------|-----------------------------------------|------------------------------------------------|
| displace         | centroid shift, Hausdorff x 10^2-10^4   | a node or way dragged across a city or country |
| area_blowup      | area ratio and area delta x 50-500      | a polygon stretched to swallow its surroundings |
| length_stretch   | length ratio and length delta x 20-200  | a road pulled out into a long spike            |
| tag_wipe         | all tags removed                        | an object stripped of its identity             |
| tag_spam         | 10-40 tags added                        | junk or promotional tagging                     |
| mass_points      | +200 to +2000 vertices                  | a scribbled or auto-traced garbage shape       |

What is **not** here, deliberately: text corruption. Tag text lives at changeset level in
this feature store, not per edit, so a per-edit text corruption would be corrupting a column
that does not exist. Mass deletion is also changeset-level. Both are recorded as known gaps
in the pretext task rather than faked.

A corruption that would leave the vector unchanged (e.g. tag_wipe on an edit with no tags)
is replaced by another, so every "corrupted" sample really differs from its source.
"""

from __future__ import annotations

import numpy as np

CORRUPTIONS = ("displace", "area_blowup", "length_stretch", "tag_wipe", "tag_spam", "mass_points")


def _col(columns, name):
    return columns.index(name) if name in columns else None


def corrupt_one(x: np.ndarray, kind: str, columns: list[str], rng: np.random.Generator) -> np.ndarray | None:
    """Return a corrupted copy of one raw edit vector, or None if `kind` cannot change it."""
    v = x.astype("float64").copy()
    c = lambda n: _col(columns, n)  # noqa: E731

    if kind == "displace":
        idx = [i for i in (c("centroid_shift_m"), c("hausdorff_m")) if i is not None]
        if not idx:
            return None
        factor = 10 ** rng.uniform(2, 4)
        for i in idx:
            # An edit that did not move (a create) still gets a real displacement.
            v[i] = max(abs(v[i]), 1.0) * factor
    elif kind == "area_blowup":
        idx = [i for i in (c("area_ratio"), c("d_area_m2")) if i is not None]
        if not idx:
            return None
        factor = rng.uniform(50, 500)
        if c("area_ratio") is not None:
            v[c("area_ratio")] = max(v[c("area_ratio")], 1.0) * factor
        if c("d_area_m2") is not None:
            v[c("d_area_m2")] = max(abs(v[c("d_area_m2")]), 100.0) * factor
    elif kind == "length_stretch":
        idx = [i for i in (c("length_ratio"), c("d_length_m")) if i is not None]
        if not idx:
            return None
        factor = rng.uniform(20, 200)
        if c("length_ratio") is not None:
            v[c("length_ratio")] = max(v[c("length_ratio")], 1.0) * factor
        if c("d_length_m") is not None:
            v[c("d_length_m")] = max(abs(v[c("d_length_m")]), 10.0) * factor
    elif kind == "tag_wipe":
        n = c("n_tags")
        if n is None or v[n] <= 0:
            return None
        if c("tags_removed") is not None:
            v[c("tags_removed")] += v[n]
        for name in ("tags_added", "tags_changed"):
            if c(name) is not None:
                v[c(name)] = 0.0
        v[n] = 0.0
    elif kind == "tag_spam":
        n = c("n_tags")
        if n is None:
            return None
        k = float(rng.integers(10, 41))
        v[n] += k
        if c("tags_added") is not None:
            v[c("tags_added")] += k
    elif kind == "mass_points":
        i = c("d_points")
        if i is None:
            return None
        v[i] += float(rng.integers(200, 2001))
        if c("n_refs") is not None:
            v[c("n_refs")] += v[i]
    else:
        raise ValueError(f"unknown corruption {kind!r}")

    return v if not np.allclose(v, x) else None


def make_contrast_set(real: np.ndarray, columns: list[str], *, seed: int,
                      kinds=CORRUPTIONS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balanced real-vs-corrupted set from raw real edit vectors.

    Returns (X, y, kind_index): y=0 real, y=1 corrupted; kind_index is -1 for real rows
    and the index into `kinds` for corrupted ones, so per-corruption accuracy can be
    reported instead of one number that hides which modes the encoder never learned.
    """
    rng = np.random.default_rng(seed)
    kinds = tuple(kinds)
    corrupted, kind_idx = [], []
    for x in real:
        order = rng.permutation(len(kinds))
        for j in order:
            out = corrupt_one(x, kinds[j], columns, rng)
            if out is not None:
                corrupted.append(out)
                kind_idx.append(int(j))
                break
    corrupted = np.asarray(corrupted, dtype="float64").reshape(-1, real.shape[1])
    X = np.vstack([real.astype("float64"), corrupted])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(corrupted))]).astype("float32")
    k = np.concatenate([np.full(len(real), -1), np.asarray(kind_idx, dtype=int)])
    perm = rng.permutation(len(X))
    return X[perm], y[perm], k[perm]
