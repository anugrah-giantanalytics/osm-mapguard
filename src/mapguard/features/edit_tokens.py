"""What an edit *did*, as tokens: the semantic channel the round-1 model was missing.

Round 1 fed the model counts - "3 tags added, 1 removed" - and it plateaued with XGBoost.
Probed on Corpus A (2026-09-22), the *identity* of the change carries the signal: the most
vandal-like tokens were `+leisure=park`, `+highway=footway`, `+highway=motorway`,
`+tourism=artwork`, `-historic` (fake parks and paths are the documented Pokemon-Go
vandalism; fake major roads and deleted landmarks are classic), and the most careful-mapper
tokens were `+source`, `+access=private`, `+fixme`. A bag of these tokens plus text lifted a
tree model from 0.764 to 0.836 PR-AUC counted one-per-author.

Tokens, per edit:

    create_node / modify_way / delete_relation   what kind of edit
    +key / -key / ~key                           tag added / removed / value changed
    +key=value / -key=value / ~key:old>new       the same, with the value, for the small set
                                                 of keys whose values are a vocabulary
                                                 (highway, building, landuse, ...)

Values of free-text keys (name, description) are *not* tokens - they are the separate
`old_text` / `text` channel read by a character model, because "Main Street" -> "poop" is
spelled, not enumerated.

Deterministic hashing (no vocabulary file) maps tokens to ids, so the same token gets the
same id on every machine and at inference time, and nothing is fitted on test data.
"""

from __future__ import annotations

import hashlib

# keys whose values are an enumerable vocabulary worth keeping as tokens
LOWCARD_KEYS = frozenset({
    "highway", "building", "amenity", "landuse", "place", "natural", "waterway", "boundary",
    "admin_level", "leisure", "shop", "railway", "power", "barrier", "surface", "oneway",
    "access", "tourism", "man_made", "route", "type", "area", "layer", "historic", "sport",
    "office", "craft", "emergency", "military", "aeroway", "public_transport", "water",
})
TOKEN_BUCKETS = 1 << 14        # hashed vocabulary size; 0 is reserved for padding
TOKENS_PER_EDIT = 12           # enough for the action token and ~5 tag changes with values
MAX_VALUE_CHARS = 32


def _base(key: str) -> str:
    return key.split(":")[0]


def edit_tokens(action: str, otype: str, old_tags: dict | None, new_tags: dict | None) -> list[str]:
    """Tokens for one edit. `old_tags` None means the previous version is unknown."""
    out = [f"{action}_{otype}"]
    if old_tags is None and action != "create":
        out.append("prev_unknown")
        old_tags = {}
    old = old_tags or {}
    new = {} if action == "delete" else (new_tags or {})
    for k in sorted(set(old) | set(new)):
        b = _base(k)
        low = b in LOWCARD_KEYS
        if k not in old:
            out.append(f"+{b}")
            if low:
                out.append(f"+{b}={str(new[k])[:MAX_VALUE_CHARS]}")
        elif k not in new:
            out.append(f"-{b}")
            if low:
                out.append(f"-{b}={str(old[k])[:MAX_VALUE_CHARS]}")
        elif old[k] != new[k]:
            out.append(f"~{b}")
            if low:
                out.append(f"~{b}:{str(old[k])[:MAX_VALUE_CHARS]}>{str(new[k])[:MAX_VALUE_CHARS]}")
    return [t.replace(" ", "_") for t in out]


def token_id(token: str, buckets: int = TOKEN_BUCKETS) -> int:
    """Stable id in [1, buckets). md5, not Python's salted hash(), so ids never change."""
    h = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "little")
    return 1 + h % (buckets - 1)


def encode_tokens(tokens: str | None, length: int = TOKENS_PER_EDIT,
                  buckets: int = TOKEN_BUCKETS) -> list[int]:
    """Space-separated tokens -> fixed-length id list, zero-padded. Action token first."""
    ids = [token_id(t, buckets) for t in (tokens or "").split()[:length]]
    return ids + [0] * (length - len(ids))
