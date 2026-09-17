"""Where each corpus lives on disk, and the rules shared by the fetcher and the builder.

Anything both `scripts/fetch_osm.py` and `mapguard.features.build_store` must agree on
lives here. If they disagreed on which N rows count, or which history a target may use,
the feature store would quietly describe a different dataset from the one fetched.

    A   Ovid benchmark, 2014-2019, revert-derived labels     corpus_a_meta + corpus_a
    M   OSMCha-reviewed, 2023-2026                           corpus_mn (corpus = M)
    N   random unreviewed changesets from M's quarters       corpus_mn (corpus = N)
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

CORPUS_MN_DB = "raw/corpus_mn.sqlite"
LABELS_DB = "raw/osmcha_labels.sqlite"
USERS_DB = "raw/users.sqlite"
HISTORY_DB = "raw/user_histories_pit.sqlite"
LEGACY_HISTORY_DB = "raw/user_histories.sqlite"
VERSIONS_DB = "raw/prev_versions.sqlite"

SEED = 20260904
M_PER_CLASS = 4000
N_TOTAL = 4000


def quarter_of(date: str | None) -> str | None:
    if not date or len(date) < 7:
        return None
    return f"{date[:4]}Q{(int(date[5:7]) + 2) // 3}"


def valid_selection(conn: sqlite3.Connection, n_total: int = N_TOTAL) -> dict[int, tuple[str, int]]:
    """cid -> (corpus, label) for M rows with metadata plus the usable N rows.

    N is oversampled because a random id can be an empty or still-open changeset. Each
    quarter's quota is filled in hash order, so which N rows count never depends on
    content.
    """
    m_harm = dict(conn.execute(
        "SELECT quarter, COUNT(*) FROM selection WHERE corpus='M' AND label=1 GROUP BY 1"))
    total_harm = sum(m_harm.values()) or 1
    quota = {q: int(round(n_total * n / total_harm)) for q, n in m_harm.items()}
    out: dict[int, tuple[str, int]] = {}
    rows = conn.execute(
        "SELECT s.cid, s.corpus, s.label, s.quarter, m.status, m.closed_at, m.num_changes,"
        " m.created_at FROM selection s LEFT JOIN meta m ON m.cid = s.cid ORDER BY s.rank")
    for cid, corpus, label, q, status, closed, n, created in rows:
        if status != 200:
            continue
        if corpus == "M":
            out[cid] = ("M", int(label))
        elif (closed and (n or 0) > 0 and quarter_of(created) == q and quota.get(q, 0) > 0):
            quota[q] -= 1
            out[cid] = ("N", 0)
    return out


def history_for(conn: sqlite3.Connection | None, uid: int | None, created_at: str | None,
                legacy: sqlite3.Connection | None = None) -> tuple[str | None, str]:
    """The stored history a target may use, and the rule that allowed it.

    exact                  fetched with this target's own timestamp as the cutoff
    complete_later_window  fetched for a later target and not truncated, so it holds
                           every changeset before this one too
    legacy_complete        same rule, from the older one-fetch-per-user store
    missing                nothing valid; the builder marks the row history-unknown

    A truncated window fetched for a *later* target is never used: its 100 rows can stop
    after this target, and the model would read an experienced mapper as a new one.
    """
    if uid is None or not created_at:
        return None, "missing"
    if conn is not None:
        row = conn.execute("SELECT json FROM histories WHERE uid=? AND before=? AND status=200",
                           (uid, created_at)).fetchone()
        if row:
            return row[0], "exact"
        row = conn.execute(
            "SELECT json FROM histories WHERE uid=? AND before>=? AND status=200 AND truncated=0"
            " ORDER BY before LIMIT 1", (uid, created_at)).fetchone()
        if row:
            return row[0], "complete_later_window"
    if legacy is not None:
        row = legacy.execute(
            "SELECT json FROM histories WHERE uid=? AND status=200 AND truncated=0 AND before>=?",
            (uid, created_at)).fetchone()
        if row:
            return row[0], "legacy_complete"
    return None, "missing"


def targets_by_user(rows) -> dict[int, set[str]]:
    out: dict[int, set[str]] = defaultdict(set)
    for _, uid, created in rows:
        if uid is not None and created:
            out[uid].add(created)
    return out
