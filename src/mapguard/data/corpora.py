"""Where each corpus lives on disk, and the rules shared by the fetcher and the builder.

Anything both `scripts/fetch_osm.py` and `mapguard.features.build_store` must agree on
lives here. If they disagreed on which N rows count, or which history a target may use,
the feature store would quietly describe a different dataset from the one fetched.

    A   Ovid benchmark, 2014-2019, revert-derived labels     corpus_a_meta + corpus_a
    M   OSMCha-reviewed, 2023-2026                           corpus_mn (corpus = M)
    N   random unreviewed changesets from M's quarters       corpus_mn (corpus = N)
    R   revert-derived, 2021-09..2026-06, Ovid's definition  corpus_mn (corpus = R),
        mined from the changeset dump by scripts/mine_changesets.py

Corpus R label rules (applied in `valid_selection`, after metadata is fetched):

    positive  reverted by a *different* account within R_MAX_DELAY_DAYS of being made,
              by a whole-changeset revert (partial reverts and reworks are excluded)
    negative  a hash-sampled changeset nobody reverted, from the same quarters, one per
              positive per quarter (balanced, as in the Ovid benchmark)
    both      closed, non-empty, created in [since, censor) where censor = dump date minus
              R_CENSOR_DAYS, so every changeset had time to be reverted - otherwise the
              newest negatives would be "not reverted *yet*"
    both      at most R_PER_AUTHOR changesets per author per class, so one mass-reverted
              account or one prolific mapper cannot define a class

Two strata, because Ovid's rule is narrower than "reverted":

    random             positives drawn uniformly from all reverted changesets (task `revert`)
    vandal_supplement  extra positives whose revert comment *mentions vandalism* - Ovid's own
                       rule (Tempelmeier & Demidova 2021, section 4.1) - plus matching
                       negatives. `revert_vandalism` and `ovid_modern` use every positive with
                       the vandalism flag, from both strata.
"""

from __future__ import annotations

import sqlite3
import re
from collections import defaultdict

# Ovid kept only reverts whose comment mentions vandalism. The mined comments are
# multilingual, so the stem is matched in the scripts it actually appears in.
VANDAL_MENTION = re.compile(r"vandal|wandal|вандал|vandál|βανδαλ", re.I)

CORPUS_MN_DB = "raw/corpus_mn.sqlite"
LABELS_DB = "raw/osmcha_labels.sqlite"
USERS_DB = "raw/users.sqlite"
HISTORY_DB = "raw/user_histories_pit.sqlite"
LEGACY_HISTORY_DB = "raw/user_histories.sqlite"
VERSIONS_DB = "raw/prev_versions.sqlite"

SEED = 20260904
M_PER_CLASS = 4000
N_TOTAL = 4000

R_POSITIVES = 6000          # valid positives wanted; negatives match them per quarter
R_OVERSAMPLE = 1.5          # candidates selected per wanted positive (self-reverts etc. drop out)
R_MAX_DELAY_DAYS = 90
R_CENSOR_DAYS = 90
R_PER_AUTHOR = 5
R_VANDAL_POSITIVES = 6000   # vandalism-flagged positives wanted across both strata


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
    has_stratum = any(r[1] == "stratum" for r in conn.execute("PRAGMA table_info(selection)"))
    stratum = "COALESCE(s.stratum, 'random')" if has_stratum else "'random'"
    rows = conn.execute(
        "SELECT s.cid, s.corpus, s.label, s.quarter, m.status, m.closed_at, m.num_changes,"
        f" m.created_at, m.uid, {stratum} FROM selection s LEFT JOIN meta m ON m.cid = s.cid"
        " ORDER BY s.rank").fetchall()
    for cid, corpus, label, q, status, closed, n, created, _, _ in rows:
        if status != 200:
            continue
        if corpus == "M":
            out[cid] = ("M", int(label))
        elif corpus == "N" and (closed and (n or 0) > 0 and quarter_of(created) == q
                                and quota.get(q, 0) > 0):
            quota[q] -= 1
            out[cid] = ("N", 0)
    out.update(_valid_r(conn, rows))
    return out


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def _days(a: str, b: str) -> float:
    from datetime import datetime

    fa = datetime.fromisoformat(a.replace("Z", "+00:00"))
    fb = datetime.fromisoformat(b.replace("Z", "+00:00"))
    return (fb - fa).total_seconds() / 86400.0


def _valid_r(conn: sqlite3.Connection, rows) -> dict[int, tuple[str, int]]:
    """Corpus R rows that satisfy the label rules in the module docstring.

    Each stratum is validated on its own: positives first (in hash order), then negatives up
    to one per valid positive per quarter. The per-author cap is shared across strata.
    """
    if not _table_exists(conn, "r_targets") or not _table_exists(conn, "r_window"):
        return {}
    window = dict(conn.execute("SELECT k, v FROM r_window"))
    since, censor = window.get("since"), window.get("censor")
    reverter = {c: (created, uid) for c, created, uid in
                conn.execute("SELECT cid, reverter_created, reverter_uid FROM r_targets")}
    out: dict[int, tuple[str, int]] = {}
    per_author: dict[tuple[int, int], int] = {}

    def usable(status, closed, n, created, uid):
        return (status == 200 and closed and (n or 0) > 0 and created
                and since <= created < censor and uid is not None)

    def take(cid, label, uid):
        key = (uid, label)
        if per_author.get(key, 0) >= R_PER_AUTHOR:
            return False
        per_author[key] = per_author.get(key, 0) + 1
        out[cid] = ("R", label)
        return True

    def stratum_of(row):
        return row[9] if len(row) > 9 else "random"

    for stratum, wanted in (("random", R_POSITIVES), ("vandal_supplement", None)):
        quota: dict[str, int] = {}
        for row in rows:
            cid, corpus, label, _, status, closed, n, created, uid = row[:9]
            if (corpus != "R" or label != 1 or stratum_of(row) != stratum
                    or (wanted is not None and wanted <= 0)
                    or not usable(status, closed, n, created, uid)):
                continue
            r_created, r_uid = reverter.get(cid, (None, None))
            if r_created is None or r_uid == uid:        # self-reverts are the author's own fix
                continue
            if not 0 <= _days(created, r_created) <= R_MAX_DELAY_DAYS:
                continue
            if take(cid, 1, uid):
                q = quarter_of(created)
                quota[q] = quota.get(q, 0) + 1
                if wanted is not None:
                    wanted -= 1
        for row in rows:
            cid, corpus, label, _, status, closed, n, created, uid = row[:9]
            if (corpus != "R" or label != 0 or stratum_of(row) != stratum
                    or not usable(status, closed, n, created, uid)):
                continue
            q = quarter_of(created)
            if quota.get(q, 0) > 0 and take(cid, 0, uid):
                quota[q] -= 1
    return out


def r_row_info(conn: sqlite3.Connection) -> dict[int, tuple[str, int]]:
    """cid -> (stratum, revert_mentions_vandalism) for corpus R selection rows."""
    if not _table_exists(conn, "r_targets"):
        return {}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(r_targets)")}
    has_stratum = any(r[1] == "stratum" for r in conn.execute("PRAGMA table_info(selection)"))
    vandal = (dict(conn.execute("SELECT cid, COALESCE(vandal, 0) FROM r_targets"))
              if "vandal" in cols else {})
    strata = (dict(conn.execute("SELECT cid, COALESCE(stratum, 'random') FROM selection WHERE corpus='R'"))
              if has_stratum else {c: "random" for (c,) in conn.execute(
                  "SELECT cid FROM selection WHERE corpus='R'")})
    return {c: (s, int(vandal.get(c, 0))) for c, s in strata.items()}


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
