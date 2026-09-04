"""User account facts, cached, for the account-age feature.

`GET /api/0.6/user/<uid>` returns `account_created`, which is all we need to compute
account age at the time of an edit. It is one request per *unique* uid, not per
changeset, so an 18k corpus costs a few thousand calls, and the cache makes a rerun free.

**Account age is the only user fact allowed to come from a point-in-time-free source.**
`account_created` is fixed for the life of the account, so reading it today tells the
model nothing about the future. Everything else about a user - edit counts, block
history, how their later edits were received - must be derived from changesets strictly
before the target changeset. See PROJECT_PLAN section 5.4 Rule 2; getting this wrong
means the model learns that the account was blocked in 2017 and calls it a 2016
prediction.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from mapguard.config import OSM_API
from mapguard.data.http import PoliteSession

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    uid             INTEGER PRIMARY KEY,
    status          INTEGER NOT NULL,
    display_name    TEXT,
    account_created TEXT,
    changesets      INTEGER,
    raw_json        TEXT,
    fetched         TEXT DEFAULT (datetime('now'))
)
"""


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(SCHEMA)
    return conn


def parse_user(xml: str) -> dict | None:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    el = root.find("user")
    if el is None:
        return None
    cs = el.find("changesets")
    return {
        "uid": int(el.attrib["id"]),
        "display_name": el.attrib.get("display_name"),
        "account_created": el.attrib.get("account_created"),
        "changesets": int(cs.attrib.get("count", 0)) if cs is not None else None,
    }


def fetch_user(session: PoliteSession, uid: int) -> tuple[dict | None, int]:
    r = session.get(f"{OSM_API}/user/{uid}")
    return (parse_user(r.text) if r.ok else None), r.status


def ensure_users(conn: sqlite3.Connection, session: PoliteSession, uids: list[int]) -> dict:
    """Fetch every uid not already cached. Returns a small progress summary."""
    have = {row[0] for row in conn.execute("SELECT uid FROM users")}
    todo = [u for u in dict.fromkeys(uids) if u not in have and u is not None]
    ok = fail = 0
    for n, uid in enumerate(todo, 1):
        info, status = fetch_user(session, uid)
        if info:
            conn.execute(
                "INSERT OR REPLACE INTO users (uid,status,display_name,account_created,changesets,raw_json)"
                " VALUES (?,?,?,?,?,?)",
                (uid, status, info["display_name"], info["account_created"],
                 info["changesets"], json.dumps(info)),
            )
            ok += 1
        else:
            # A deleted account 404s. That is a fact about the corpus, not an error.
            conn.execute(
                "INSERT OR REPLACE INTO users (uid,status,display_name,account_created,changesets,raw_json)"
                " VALUES (?,?,?,?,?,?)",
                (uid, status, None, None, None, None),
            )
            fail += 1
        if n % 200 == 0:
            print(f"  users {n}/{len(todo)} (ok {ok}, missing {fail})", flush=True)
    return {"requested": len(todo), "ok": ok, "missing": fail, "cached": len(have)}


def account_age_days(account_created: str | None, at: str | None) -> float | None:
    """Days between account creation and the moment of the edit.

    Negative values are impossible and indicate bad data; they are returned as None so
    they show up as missing rather than as a suspiciously predictive feature.
    """
    if not account_created or not at:
        return None
    try:
        a = datetime.fromisoformat(account_created.replace("Z", "+00:00"))
        b = datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if a.tzinfo is None:
        a = a.replace(tzinfo=timezone.utc)
    if b.tzinfo is None:
        b = b.replace(tzinfo=timezone.utc)
    days = (b - a).total_seconds() / 86400.0
    return days if days >= 0 else None
