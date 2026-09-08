"""OSMCha - human-reviewed labels for the modern corpora.

The Ovid labels are revert-derived: a changeset counts as damaging because someone later
reverted it. That is a reasonable proxy and it is what the published benchmark uses, but
it is a proxy. OSMCha labels are different in kind - a human opened the changeset, read
it, and pressed a button. For Corpus B and Corpus D this is the high-precision route
(PROJECT_PLAN section 8.5), with revert mining supplying volume.

Two label classes, and the second one matters more than it looks:

    harmful=True                a reviewer marked it damaging
    checked=True, harmful=False a reviewer looked and said it is fine

The second is a *reviewed* negative, not an assumed one. Treating unreviewed changesets
as clean would bake the reviewers' attention patterns into the negative class - they look
at suspicious things, so anything they never looked at is clean by construction. Reviewed
negatives avoid that; they are scarcer, and worth more.

Auth is a token from the OSMCha user page, sent as `Authorization: Token <token>`. Kept
in a gitignored .env, never in the repository. Without it every endpoint returns 401.

Rate: OSMCha is a small volunteer-run service and the changeset query is expensive - an
unfiltered `harmful=True` page times out. Always bound the query by date.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from mapguard.data.http import PoliteSession

OSMCHA_API = "https://osmcha.org/api/v1"
# The changeset endpoint is heavy. One request every two seconds, and a generous read
# timeout, is the difference between a working harvest and a wall of 504s.
OSMCHA_RATE = 0.5
# 180 s suits the heavy count queries, which scan. Page queries return in seconds, so a
# long timeout there only buys silence: at 4 retries a single refusing window costs ~12
# minutes and a run of them collapses throughput entirely. Harvesting fails fast and
# records the gap; gaps get their own retry pass afterwards, when the service has calmed.
OSMCHA_TIMEOUT = 180
OSMCHA_PAGE_TIMEOUT = 25
OSMCHA_PAGE_RETRIES = 2
PAGE_SIZE = 100


def load_env(path: Path | None = None) -> None:
    """Read a .env into the environment. Absent file is fine; absent token is not."""
    p = path or Path(__file__).resolve().parents[3] / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def token() -> str:
    load_env()
    tok = os.environ.get("OSMCHA_TOKEN", "").strip()
    if not tok:
        raise SystemExit(
            "OSMCHA_TOKEN is not set. Copy .env.example to .env and paste the token from "
            "your OSMCha user page. Every endpoint 401s without it."
        )
    return tok


def session(*, fast: bool = False) -> PoliteSession:
    """`fast=True` for paging: short timeout, few retries, record the gap and move on."""
    return PoliteSession(
        OSMCHA_RATE,
        max_retries=OSMCHA_PAGE_RETRIES if fast else 4,
        timeout=OSMCHA_PAGE_TIMEOUT if fast else OSMCHA_TIMEOUT,
        headers={
            "Authorization": f"Token {token()}",
            "User-Agent": "MapGuard-research/0.2 (academic OSM vandalism-detection study)",
        },
    )


@dataclass
class OsmchaLabel:
    """One reviewed changeset. `harmful` is the label; the rest is provenance."""

    cid: int
    harmful: bool | None
    checked: bool
    check_user: str | None
    check_date: str | None
    date: str | None
    uid: int | None
    user: str | None
    editor: str | None
    comment: str | None
    n_create: int | None
    n_modify: int | None
    n_delete: int | None
    is_suspect: bool | None
    reasons: str          # why OSMCha flagged it, comma-joined
    reviewed_features: int


def _reasons(props: dict) -> str:
    out = []
    for r in props.get("reasons") or []:
        out.append(r.get("name", "") if isinstance(r, dict) else str(r))
    return ",".join(x for x in out if x)


def parse_feature(feat: dict) -> OsmchaLabel | None:
    try:
        cid = int(feat["id"])
    except (KeyError, TypeError, ValueError):
        return None
    p = feat.get("properties", {}) or {}
    return OsmchaLabel(
        cid=cid,
        harmful=p.get("harmful"),
        checked=bool(p.get("checked")),
        check_user=p.get("check_user"),
        check_date=p.get("check_date"),
        date=p.get("date"),
        uid=int(p["uid"]) if str(p.get("uid", "")).isdigit() else None,
        user=p.get("user"),
        editor=p.get("editor"),
        comment=p.get("comment"),
        n_create=p.get("create"),
        n_modify=p.get("modify"),
        n_delete=p.get("delete"),
        is_suspect=p.get("is_suspect"),
        reasons=_reasons(p),
        reviewed_features=len(p.get("reviewed_features") or []),
    )


def iter_labels(
    sess: PoliteSession,
    *,
    harmful: bool,
    date_gte: str,
    date_lte: str | None = None,
    max_pages: int = 100,
) -> Iterator[OsmchaLabel]:
    """Page through reviewed changesets in a date window.

    `harmful=False` is sent together with `checked=True`, otherwise the filter would
    return every unreviewed changeset as a negative - see the module docstring.
    """
    params = {
        "date__gte": date_gte,
        "page_size": PAGE_SIZE,
        "harmful": "True" if harmful else "False",
    }
    if not harmful:
        params["checked"] = "True"
    if date_lte:
        params["date__lte"] = date_lte

    for page in range(1, max_pages + 1):
        r = sess.get(f"{OSMCHA_API}/changesets/", params={**params, "page": page})
        if not r.ok:
            print(f"  page {page}: HTTP {r.status} - stopping this class")
            return
        try:
            payload = json.loads(r.text)
        except ValueError:
            print(f"  page {page}: unparseable response - stopping this class")
            return

        feats = payload.get("features") or []
        if not feats:
            return
        for f in feats:
            label = parse_feature(f)
            if label is not None:
                yield label
        if not payload.get("next"):
            return


def _shift(date: str, days: int) -> str:
    from datetime import datetime, timedelta

    return (datetime.fromisoformat(date) + timedelta(days=days)).strftime("%Y-%m-%d")


def iter_labels_windowed(
    sess: PoliteSession,
    *,
    harmful: bool,
    date_gte: str,
    date_lte: str,
    window_days: int = 7,
    max_pages_per_window: int = 12,
    on_gap=None,
    on_window_done=None,
    skip_windows=None,
) -> Iterator[OsmchaLabel]:
    """Walk the range in date slices instead of paging deep into one query.

    Offset pagination on this endpoint degrades badly: the harmful class for 2026 died
    with a connection timeout at page 34, because every page makes the server scan past
    everything before it. Slicing by date keeps every offset shallow - a week of harmful
    changesets is a few hundred rows - and a failed slice costs one week, not the run.

    A window that fills `max_pages_per_window` is split in half and retried, so a busy
    week degrades into two quiet ones rather than silently truncating.
    """
    cursor = date_gte
    while cursor < date_lte:
        end = min(_shift(cursor, window_days), date_lte)
        if skip_windows and (cursor, end) in skip_windows:
            cursor = end
            continue
        got = 0
        failed = False
        saturated = False
        for page in range(1, max_pages_per_window + 1):
            params = {
                "date__gte": cursor,
                "date__lte": end,
                "page_size": PAGE_SIZE,
                "page": page,
                "harmful": "True" if harmful else "False",
            }
            if not harmful:
                params["checked"] = "True"
            r = sess.get(f"{OSMCHA_API}/changesets/", params=params)
            if not r.ok:
                # A skipped window is a hole in the dataset. Record it so coverage can be
                # measured and the window retried, rather than discovered as a gap later.
                print(f"  {cursor}..{end} page {page}: HTTP {r.status}, recording gap")
                if on_gap is not None:
                    on_gap(cursor, end, "harmful" if harmful else "clean", r.status)
                failed = True
                break
            try:
                payload = json.loads(r.text)
            except ValueError:
                break
            feats = payload.get("features") or []
            for f in feats:
                label = parse_feature(f)
                if label is not None:
                    yield label
                    got += 1
            if not payload.get("next"):
                break
        else:
            # The for-loop ran to exhaustion rather than breaking, which means the page
            # budget ran out before the data did: this window is saturated.
            saturated = True

        if failed:
            pass                      # gap already recorded; do not mark it complete
        elif saturated and window_days > 1:
            print(f"  {cursor}..{end} saturated at {got}, splitting")
            yield from iter_labels_windowed(
                sess, harmful=harmful, date_gte=cursor, date_lte=end,
                window_days=max(1, window_days // 2),
                max_pages_per_window=max_pages_per_window,
                on_gap=on_gap, on_window_done=on_window_done, skip_windows=skip_windows,
            )
        elif on_window_done is not None:
            # Recorded so a restart resumes instead of re-walking three years of slices.
            # At roughly 225 rows/min the harvest cannot afford to start over each time
            # it drops.
            on_window_done(cursor, end, "harmful" if harmful else "clean")

        cursor = end


def count_for(sess: PoliteSession, *, harmful: bool, date_gte: str, date_lte: str | None = None) -> int | None:
    """How many reviewed changesets match, without paging through them."""
    params = {
        "date__gte": date_gte,
        "page_size": 1,
        "harmful": "True" if harmful else "False",
    }
    if not harmful:
        params["checked"] = "True"
    if date_lte:
        params["date__lte"] = date_lte
    r = sess.get(f"{OSMCHA_API}/changesets/", params=params)
    if not r.ok:
        return None
    try:
        return json.loads(r.text).get("count")
    except ValueError:
        return None
