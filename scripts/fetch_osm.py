"""Every OSM API fetch the training data still needs, in one resumable tool.

Subcommands, in run order (each is resumable and safe to re-run):

    select     choose corpus M (OSMCha-reviewed) and N (random unreviewed) changesets,
               and/or corpus R (revert-derived, 2021-09..) from scripts/mine_changesets.py
    changesets fetch metadata and osmChange contents for M and N
    users      account creation dates for every author in A, M and N
    histories  point-in-time changeset history for every target changeset
    versions   the previous version of every modified/deleted object (tag diffs, node moves)
    status     progress of all of the above

    python scripts/fetch_osm.py select --corpora R        # after mine_changesets.py
    python scripts/fetch_osm.py select --corpora MN       # after the OSMCha harvest
    python scripts/fetch_osm.py changesets --rate 2
    python scripts/fetch_osm.py users
    python scripts/fetch_osm.py histories --rate 1.5
    python scripts/fetch_osm.py versions --rate 1
    python scripts/fetch_osm.py status

Why a previous-version route exists. Overpass adiff gives full before/after way geometry,
but at a measured ~122 s per changeset it covers ~1,000 of 18,276 Corpus A changesets in
the time available. The OSM API multi-fetch endpoint returns 725 exact `id vN` states per
call, so the previous version of every touched object is affordable for every corpus. That
gives exact tag diffs for every object, exact displacement for every node, and node-count
change for every way. Way length and area still come only from adiff, and the builder
records which route each edit's geometry came from.

Point-in-time user histories. The API returns at most 100 changesets, newest first. One
fetch per user at their *latest* target covers every earlier target exactly when it
returns fewer than 100 rows (the history is complete). When it is truncated, each other
target of that user gets its own fetch, so no target sees a history cut off by a later
target's window.

Sampling is by a hash of the changeset id, never by size, author or content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import random
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapguard.config import (  # noqa: E402
    CORPUS_A_DB,
    CORPUS_A_META_DB,
    MULTIFETCH_MAX,
    OSM_API,
    data_path,
)
from mapguard.data.corpora import (  # noqa: E402
    CORPUS_MN_DB,
    HISTORY_DB,
    LABELS_DB,
    M_PER_CLASS,
    N_TOTAL,
    R_CENSOR_DAYS,
    R_OVERSAMPLE,
    R_POSITIVES,
    SEED,
    USERS_DB,
    VERSIONS_DB,
    quarter_of,
    valid_selection,
)
from mapguard.data.http import PoliteSession  # noqa: E402
from mapguard.data.osm_api import fetch_changeset_download, parse_changeset_metas  # noqa: E402
from mapguard.data.osmchange import parse_safe  # noqa: E402
from mapguard.data.store import SCHEMA as STORE_SCHEMA  # noqa: E402
from mapguard.features.user_history import HISTORY_LIMIT, fetch_user_history  # noqa: E402

N_OVERSAMPLE = 1.6
VERSIONS_PER_CHANGESET = 2000     # cap per changeset; coverage is recorded, not assumed
UNRESOLVED_GROUP = 32
MAX_PARAM_CHARS = 5000            # the API answered 414 URI Too Long above ~8,000 URL characters (commas are sent as %2C)

SELECTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS selection (
    cid      INTEGER PRIMARY KEY,
    corpus   TEXT NOT NULL,        -- M | N
    label    INTEGER,              -- M: 1 harmful / 0 reviewed-clean.  N: 0 presumed clean
    quarter  TEXT NOT NULL,
    rank     TEXT NOT NULL         -- sha256(seed:cid), the deterministic sampling order
);
"""
USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    uid             INTEGER PRIMARY KEY,
    status          INTEGER NOT NULL,
    display_name    TEXT,
    account_created TEXT,
    changesets      INTEGER,
    raw_json        TEXT,
    fetched         TEXT DEFAULT (datetime('now'))
);
"""
HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS histories (
    uid       INTEGER NOT NULL,
    before    TEXT NOT NULL,
    status    INTEGER NOT NULL,
    n         INTEGER,
    truncated INTEGER,
    json      TEXT,
    fetched   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (uid, before)
);
"""
VERSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS wanted (
    otype   TEXT NOT NULL,
    oid     INTEGER NOT NULL,
    version INTEGER NOT NULL,
    PRIMARY KEY (otype, oid, version)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS versions (
    otype     TEXT NOT NULL,
    oid       INTEGER NOT NULL,
    version   INTEGER NOT NULL,
    status    INTEGER NOT NULL,
    visible   INTEGER,
    lat       REAL,
    lon       REAL,
    n_refs    INTEGER,
    tags_json TEXT,
    PRIMARY KEY (otype, oid, version)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS planned (
    corpus    TEXT NOT NULL,
    cid       INTEGER NOT NULL,
    n_edits   INTEGER,
    n_wanted  INTEGER,
    capped    INTEGER,
    PRIMARY KEY (corpus, cid)
) WITHOUT ROWID;
"""
R_SCHEMA = """
CREATE TABLE IF NOT EXISTS r_targets (
    cid              INTEGER PRIMARY KEY,   -- the reverted changeset
    reverter_cid     INTEGER NOT NULL,      -- earliest whole-changeset revert of it
    reverter_created TEXT NOT NULL,
    reverter_uid     INTEGER,
    kind             TEXT                   -- osm-revert | josm-reverter | simple_revert | comment
);
CREATE TABLE IF NOT EXISTS r_window (k TEXT PRIMARY KEY, v TEXT);
"""
MN_SCHEMA = (SELECTION_SCHEMA + STORE_SCHEMA["meta"] + ";" + STORE_SCHEMA["contents"] + ";"
             + R_SCHEMA)
DUMP_DB = "raw/changeset_dump.sqlite"


def connect(rel: str, schema: str) -> sqlite3.Connection:
    conn = sqlite3.connect(data_path(*rel.split("/")), timeout=120, isolation_level=None,
                           check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(schema)
    return conn


def read_only(rel: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{data_path(*rel.split('/'))}?mode=ro", uri=True,
                           timeout=120, check_same_thread=False)


def rank(cid: int) -> str:
    return hashlib.sha256(f"{SEED}:{int(cid)}".encode()).hexdigest()


def run_pool(items, fn, write, *, label: str, workers: int, total: int | None = None):
    """Threads call fn(item) under the shared token bucket; only the main thread writes."""
    work: queue.Queue = queue.Queue(maxsize=workers * 4)
    out: queue.Queue = queue.Queue()

    def feeder():
        for it in items:
            work.put(it)
        for _ in range(workers):
            work.put(None)

    def worker():
        while True:
            it = work.get()
            if it is None:
                return
            try:
                out.put(("ok", fn(it)))
            except Exception as exc:          # one bad item must not kill a five-hour run
                out.put(("error", f"{repr(it)[:80]}: {exc!r}"[:300]))

    feed = threading.Thread(target=feeder, daemon=True)
    workers_t = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    feed.start()
    for t in workers_t:
        t.start()

    started, n, errors = time.monotonic(), 0, 0
    while any(t.is_alive() for t in workers_t) or not out.empty():
        try:
            kind, res = out.get(timeout=1.0)
        except queue.Empty:
            continue
        if kind == "error":
            errors += 1
            print(f"  error {res}", flush=True)
            continue
        write(res)
        n += 1
        if n % 100 == 0:
            rate = n / max(1e-6, time.monotonic() - started)
            eta = f"  eta {(total - n) / max(rate, 1e-6) / 60:.0f} min" if total else ""
            print(f"  {label}: {n}{'/' + str(total) if total else ''}  {rate:.2f}/s{eta}", flush=True)
    print(f"  {label}: finished {n}, errors {errors}", flush=True)
    return n, errors


# ------------------------------------------------------------------------- select


def cmd_select(args) -> int:
    conn = connect(CORPUS_MN_DB, MN_SCHEMA)
    corpora = {c.strip().upper() for c in args.corpora.split(",")}
    if "R" in corpora:
        select_r(conn, args)
    if "MN" in corpora:
        select_mn(conn, args)
    return 0


def select_r(conn: sqlite3.Connection, args) -> None:
    """Corpus R candidates from the mined dump. Label rules are applied after metadata is
    fetched, in mapguard.data.corpora._valid_r; this only picks candidates, by hash."""
    import bisect
    from datetime import datetime, timedelta

    if conn.execute("SELECT COUNT(*) FROM selection WHERE corpus='R'").fetchone()[0] and not args.force:
        print("corpus R selection exists; pass --force to rebuild it (only before fetching)")
        return
    conn.execute("DELETE FROM selection WHERE corpus='R'")
    conn.execute("DELETE FROM r_targets")
    dump = read_only(DUMP_DB)
    src = dict(dump.execute("SELECT k, v FROM source"))
    since = src["since"]
    censor = (datetime.fromisoformat(src["until"]) - timedelta(days=R_CENSOR_DAYS)).strftime("%Y-%m-%d")
    conn.executemany("INSERT OR REPLACE INTO r_window VALUES (?, ?)",
                     [("since", since), ("censor", censor), ("dump_file", src["file"]),
                      ("dump_last_modified", src["last_modified"])])

    # cid -> date from the 1-in-400 sample: ids are monotone in time, so interpolation is
    # accurate to hours. Used only to place candidates in quarters; exact dates come later.
    sample = dump.execute("SELECT cid, created_at FROM sample ORDER BY cid").fetchall()
    cids = [c for c, _ in sample]

    def approx_date(cid: int) -> str | None:
        i = bisect.bisect_left(cids, cid)
        if i == 0 or i >= len(cids):
            return None
        return sample[i][1]

    reverts = dump.execute("SELECT cid, created_at, uid, kind, targets, partial FROM reverts"
                           " ORDER BY created_at").fetchall()
    reverter_ids = {r[0] for r in reverts}
    ever_reverted: set[int] = set()
    first_revert: dict[int, tuple] = {}
    for cid, created, uid, kind, targets, partial in reverts:
        ts = json.loads(targets)
        ever_reverted.update(ts)
        if partial:
            continue
        for t in ts:
            if t not in reverter_ids and t not in first_revert:
                first_revert[t] = (cid, created, uid, kind)

    pools: dict[str, list[int]] = defaultdict(list)
    for t in first_revert:
        d = approx_date(t)
        if d and since <= d < censor:
            pools[quarter_of(d)].append(t)
    quarters = sorted(pools)
    want = int(R_POSITIVES * R_OVERSAMPLE)
    take = {q: 0 for q in quarters}
    open_q = [q for q in quarters]
    avail = {q: sorted(pools[q], key=rank) for q in quarters}
    while want > 0 and open_q:
        share = max(1, want // len(open_q))
        for q in list(open_q):
            add = min(share, len(avail[q]) - take[q], want)
            take[q] += add
            want -= add
            if take[q] >= len(avail[q]):
                open_q.remove(q)
            if want <= 0:
                break
    pos_rows, tgt_rows = [], []
    for q in quarters:
        for t in avail[q][: take[q]]:
            pos_rows.append((t, "R", 1, q, rank(t)))
            r = first_revert[t]
            tgt_rows.append((t, r[0], r[1], r[2], r[3]))
    conn.executemany("INSERT INTO selection (cid,corpus,label,quarter,rank) VALUES (?,?,?,?,?)", pos_rows)
    conn.executemany("INSERT INTO r_targets VALUES (?,?,?,?,?)", tgt_rows)

    # negatives: sampled changesets nobody reverted, same quarters, ~1.2x positives each
    neg_pool: dict[str, list[int]] = defaultdict(list)
    for cid, created, closed, n in dump.execute(
            "SELECT cid, created_at, closed_at, num_changes FROM sample"):
        if (since <= created < censor and closed and (n or 0) > 0
                and cid not in ever_reverted and cid not in reverter_ids):
            neg_pool[quarter_of(created)].append(cid)
    neg_rows = []
    for q in quarters:
        k = int(round(take[q] / R_OVERSAMPLE * 1.3))
        neg_rows += [(c, "R", 0, q, rank(c)) for c in sorted(neg_pool.get(q, []), key=rank)[:k]]
    conn.executemany("INSERT INTO selection (cid,corpus,label,quarter,rank) VALUES (?,?,?,?,?)", neg_rows)
    print(json.dumps({
        "corpus_R": {"window": [since, censor], "revert_changesets": len(reverts),
                     "reverted_targets_in_window": sum(len(v) for v in pools.values()),
                     "positive_candidates": len(pos_rows), "negative_candidates": len(neg_rows)},
        "by_quarter": [list(r) for r in conn.execute(
            "SELECT quarter, label, COUNT(*) FROM selection WHERE corpus='R' GROUP BY 1,2")],
    }))


def select_mn(conn: sqlite3.Connection, args) -> None:
    labels = read_only(LABELS_DB)
    if conn.execute("SELECT COUNT(*) FROM selection WHERE corpus IN ('M','N')").fetchone()[0] \
            and not args.force:
        print("corpus M/N selection exists; pass --force to rebuild it (only before fetching)")
        return
    conn.execute("DELETE FROM selection WHERE corpus IN ('M','N')")

    rows = labels.execute(
        "SELECT cid, harmful, date FROM labels WHERE checked = 1 AND harmful IS NOT NULL"
        " AND date IS NOT NULL"
        " AND COALESCE(n_create,0)+COALESCE(n_modify,0)+COALESCE(n_delete,0) > 0"
        " AND date >= ? AND date < ?", (args.since, args.until)).fetchall()
    labelled = {r[0] for r in labels.execute("SELECT cid FROM labels")}

    pools: dict[tuple[str, int], list[int]] = defaultdict(list)
    cid_range: dict[str, list[int]] = {}
    for cid, harmful, date in rows:
        q = quarter_of(date)
        pools[(q, int(harmful))].append(cid)
        lo_hi = cid_range.setdefault(q, [cid, cid])
        lo_hi[0], lo_hi[1] = min(lo_hi[0], cid), max(lo_hi[1], cid)
    quarters = sorted(cid_range)

    # corpus M: water-fill each class across quarters so no quarter dominates
    chosen_m: dict[tuple[str, int], list[int]] = {}
    for label in (1, 0):
        avail = {q: sorted(pools.get((q, label), []), key=rank) for q in quarters}
        take = {q: 0 for q in quarters}
        left = M_PER_CLASS
        open_q = [q for q in quarters if avail[q]]
        while left > 0 and open_q:
            share = max(1, left // len(open_q))
            for q in list(open_q):
                add = min(share, len(avail[q]) - take[q], left)
                take[q] += add
                left -= add
                if take[q] >= len(avail[q]):
                    open_q.remove(q)
                if left <= 0:
                    break
        for q in quarters:
            chosen_m[(q, label)] = avail[q][: take[q]]
            conn.executemany(
                "INSERT INTO selection (cid,corpus,label,quarter,rank) VALUES (?,?,?,?,?)",
                [(c, "M", label, q, rank(c)) for c in chosen_m[(q, label)]])

    # corpus N: random ids inside each quarter's id range, date-matched to M's positives
    n_harm = sum(len(chosen_m[(q, 1)]) for q in quarters)
    rng = random.Random(SEED)
    taken = set(labelled)
    for q in quarters:
        want = int(round(N_TOTAL * len(chosen_m[(q, 1)]) / max(1, n_harm) * N_OVERSAMPLE))
        lo, hi = cid_range[q]
        got: list[int] = []
        while len(got) < want:
            c = rng.randint(lo, hi)
            if c not in taken:
                taken.add(c)
                got.append(c)
        conn.executemany(
            "INSERT INTO selection (cid,corpus,label,quarter,rank) VALUES (?,?,?,?,?)",
            [(c, "N", 0, q, rank(c)) for c in got])

    print(json.dumps({
        "selection": [list(r) for r in conn.execute(
            "SELECT corpus, label, COUNT(*) FROM selection GROUP BY 1,2")],
        "by_quarter": [list(r) for r in conn.execute(
            "SELECT quarter, corpus, label, COUNT(*) FROM selection WHERE corpus IN ('M','N')"
            " GROUP BY 1,2,3")],
    }))


# --------------------------------------------------------------------- changesets


def cmd_changesets(args) -> int:
    conn = connect(CORPUS_MN_DB, MN_SCHEMA)
    sess = PoliteSession(args.rate)

    have_meta = {r[0] for r in conn.execute("SELECT cid FROM meta WHERE status IN (200, 404, 410)")}
    todo = [r[0] for r in conn.execute("SELECT cid FROM selection ORDER BY rank")
            if r[0] not in have_meta]
    batches = [todo[i:i + 100] for i in range(0, len(todo), 100)]
    print(f"metadata: {len(todo)} to fetch in {len(batches)} batch calls", flush=True)

    def get_meta(batch):
        # /changesets?changesets=... returns up to 100 per call; ids it omits do not exist
        # (or are hidden), and are recorded as 404 so they are not asked for again.
        r = sess.get(f"{OSM_API}/changesets", params={"changesets": ",".join(map(str, batch))})
        if not r.ok:
            return []                          # transient; retried on the next run
        got = {m.cid: m for m in parse_changeset_metas(r.text)}
        rows = []
        for cid in batch:
            meta = got.get(cid)
            if meta is None:
                rows.append((cid, 404) + (None,) * 13 + (round(r.seconds, 2), "absent from batch"))
            else:
                rows.append((cid, 200, meta.created_at, meta.closed_at, meta.uid, meta.user,
                             meta.num_changes, meta.min_lat, meta.min_lon, meta.max_lat, meta.max_lon,
                             meta.comments_count, meta.comment, meta.editor,
                             json.dumps(meta.tags, ensure_ascii=False), round(r.seconds, 2), None))
        return rows

    run_pool(batches, get_meta, lambda rows: conn.executemany(
        "INSERT OR REPLACE INTO meta (cid,status,created_at,closed_at,uid,user,num_changes,"
        "min_lat,min_lon,max_lat,max_lon,comments_count,comment,editor,tags_json,seconds,error)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows),
        label="meta batches", workers=args.workers, total=len(batches))

    valid = valid_selection(conn)
    have = {r[0] for r in conn.execute("SELECT cid FROM contents WHERE status = 200")}
    todo = [c for c in valid if c not in have]
    print(f"contents: {len(todo)} to fetch ({len(valid)} valid selected)", flush=True)

    def get_contents(cid):
        r = fetch_changeset_download(sess, cid)
        return (cid, r.status, r.text if r.ok else None, len(r.content), round(r.seconds, 2), r.error)

    run_pool(todo, get_contents, lambda row: conn.execute(
        "INSERT OR REPLACE INTO contents (cid,status,xml,bytes,seconds,error) VALUES (?,?,?,?,?,?)", row),
        label="contents", workers=args.workers, total=len(todo))
    return 0


# -------------------------------------------------------------------------- users


def all_targets() -> list[tuple[int, int, str]]:
    """(cid, uid, created_at) for every changeset that will be featurised."""
    out = list(read_only(CORPUS_A_META_DB).execute(
        "SELECT cid, uid, created_at FROM meta WHERE status=200 AND uid IS NOT NULL"
        " AND created_at IS NOT NULL"))
    if data_path(*CORPUS_MN_DB.split("/")).exists():
        mn = read_only(CORPUS_MN_DB)
        try:
            valid = valid_selection(mn)
            out += [r for r in mn.execute(
                "SELECT cid, uid, created_at FROM meta WHERE status=200"
                " AND uid IS NOT NULL AND created_at IS NOT NULL") if r[0] in valid]
        except sqlite3.OperationalError:
            pass
    return out


def _user_fields(el):
    cs = el.find("changesets")
    return (el.attrib.get("display_name"), el.attrib.get("account_created"),
            int(cs.attrib.get("count", 0)) if cs is not None else None)


def cmd_users(args) -> int:
    conn = connect(USERS_DB, USERS_SCHEMA)
    have = {r[0] for r in conn.execute("SELECT uid FROM users")}
    uids = sorted({u for _, u, _ in all_targets()} - have)
    batches = [uids[i:i + 100] for i in range(0, len(uids), 100)]
    print(f"users: {len(uids)} to fetch in {len(batches)} calls", flush=True)
    sess = PoliteSession(args.rate)

    def get(batch):
        r = sess.get(f"{OSM_API}/users", params={"users": ",".join(map(str, batch))})
        found = {}
        if r.ok:
            for el in ET.fromstring(r.text).findall("user"):
                found[int(el.attrib["id"])] = _user_fields(el)
        elif r.status in (400, 404, 410):
            for u in batch:                   # one bad id can poison a batch
                rr = sess.get(f"{OSM_API}/user/{u}")
                if rr.ok:
                    el = ET.fromstring(rr.text).find("user")
                    if el is not None:
                        found[u] = _user_fields(el)
        else:
            return []                         # transient; the next run retries the batch
        return [(u, 200, *found[u]) if u in found else (u, 404, None, None, None) for u in batch]

    run_pool(batches, get, lambda rows: conn.executemany(
        "INSERT OR REPLACE INTO users (uid,status,display_name,account_created,changesets)"
        " VALUES (?,?,?,?,?)", rows), label="user batches", workers=args.workers, total=len(batches))
    print(dict(conn.execute("SELECT status, COUNT(*) FROM users GROUP BY 1")))
    return 0


# ---------------------------------------------------------------------- histories


def cmd_histories(args) -> int:
    conn = connect(HISTORY_DB, HISTORY_SCHEMA)
    by_user: dict[int, set[str]] = defaultdict(set)
    for _, uid, created in all_targets():
        by_user[uid].add(created)
    sess = PoliteSession(args.rate)

    def get(item):
        uid, before = item
        hist, status = fetch_user_history(sess, uid, before)
        return (uid, before, status, len(hist), 1 if len(hist) >= HISTORY_LIMIT else 0,
                json.dumps([h.__dict__ for h in hist], ensure_ascii=False) if status == 200 else None)

    def write(row):
        if row[2] == 0 or 500 <= row[2] < 600 or row[2] in (429, 509):
            return                            # transient: left for the next run
        conn.execute("INSERT OR REPLACE INTO histories (uid,before,status,n,truncated,json)"
                     " VALUES (?,?,?,?,?,?)", row)

    for pass_no in (1, 2):
        stored: dict[int, dict] = defaultdict(dict)
        for uid, before, status, trunc in conn.execute(
                "SELECT uid, before, status, truncated FROM histories"):
            stored[uid][before] = (status, trunc)
        todo = []
        for uid, targets in by_user.items():
            latest = max(targets)
            if pass_no == 1:
                if latest not in stored[uid]:
                    todo.append((uid, latest))
            else:
                first = stored[uid].get(latest)
                if first and first[0] == 200 and first[1]:
                    todo += [(uid, t) for t in sorted(targets) if t not in stored[uid]]
        print(f"histories pass {pass_no}: {len(todo)} fetches for {len(by_user)} users", flush=True)
        run_pool(todo, get, write, label=f"histories p{pass_no}", workers=args.workers, total=len(todo))
    print(dict(conn.execute("SELECT status, COUNT(*) FROM histories GROUP BY 1")))
    return 0


# ----------------------------------------------------------------------- versions


def iter_contents():
    a = read_only(CORPUS_A_DB)
    for cid, xml in a.execute("SELECT id, xml FROM changeset WHERE status=200 ORDER BY id"):
        yield "A", cid, xml
    if data_path(*CORPUS_MN_DB.split("/")).exists():
        mn = read_only(CORPUS_MN_DB)
        try:
            for cid, xml in mn.execute("SELECT cid, xml FROM contents WHERE status=200 ORDER BY cid"):
                yield "MN", cid, xml
        except sqlite3.OperationalError:
            return


def plan_versions(conn: sqlite3.Connection) -> None:
    planned = {(c, i) for c, i in conn.execute("SELECT corpus, cid FROM planned")}
    t0, n_new = time.monotonic(), 0
    conn.execute("BEGIN")
    for corpus, cid, xml in iter_contents():
        if (corpus, cid) in planned:
            continue
        edits, _ = parse_safe(xml)
        want = [(e.otype, e.oid, e.version - 1) for e in edits
                if e.action in ("modify", "delete") and e.version and e.version > 1]
        capped = len(want) > VERSIONS_PER_CHANGESET
        want = want[:VERSIONS_PER_CHANGESET]
        conn.executemany("INSERT OR IGNORE INTO wanted VALUES (?,?,?)", want)
        conn.execute("INSERT INTO planned VALUES (?,?,?,?,?)",
                     (corpus, cid, len(edits), len(want), int(capped)))
        n_new += 1
        if n_new % 1000 == 0:
            conn.execute("COMMIT")
            conn.execute("BEGIN")
            print(f"  planned {n_new} changesets  {time.monotonic() - t0:.0f}s", flush=True)
    conn.execute("COMMIT")
    total = conn.execute("SELECT COUNT(*) FROM wanted").fetchone()[0]
    print(f"planned {n_new} new changesets; {total} object versions wanted", flush=True)


def cmd_versions(args) -> int:
    conn = connect(VERSIONS_DB, VERSIONS_SCHEMA)
    plan_versions(conn)
    if args.plan_only:
        return 0

    sess = PoliteSession(args.rate)
    reader = sqlite3.connect(data_path(*VERSIONS_DB.split("/")), timeout=120, check_same_thread=False)
    todo_n = reader.execute(
        "SELECT COUNT(*) FROM wanted w WHERE NOT EXISTS (SELECT 1 FROM versions v"
        " WHERE v.otype=w.otype AND v.oid=w.oid AND v.version=w.version)").fetchone()[0]
    calls = todo_n // MULTIFETCH_MAX + 1
    print(f"versions: {todo_n} to fetch (~{calls} calls)", flush=True)
    plural = {"node": "nodes", "way": "ways", "relation": "relations"}

    def batches():
        for otype in ("node", "way", "relation"):
            last = (-1, -1)
            while True:
                rows = reader.execute(
                    "SELECT oid, version FROM wanted w WHERE otype=? AND (oid, version) > (?, ?)"
                    " AND NOT EXISTS (SELECT 1 FROM versions v WHERE v.otype=w.otype"
                    " AND v.oid=w.oid AND v.version=w.version)"
                    " ORDER BY oid, version LIMIT ?", (otype, *last, MULTIFETCH_MAX)).fetchall()
                if not rows:
                    break
                last = tuple(rows[-1])
                # 725 ten-digit node ids overflow the API's URL limit (HTTP 414, measured),
                # so split by encoded length as well as by count.
                batch, length = [], 0
                for oid, version in rows:
                    item = len(f"{oid}v{version},")
                    if batch and length + item > MAX_PARAM_CHARS:
                        yield otype, batch
                        batch, length = [], 0
                    batch.append((oid, version))
                    length += item
                if batch:
                    yield otype, batch

    def fetch(otype, refs):
        r = sess.get(f"{OSM_API}/{plural[otype]}",
                     params={plural[otype]: ",".join(f"{i}v{v}" for i, v in refs)})
        if r.ok:
            got = {}
            for el in ET.fromstring(r.text).findall(otype):
                key = (int(el.attrib["id"]), int(el.attrib["version"]))
                tags = {t.attrib["k"]: t.attrib["v"] for t in el.findall("tag")}
                n_refs = len(el.findall("nd")) if otype == "way" else len(el.findall("member"))
                got[key] = (otype, key[0], key[1], 200,
                            0 if el.attrib.get("visible") == "false" else 1,
                            float(el.attrib["lat"]) if "lat" in el.attrib else None,
                            float(el.attrib["lon"]) if "lon" in el.attrib else None,
                            n_refs, json.dumps(tags, ensure_ascii=False))
            return [got.get((i, v), (otype, i, v, 404, None, None, None, None, None)) for i, v in refs]
        if r.status == 414 and len(refs) > 1:
            mid = len(refs) // 2
            return fetch(otype, refs[:mid]) + fetch(otype, refs[mid:])
        if r.status in (400, 403, 404, 410):
            # One redacted or missing version fails the whole batch (measured: node
            # 4039437873 v1 is moderator-hidden - 403 alone, 404 inside a batch). Hidden
            # versions come in runs of consecutive ids from one changeset, so a binary split
            # costs ~2 log2(n) calls per hidden id. Split 8 ways instead, and give up on
            # groups of <= UNRESOLVED_GROUP: they are stored as 499 (not fetched, counted in
            # coverage) rather than queried one by one.
            if len(refs) > UNRESOLVED_GROUP:
                step = -(-len(refs) // 8)
                out = []
                for i in range(0, len(refs), step):
                    out += fetch(otype, refs[i:i + step])
                return out
            return [(otype, i, v, 499, None, None, None, None, None) for i, v in refs]
        return []                             # transient; retried on the next run

    run_pool(batches(), lambda b: fetch(*b), lambda rows: conn.executemany(
        "INSERT OR REPLACE INTO versions VALUES (?,?,?,?,?,?,?,?,?)", rows),
        label="version batches", workers=args.workers, total=calls)
    print(dict(conn.execute("SELECT status, COUNT(*) FROM versions GROUP BY 1")))
    return 0


# ------------------------------------------------------------------------- status


def cmd_status(args) -> int:
    def q(rel, sql):
        if not data_path(*rel.split("/")).exists():
            return "absent"
        try:
            return [list(r) for r in read_only(rel).execute(sql)]
        except sqlite3.Error as exc:
            return str(exc)

    print(json.dumps({
        "selection": q(CORPUS_MN_DB, "SELECT corpus, label, COUNT(*) FROM selection GROUP BY 1,2"),
        "dump_mining": q(DUMP_DB, "SELECT COUNT(*), MAX(last_cid) FROM chunks"),
        "dump_reverts": q(DUMP_DB, "SELECT substr(created_at,1,4), COUNT(*) FROM reverts GROUP BY 1"),
        "mn_meta": q(CORPUS_MN_DB, "SELECT status, COUNT(*) FROM meta GROUP BY 1"),
        "mn_contents": q(CORPUS_MN_DB, "SELECT status, COUNT(*), SUM(bytes) FROM contents GROUP BY 1"),
        "users": q(USERS_DB, "SELECT status, COUNT(*) FROM users GROUP BY 1"),
        "histories": q(HISTORY_DB, "SELECT status, truncated, COUNT(*) FROM histories GROUP BY 1,2"),
        "versions_planned": q(VERSIONS_DB, "SELECT corpus, COUNT(*), SUM(n_wanted), SUM(capped) FROM planned GROUP BY 1"),
        "versions_wanted": q(VERSIONS_DB, "SELECT otype, COUNT(*) FROM wanted GROUP BY 1"),
        "versions_fetched": q(VERSIONS_DB, "SELECT otype, status, COUNT(*) FROM versions GROUP BY 1,2"),
    }, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--corpora", default="MN", help="MN, R or MN,R")
    s.add_argument("--since", default="2023-01-01")
    s.add_argument("--until", default="2026-09-17")
    s.add_argument("--force", action="store_true")
    for name, rate in (("changesets", 2.0), ("users", 1.0), ("histories", 1.5), ("versions", 1.0)):
        p = sub.add_parser(name)
        p.add_argument("--rate", type=float, default=rate)
        p.add_argument("--workers", type=int, default=3)
        if name == "versions":
            p.add_argument("--plan-only", action="store_true")
    sub.add_parser("status")
    args = ap.parse_args()
    return {"select": cmd_select, "changesets": cmd_changesets, "users": cmd_users,
            "histories": cmd_histories, "versions": cmd_versions,
            "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
