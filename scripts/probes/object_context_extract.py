"""Probe A1: object context - what each modified/deleted object WAS before the edit.

    python scripts/probes/object_context_extract.py A        # Corpus A only
    python scripts/probes/object_context_extract.py A,MN

Same iteration, hidden-version drop and old-tag sources (adiff + previous versions) as the store
builder, so the context covers exactly the edits the store has. Read-only on raw stores.
Writes C:/osm-data/probes/object_context_<corpora>.parquet (cid, ctx_tokens, numeric features).
Not run yet (2026-10-10).
"""
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import pandas as pd  # noqa: E402

from mapguard.features import build_store as bs  # noqa: E402
from mapguard.features.edit_tokens import LOWCARD_KEYS, MAX_VALUE_CHARS  # noqa: E402

PROMINENT = ("wikidata", "wikipedia", "name", "historic", "tourism", "heritage", "website",
             "brand", "opening_hours", "addr")
corpora = sys.argv[1].split(",") if len(sys.argv) > 1 else ["A", "MN"]
OUT = Path("C:/osm-data/probes")
OUT.mkdir(parents=True, exist_ok=True)

vconn = bs.open_ro(bs.VERSIONS_DB)
rows, t0, seen = [], time.time(), 0


def sources():
    if "A" in corpora:
        yield from bs.iter_corpus_a(bs.load_labels(), None)
    if "MN" in corpora:
        yield from bs.iter_corpus_mn(None)


for corpus, label, meta, xml, adiff_blob in sources():
    if not xml:
        continue
    edits, err = bs.parse_safe(xml)
    if err:
        continue
    edits, _ = bs.drop_hidden(edits, vconn)
    if not edits:
        continue
    adiff = bs.geometry_actions(adiff_blob)
    routes = {(a.otype, a.oid): 1 for a in adiff}
    vacts, _, _ = bs.version_actions(edits, vconn, set(routes), {})
    old_tags = {(a.otype, a.oid): (a.old.tags if a.old else {}) for a in adiff + vacts}

    toks, n_mod, n_known, prom = Counter(), 0, 0, Counter()
    for e in edits:
        if e.action == "create":
            continue
        n_mod += 1
        old = old_tags.get((e.otype, e.oid))
        if old is None:
            continue
        n_known += 1
        bases = {k.split(":")[0] for k in old}
        for b in bases:
            toks[f"@{b}"] += 1
            toks[f"{e.action[0]}@{b}"] += 1
        for k, v in old.items():
            b = k.split(":")[0]
            if b in LOWCARD_KEYS:
                toks[f"@{b}={str(v)[:MAX_VALUE_CHARS]}".replace(" ", "_")] += 1
        for p in PROMINENT:
            if p in bases:
                prom[p] += 1
        toks[f"@ntags_{min(len(old), 20) // 5}"] += 1
    bag = " ".join(t for t, c in toks.items() for _ in range(min(c, 20)))
    row = {"cid": meta["cid"], "ctx_tokens": bag, "ctx_n_modified": n_mod, "ctx_n_known": n_known,
           "ctx_known_frac": (n_known / n_mod) if n_mod else 0.0}
    for p in PROMINENT:
        row[f"ctx_frac_{p}"] = prom[p] / n_known if n_known else 0.0
        row[f"ctx_any_{p}"] = float(prom[p] > 0)
    rows.append(row)
    seen += 1
    if seen % 2000 == 0:
        print(f"{seen} changesets  {time.time() - t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
dest = OUT / f"object_context_{'_'.join(corpora)}.parquet"
df.to_parquet(dest, index=False)
print(f"wrote {dest}: {len(df)} rows in {time.time() - t0:.0f}s; "
      f"changesets with any known old object: {(df.ctx_n_known > 0).mean():.3f}")
