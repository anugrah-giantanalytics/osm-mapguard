"""Paths, endpoints and the one User-Agent every request in this project must carry.

Two rules this module exists to enforce, both from PROJECT_PLAN.md section 9:

  1. Bulk data never lives in the repository. It goes wherever MAPGUARD_DATA points,
     default ``C:\\osm-data``. The laptop has 0.77 GB of free RAM and the repo used to
     sit inside OneDrive; both mistakes are cheap to make again and expensive to undo.
  2. Every OSM request identifies itself. The API is donated infrastructure, and an
     anonymous scraper hammering it is how a project gets blocked.

Nothing here reads a file or hits the network at import time, so it is safe to import
from a notebook cell, a test, or a script.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- repository layout (code, docs, small tracked artefacts) -----------------

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "data" / "labels" / "ovid_labels.tsv"
RESULTS = REPO / "results"

# --- bulk data (never in the repository, never in OneDrive) ------------------


def data_root() -> Path:
    """Where bulk data lives. Override with MAPGUARD_DATA.

    Read through a function rather than a module constant so a test can point it at a
    tmp_path by setting the environment variable, without reloading the module.
    """
    root = Path(os.environ.get("MAPGUARD_DATA", r"C:\osm-data"))
    return root


def data_path(*parts: str) -> Path:
    """A path under the bulk data root, with parent directories created."""
    p = data_root().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir() -> Path:
    d = data_root() / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def interim_dir() -> Path:
    d = data_root() / "interim"
    d.mkdir(parents=True, exist_ok=True)
    return d


CORPUS_A_DB = "raw/corpus_a.sqlite"
CORPUS_A_META_DB = "raw/corpus_a_meta.sqlite"
CORPUS_A_GEOM_DB = "raw/corpus_a_geom.sqlite"

# --- endpoints ---------------------------------------------------------------

OSM_API = "https://api.openstreetmap.org/api/0.6"
OVERPASS = "https://overpass-api.de/api/interpreter"
PLANET = "https://planet.openstreetmap.org"
REPLICATION_CHANGESETS = f"{PLANET}/replication/changesets"
REPLICATION_MINUTE = f"{PLANET}/replication/minute"
REAL_CHANGESETS = "https://real-changesets.s3.us-west-2.amazonaws.com"

# The OSM multi-fetch endpoints cap at 725 elements per request. Verified 2026-09-04.
MULTIFETCH_MAX = 725

# Gate 1b measured public Overpass at 3.17 queries/minute, projecting to 96 hours for the
# full corpus. See docs/planning/GATE1B_DECISION.md. Geometry is therefore enriched on a fixed
# subset chosen by a hash of the changeset id - independent of size, author and content,
# because sampling by any of those would correlate "has geometry" with "is damaging".
GEOMETRY_SUBSET_PERMILLE = 328          # ~32.8%, about 6,000 of 18,276


def in_geometry_subset(cid: int) -> bool:
    """Deterministic, reproducible from the id alone, no stored list needed."""
    import hashlib

    digest = hashlib.sha256(str(int(cid)).encode()).hexdigest()
    return int(digest[:8], 16) % 1000 < GEOMETRY_SUBSET_PERMILLE

# --- identification ----------------------------------------------------------

CONTACT = os.environ.get("MAPGUARD_CONTACT", "ayushmgarg10@gmail.com")
USER_AGENT = (
    f"MapGuard-research/0.2 (academic OSM vandalism-detection study; {CONTACT})"
)
HEADERS = {"User-Agent": USER_AGENT}

# --- politeness defaults -----------------------------------------------------

# Measured 2026-09-04: the changeset download endpoint answers in ~1.2 s, with an
# 8 s tail. PROJECT_PLAN v1 claimed 0.2 s; every budget built on that number was wrong.
OSM_MEDIAN_SECONDS = 1.2

DEFAULT_RATE = 4.0        # global requests/second across all workers
DEFAULT_WORKERS = 4
OVERPASS_RATE = 0.5       # the public instance runs two slots; do not crowd them
TIMEOUT = 60
