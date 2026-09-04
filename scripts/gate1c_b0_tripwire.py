"""Gate 1c - the label-leakage tripwire.

Baseline B0 sees three things and nothing else: changeset comment length, the editor
string, and account age at the time of the edit. None of those describe what was edited.
If that scores well, the labels are not measuring vandalism - they are measuring
something that leaked into the record, and every number downstream of it is void.

The obvious leak to catch: the Ovid labels are derived from revert history, so if any
revert-related text or tooling signature ended up inside the changeset record itself, a
model can read the answer straight off it.

Pass criterion (PROJECT_PLAN section 8.4): F1 below 0.70 under the user-disjoint split.

Run after scripts/fetch_corpus_a_meta.py has populated the metadata store.

    python scripts/gate1c_b0_tripwire.py
    python scripts/gate1c_b0_tripwire.py --no-user   # skip the account-age fetch
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mapguard.config import CORPUS_A_META_DB, DEFAULT_RATE, LABELS, RESULTS, data_path
from mapguard.data.http import PoliteSession
from mapguard.data.users import account_age_days, ensure_users, open_cache

USERS_DB = "raw/users.sqlite"


def load_labels() -> dict[int, int]:
    out = {}
    with LABELS.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            cid, label = line.rstrip("\n").split("\t")
            out[int(cid)] = 1 if label.strip().lower() == "true" else 0
    return out


def user_disjoint_split(uids: list[int], seed: int = 20260904, frac: float = 0.2):
    """Hold out whole users, not whole changesets.

    A random split puts the same vandal on both sides and the model learns the account
    rather than the edit. PROJECT_PLAN section 11 reports this split (R1b) alongside the
    leaky one (R1a); the tripwire only makes sense on the honest one.
    """
    rng = np.random.default_rng(seed)
    unique = sorted({u for u in uids if u is not None})
    rng.shuffle(unique)
    n_test = int(len(unique) * frac)
    test_users = set(unique[:n_test])
    return np.array([u in test_users for u in uids])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-user", action="store_true", help="skip the account-age feature")
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    labels = load_labels()
    meta_db = data_path(*CORPUS_A_META_DB.split("/"))
    if not meta_db.exists():
        print(f"no metadata store at {meta_db} - run scripts/fetch_corpus_a_meta.py first")
        return 2

    conn = sqlite3.connect(meta_db)
    rows = conn.execute(
        "SELECT cid, uid, created_at, comment, editor, num_changes FROM meta WHERE status = 200"
    ).fetchall()
    conn.close()
    rows = [r for r in rows if r[0] in labels]
    if len(rows) < 500:
        print(f"only {len(rows)} rows fetched so far - let the metadata fetch finish")
        return 2

    cids = [r[0] for r in rows]
    uids = [r[1] for r in rows]
    created = [r[2] for r in rows]
    comments = [r[3] or "" for r in rows]
    editors = [r[4] or "" for r in rows]
    y = np.array([labels[c] for c in cids])

    ages = np.zeros(len(rows))
    user_summary = None
    if not args.no_user:
        ucache = open_cache(data_path(*USERS_DB.split("/")))
        session = PoliteSession(DEFAULT_RATE)
        print(f"fetching account_created for {len(set(uids))} unique users...")
        user_summary = ensure_users(ucache, session, uids)
        created_map = dict(ucache.execute("SELECT uid, account_created FROM users"))
        ucache.close()
        ages = np.array(
            [account_age_days(created_map.get(u), t) or -1.0 for u, t in zip(uids, created)]
        )

    # --- B0 features. Three signals, none of them about what was edited. ---
    comment_len = np.array([len(c) for c in comments], dtype=float)
    numeric = np.column_stack([comment_len, ages])

    editor_vec = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z]+", max_features=50)
    E = editor_vec.fit_transform(editors).toarray()
    X = np.hstack([numeric, E])

    test_mask = user_disjoint_split(uids, seed=args.seed)
    Xtr, Xte, ytr, yte = X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)

    f1 = float(f1_score(yte, pred))
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gate": "1c",
        "question": "do the labels leak into comment length, editor string and account age?",
        "pass_criterion": "F1 below 0.70 on a user-disjoint split",
        "n_rows": len(rows),
        "n_train": int((~test_mask).sum()),
        "n_test": int(test_mask.sum()),
        "test_positive_rate": round(float(yte.mean()), 4),
        "features": ["comment_length", "account_age_days" if not args.no_user else "account_age_days (disabled)"]
        + [f"editor:{t}" for t in editor_vec.get_feature_names_out()[:10]],
        "f1": round(f1, 4),
        "roc_auc": round(float(roc_auc_score(yte, p)), 4),
        "pr_auc": round(float(average_precision_score(yte, p)), 4),
        "user_cache": user_summary,
        "verdict": "PASS" if f1 < 0.70 else "FAIL - labels may be leaking, investigate before modelling",
    }

    out = RESULTS / "gate1c_b0_tripwire.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n--- Gate 1c ---")
    print(json.dumps(summary, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
