"""[BASELINES · R1/R2] B0 leakage tripwire, logistic regression and XGBoost.

Run order 1 of 6. Plan stage: baselines. CPU only - no TensorFlow needed.

    python train/01_baselines/baselines_train.py
    python train/01_baselines/baselines_train.py --set task=T1
    python train/01_baselines/baselines_train.py --set split=R1a --set tag=ovid_protocol

Three models, three jobs:

  B0        comment length + editor string + time since the author's first earlier
            changeset (point-in-time), and nothing about what was
            edited. The tripwire: if this separates the classes well, the labels are
            readable from metadata and every later number is suspect. Gate 1c measured
            AUC 0.656 on Corpus A; that is a weak prior, not a leak.
  LogReg    every tabular feature - the floor a neural model must clear to justify itself
  XGBoost   the strongest non-neural reference; if Stage 2 cannot beat it, say so
  Hybrid    XGBoost on the tabular features plus two out-of-fold text models: a bag of edit
            tokens (what each edit did - see mapguard/features/edit_tokens.py) and character
            n-grams of the comment and tag text. Trained with author-balanced weights. This is
            the "gradient-boosted trees over engineered + learned signals" design that large
            wiki and map moderation systems use; round 1 showed a bare deep model only ties
            plain XGBoost, and this is what it has to beat instead.

Running this first also creates the frozen split, so Stage 2 trains on exactly these rows.
"""

import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN.parent / "src")]

import numpy as np  # noqa: E402

from common import scoring, stage  # noqa: E402
from common.preprocess import Normaliser  # noqa: E402


def b0_features(store, rows_all):
    """Metadata only. Columns that are absent are recorded, never silently zero-filled."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    cs = store.changesets
    used, missing = [], []
    blocks = []
    # account_age_days was replaced by the point-in-time user_days_active_span: account age is
    # read from today's account, and deleted accounts (mostly vandals) made its missingness a
    # leak. See the note above TABULAR_FEATURES in mapguard/training/dataset.py.
    for col in ("comment_length", "user_days_active_span"):
        if col in cs.columns:
            v = cs[col].to_numpy(dtype="float64", na_value=np.nan)
            blocks.append(np.nan_to_num(np.sign(v) * np.log1p(np.abs(v)), nan=-1.0)[:, None])
            used.append(col)
        else:
            missing.append(col)
    vec = None
    if "editor" in cs.columns:
        vec = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z]+", max_features=50)
        editors = cs["editor"].fillna("").astype(str).to_numpy()
        vec.fit(editors[rows_all])
        blocks.append(vec.transform(editors).toarray())
        used.append("editor (tf-idf, 50 tokens)")
    else:
        missing.append("editor")
    if not blocks:
        return None, used, missing, None
    return np.hstack(blocks), used, missing, vec


def fit_predict_logreg(X, y, tr, va, te, seed):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)
    clf.fit(X[tr], y[tr])
    return clf, clf.predict_proba(X[va])[:, 1], clf.predict_proba(X[te])[:, 1]


def author_weights(uids, rows) -> np.ndarray:
    """1/sqrt(changesets by the same author among `rows`): a bulk-reverted account with 200
    positives counts ~14x one changeset, not 200x. None-author rows get weight 1."""
    from collections import Counter

    counts = Counter(u for u, r in zip(uids, rows) if r and u is not None)
    return np.array([1.0 / np.sqrt(counts.get(u, 1)) if u is not None else 1.0 for u in uids])


def fit_hybrid(store, prep, tr, va, te, y, seed, params, weighting=True):
    """Out-of-fold stacking: text models are fitted on 4/5 of the training authors and score
    the held-out fifth, so the tree never sees a text score that was fitted on its own row."""
    import xgboost as xgb
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    from mapguard.training.dataset import text_input

    cs = store.changesets
    if "diff_tokens" not in cs.columns:
        return None
    tokens = cs["diff_tokens"].fillna("").astype(str).tolist()
    texts = [text_input(c, b) for c, b in zip(cs.get("comment", [""] * len(cs)), cs["text_blob"])]
    tri, vai, tei = (np.flatnonzero(m) for m in (tr, va, te))
    sw = author_weights(prep.uids, tr) if weighting else np.ones(len(y))

    def views():
        tv = TfidfVectorizer(token_pattern=r"\S+", min_df=3, sublinear_tf=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                             max_features=200_000, sublinear_tf=True)
        tv.fit([tokens[i] for i in tri])
        cv.fit([texts[i] for i in tri])
        return tv.transform(tokens), cv.transform(texts), tv

    Xk, Xc, tv = views()
    oof = np.zeros((len(y), 2))
    groups = np.array([-1 if u is None else u for u in prep.uids])[tri]
    for a, b in GroupKFold(5).split(tri, groups=groups):
        A, B = tri[a], tri[b]
        for j, (X, C) in enumerate(((Xk, 1.0), (Xc, 2.0))):
            m = LogisticRegression(C=C, max_iter=4000).fit(X[A], y[A], sample_weight=sw[A])
            oof[B, j] = m.predict_proba(X[B])[:, 1]
    lk = LogisticRegression(C=1.0, max_iter=4000).fit(Xk[tri], y[tri], sample_weight=sw[tri])
    lc = LogisticRegression(C=2.0, max_iter=4000).fit(Xc[tri], y[tri], sample_weight=sw[tri])
    rest = np.concatenate([vai, tei])
    oof[rest, 0] = lk.predict_proba(Xk[rest])[:, 1]
    oof[rest, 1] = lc.predict_proba(Xc[rest])[:, 1]

    raw = np.nan_to_num(prep.arrays["tabular"], nan=0.0, posinf=0.0, neginf=0.0)
    X2 = np.column_stack([raw, oof])
    clf = xgb.XGBClassifier(**params, eval_metric="aucpr", random_state=seed, n_jobs=-1)
    clf.fit(X2[tri], y[tri], sample_weight=sw[tri], eval_set=[(X2[vai], y[vai])], verbose=False)
    coef = sorted(zip(tv.get_feature_names_out(), lk.coef_[0]), key=lambda kv: kv[1])
    info = {"author_weighting": "1/sqrt(n)" if weighting else None,
            "most_damaging_tokens": [k for k, _ in coef[-20:][::-1]],
            "most_benign_tokens": [k for k, _ in coef[:20]],
            "stack_importance": {"tokens_score": round(float(clf.feature_importances_[-2]), 4),
                                 "text_score": round(float(clf.feature_importances_[-1]), 4)}}
    return clf.predict_proba(X2[vai])[:, 1], clf.predict_proba(X2[tei])[:, 1], info


def main() -> int:
    args = stage.parse_args(__doc__.splitlines()[0], Path(__file__).with_name("config.json"))
    cfg, run, store, prep, masks = stage.setup(args)
    b = cfg.get("baselines", {})
    seed = int(cfg.get("seed", 0))
    tr, va, te = masks["train"], masks["val"], masks["test"]
    y = prep.y.astype(int)
    results = {"models": {}, "split_leaky": cfg["split"] == "R1a",
               "data_missing_optional_columns": store.missing_optional}

    # --- B0 tripwire -------------------------------------------------------------
    if b.get("b0_tripwire", True):
        X0, used, missing, vec = b0_features(store, tr)
        if X0 is None:
            results["models"]["b0_tripwire"] = {"skipped": "no metadata columns available",
                                                "missing": missing}
        else:
            clf, pv, pt = fit_predict_logreg(X0, y, tr, va, te, seed)
            rep = scoring.report(y[va], pv, y[te], pt, seed=seed, test_groups=prep.uids[te])
            auc = rep["test_intervals_95"]["roc_auc"][0]
            leak_at = float(b.get("b0_leak_auc", 0.85))
            results["models"]["b0_tripwire"] = {
                "features_used": used, "features_missing": missing,
                "headline": scoring.headline(rep), "evaluation": rep,
                "verdict": ("FAIL - labels readable from metadata alone; investigate before "
                            "trusting any other result" if auc >= leak_at else
                            "PASS - weak prior, not leakage"),
            }
            print(f"B0 tripwire  ROC-AUC {auc}  -> {results['models']['b0_tripwire']['verdict']}")

    # --- tabular features, normalised on training rows ---------------------------
    norm = Normaliser().fit(prep, tr, groups=("tabular",))
    Xt = norm._z(np.sign(prep.arrays["tabular"]) * np.log1p(np.abs(prep.arrays["tabular"])),
                 "tabular", prep.tabular_columns)
    norm.save(run.path("normaliser_tabular.json"))

    if b.get("logistic_regression", True):
        clf, pv, pt = fit_predict_logreg(Xt, y, tr, va, te, seed)
        rep = scoring.report(y[va], pv, y[te], pt, seed=seed, test_groups=prep.uids[te])
        coefs = sorted(zip(prep.tabular_columns, clf.coef_[0]), key=lambda kv: -abs(kv[1]))[:15]
        results["models"]["logistic_regression"] = {
            "n_features": int(Xt.shape[1]), "headline": scoring.headline(rep), "evaluation": rep,
            "top_coefficients": {k: round(float(v), 4) for k, v in coefs},
        }
        np.savez_compressed(run.path("predictions_logreg.npz"), test_cids=prep.cids[te],
                            test_y=y[te], test_p=pt, val_cids=prep.cids[va], val_y=y[va], val_p=pv)
        print(f"LogReg       PR-AUC {rep['test_intervals_95']['pr_auc']}")

    if b.get("xgboost", True):
        try:
            import xgboost as xgb
        except ImportError:
            results["models"]["xgboost"] = {"skipped": "xgboost not installed"}
        else:
            raw = np.nan_to_num(prep.arrays["tabular"], nan=0.0, posinf=0.0, neginf=0.0)
            clf = xgb.XGBClassifier(**b.get("xgboost_params", {}), eval_metric="aucpr",
                                    random_state=seed, n_jobs=-1)
            clf.fit(raw[tr], y[tr], eval_set=[(raw[va], y[va])], verbose=False)
            pv, pt = clf.predict_proba(raw[va])[:, 1], clf.predict_proba(raw[te])[:, 1]
            rep = scoring.report(y[va], pv, y[te], pt, seed=seed, test_groups=prep.uids[te])
            imp = sorted(zip(prep.tabular_columns, clf.feature_importances_), key=lambda kv: -kv[1])[:15]
            results["models"]["xgboost"] = {
                "params": b.get("xgboost_params", {}), "headline": scoring.headline(rep),
                "evaluation": rep, "top_importances": {k: round(float(v), 4) for k, v in imp},
            }
            np.savez_compressed(run.path("predictions_xgboost.npz"), test_cids=prep.cids[te],
                                test_y=y[te], test_p=pt, val_cids=prep.cids[va], val_y=y[va], val_p=pv)
            print(f"XGBoost      PR-AUC {rep['test_intervals_95']['pr_auc']}")

    if b.get("hybrid", True):
        try:
            got = fit_hybrid(store, prep, tr, va, te, y, seed, b.get("xgboost_params", {}),
                             weighting=b.get("author_weighting", "sqrt") == "sqrt")
        except ImportError as exc:
            got, results["models"]["hybrid"] = None, {"skipped": repr(exc)}
        if got is None:
            results["models"].setdefault("hybrid", {"skipped": "store has no diff_tokens (schema < 4)"})
        else:
            pv, pt, info = got
            rep = scoring.report(y[va], pv, y[te], pt, seed=seed, test_groups=prep.uids[te])
            results["models"]["hybrid"] = {**info, "headline": scoring.headline(rep), "evaluation": rep}
            np.savez_compressed(run.path("predictions_hybrid.npz"), test_cids=prep.cids[te],
                                test_y=y[te], test_p=pt, val_cids=prep.cids[va], val_y=y[va], val_p=pv)
            print(f"Hybrid       PR-AUC {rep['test_intervals_95']['pr_auc']}  "
                  f"one-per-author {rep['author_robust'].get('one_per_author_pr_auc')}")

    best = max(((k, v["headline"]["test_pr_auc"][0]) for k, v in results["models"].items()
                if "headline" in v and k != "b0_tripwire"), key=lambda kv: kv[1], default=None)
    results["headline"] = {"best_baseline": best[0], "best_test_pr_auc": best[1]} if best else None
    run.finish(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
