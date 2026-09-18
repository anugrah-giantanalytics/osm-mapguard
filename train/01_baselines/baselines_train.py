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
            rep = scoring.report(y[va], pv, y[te], pt, seed=seed)
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
        rep = scoring.report(y[va], pv, y[te], pt, seed=seed)
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
            rep = scoring.report(y[va], pv, y[te], pt, seed=seed)
            imp = sorted(zip(prep.tabular_columns, clf.feature_importances_), key=lambda kv: -kv[1])[:15]
            results["models"]["xgboost"] = {
                "params": b.get("xgboost_params", {}), "headline": scoring.headline(rep),
                "evaluation": rep, "top_importances": {k: round(float(v), 4) for k, v in imp},
            }
            np.savez_compressed(run.path("predictions_xgboost.npz"), test_cids=prep.cids[te],
                                test_y=y[te], test_p=pt, val_cids=prep.cids[va], val_y=y[va], val_p=pv)
            print(f"XGBoost      PR-AUC {rep['test_intervals_95']['pr_auc']}")

    best = max(((k, v["headline"]["test_pr_auc"][0]) for k, v in results["models"].items()
                if "headline" in v and k != "b0_tripwire"), key=lambda kv: kv[1], default=None)
    results["headline"] = {"best_baseline": best[0], "best_test_pr_auc": best[1]} if best else None
    run.finish(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
