"""[STACK · v3 final model candidate] Deep model as an out-of-fold feature inside the tree model.

Run order 9 - after 01 and 02 (it trains its own deep models; it needs no earlier run).

    python train/09_stack/stack_train.py --set task=ovid
    python train/09_stack/stack_train.py --set task=T1 --set stack.folds=5

Why. The stage-07 blend picks one weight on validation; on small validation sets it overfits
(T1, rounds 2-2c: the blend lost to the hybrid). Stacking lets the tree learn *when* to trust
the deep score, per changeset, from the training rows - the standard way production systems
combine a neural encoder with gradient-boosted trees.

Leakage control. A deep score fitted on a row cannot be a feature for that row. So:
  1. the training rows are split into K folds by author (GroupKFold on uid);
  2. a deep model is trained on K-1 folds and scores the held-out fold (out-of-fold);
  3. validation and test rows get the mean score of the K fold models (none saw them);
  4. the hybrid tree is trained with that score as one more column.
Every fold model also early-stops on the validation rows only, never on test.

Writes a run dir `stack` with predictions_stack_<variant>.npz, one results entry per hybrid
variant (with and without the deep column, same split), and per-fold deep runs under folds/.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN / "01_baselines"), str(TRAIN.parent / "src")]

import numpy as np  # noqa: E402

import baselines_train as bt  # noqa: E402
from common import config as config_mod  # noqa: E402
from common import preprocess, scoring, stage  # noqa: E402
from common.fit import train_stage2  # noqa: E402
from common.run import Run  # noqa: E402

DEEP_CONFIG = TRAIN / "02_stage2_supervised" / "config.json"
BASE_CONFIG = TRAIN / "01_baselines" / "config.json"


def main() -> int:
    args = stage.parse_args(__doc__.splitlines()[0], DEEP_CONFIG)
    # own stage name: the index must never offer a stack run as "the newest stage-2 run"
    args.set = list(args.set or []) + ["stage=stack"]
    cfg, run, store, prep, masks = stage.setup(args, tag="stack")
    s_cfg = cfg.get("stack", {})
    k = int(s_cfg.get("folds", 5))
    tr, va, te = masks["train"], masks["val"], masks["test"]
    y = prep.y.astype(int)
    seed = int(cfg.get("seed", 0))

    # ---------------------------------------------------------------- deep OOF scores
    from sklearn.model_selection import GroupKFold

    tri = np.flatnonzero(tr)
    groups = np.array([-1 if u is None else u for u in prep.uids])[tri]
    deep = np.full(len(y), np.nan)
    val_sum, test_sum = np.zeros(int(va.sum())), np.zeros(int(te.sum()))
    folds = []
    for f, (a, b) in enumerate(GroupKFold(k).split(tri, groups=groups)):
        fold_train = np.zeros(len(y), bool)
        fold_train[tri[a]] = True
        held = np.zeros(len(y), bool)
        held[tri[b]] = True
        import tensorflow as tf

        tf.keras.backend.clear_session()         # Keras 2 keeps every fold's graph otherwise
        fcfg = copy.deepcopy(cfg)
        fcfg["stage"] = "stack_fold"
        fcfg.setdefault("evaluate", {})["attribution"] = False   # test attribution: not needed per fold
        frun = Run(fcfg, tag=f"fold{f}", root=run.path("folds"))
        res = train_stage2(fcfg, prep, masks, frun, train_rows=fold_train, extra_rows=held)
        frun.finish({**res, "stage": "stack_fold", "task": cfg.get("task"), "split": cfg["split"],
                     "seed": seed, "fold": f})
        z = np.load(frun.path("predictions.npz"))
        pos = {int(c): i for i, c in enumerate(prep.cids)}
        deep[[pos[int(c)] for c in z["extra_cids"]]] = z["extra_p"]
        val_sum += z["val_p"]
        test_sum += z["test_p"]
        folds.append({"fold": f, "train_rows": int(fold_train.sum()), "held_out": int(held.sum()),
                      "test_pr_auc": res["headline"]["test_pr_auc"]})
        print(f"fold {f}: trained on {fold_train.sum()}, scored {held.sum()} held-out rows", flush=True)
    deep[np.flatnonzero(va)] = val_sum / k
    deep[np.flatnonzero(te)] = test_sum / k
    if np.isnan(deep[tr | va | te]).any():
        raise SystemExit("stacking: some rows have no deep score - fold bookkeeping is wrong")

    # ---------------------------------------------------------------- trees with / without deep
    import json

    overrides = [f"task={cfg.get('task')}"]
    excl = cfg.get("model", {}).get("exclude_columns")
    if excl:               # e.g. ovid_modern: adiff-only columns removed, exactly as in stage 01
        overrides.append(f"model.exclude_columns={json.dumps(excl)}")
    base_cfg = config_mod.load(str(BASE_CONFIG), overrides)
    bprep = preprocess.prepare(store, base_cfg)
    if not np.array_equal(bprep.cids, prep.cids):
        raise SystemExit("stacking: baseline and deep row orders differ")
    params = base_cfg.get("baselines", {}).get("xgboost_params", {})
    results = {"folds": folds, "deep_oof_rows": int(tr.sum()), "models": {}}
    for variant in s_cfg.get("hybrid_variants", ["base", "ctx_emb"]):
        kw = bt.hybrid_inputs(store, bprep, variant)
        if kw is None:
            results["models"][variant] = {"skipped": "side table missing"}
            continue
        for with_deep in (False, True):
            key = f"stack_{variant}" if with_deep else f"hybrid_{variant}_same_split"
            extra = {"deep_oof": deep} if with_deep else None
            pv, pt, info = bt.fit_hybrid(store, bprep, tr, va, te, y, seed, params, weighting=True,
                                         extra_stack=extra, **kw)
            rep = scoring.report(y[va], pv, y[te], pt, seed=seed, test_groups=prep.uids[te])
            results["models"][key] = {**info, "variant": variant, "deep_column": with_deep,
                                      "headline": scoring.headline(rep), "evaluation": rep}
            np.savez_compressed(run.path(f"predictions_{key}.npz"), test_cids=prep.cids[te],
                                test_y=y[te], test_p=pt, val_cids=prep.cids[va], val_y=y[va], val_p=pv)
            print(f"{key:30s} PR-AUC {rep['test_intervals_95']['pr_auc']}  one-per-author "
                  f"{rep['author_robust'].get('one_per_author_pr_auc')}  F1 {rep['ovid_table']['f1']}",
                  flush=True)
    np.savez_compressed(run.path("deep_oof.npz"), cids=prep.cids, deep=deep,
                        role=np.where(tr, 0, np.where(va, 1, np.where(te, 2, -1))))
    best = max(((kk, v["headline"]["test_pr_auc"][0]) for kk, v in results["models"].items()
                if "headline" in v), key=lambda kv: kv[1], default=None)
    # informational only - nothing is selected by test score
    results["headline"] = ({"highest_test_pr_auc_model": best[0], "test_pr_auc": best[1]}
                           if best else None)
    run.finish(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
