"""Round-2 deep dive: reads raw/ only, writes analysis/deep_dive.json. Never edits raw/."""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\AyushGarg\projects\personal\osm_mapguard")
sys.path[:0] = [str(REPO / "train"), str(REPO / "src")]
from common import scoring  # noqa: E402
from mapguard.evaluation.metrics import average_precision  # noqa: E402

ROUND = REPO / "experiments" / "rounds" / (sys.argv[1] if len(sys.argv) > 1 else "r2_2026-10-07_altair_h100")
RUNS = ROUND / "raw" / "experiments" / "runs"
TASKS = ["ovid", "ovid_modern", "T1", "T2", "revert_vandalism", "revert"]
B = 2000
rng = np.random.default_rng(0)

uid_of = dict(pd.read_parquet("C:/osm-data/interim/changesets.parquet", columns=["cid", "uid"])
              .astype({"cid": int}).itertuples(index=False, name=None))


def one(stage, task, extra=""):
    ds = [d for d in glob.glob(str(RUNS / stage / f"*_{task}_R1b{extra}"))
          if not d.endswith("_quick")]
    assert len(ds) == 1, (stage, task, ds)
    return Path(ds[0])


def load(path):
    z = np.load(path)
    return {f: dict(zip(z[f"{f}_cids"].astype(int), z[f"{f}_p"].astype(float))) for f in ("val", "test")}, \
        dict(zip(z["test_cids"].astype(int), z["test_y"].astype(int)))


out = {}
for task in TASKS:
    deep_dir, base_dir, ens_dir = one("stage2_supervised", task), one("baselines", task), one("ensemble", task)
    d, y = load(deep_dir / "predictions.npz")
    h, _ = load(base_dir / "predictions_hybrid.npz")
    x, _ = load(base_dir / "predictions_xgboost.npz")
    w = json.load(open(ens_dir / "results.json"))["deep_weight"]
    ids = sorted(d["test"])
    assert set(ids) == set(h["test"]) == set(x["test"])
    yt = np.array([y[c] for c in ids])
    P = {"deep": np.array([d["test"][c] for c in ids]),
         "hybrid": np.array([h["test"][c] for c in ids]),
         "xgboost": np.array([x["test"][c] for c in ids])}
    P["ensemble"] = w * P["deep"] + (1 - w) * P["hybrid"]
    groups = [uid_of.get(c) for c in ids]

    point = {k: float(average_precision(yt, p)) for k, p in P.items()}
    robust = {k: scoring.author_robust(yt, p, groups).get("one_per_author_pr_auc") for k, p in P.items()}

    # paired bootstrap over authors (cluster bootstrap): resample authors, keep all their rows
    g = np.array([str(u) for u in groups])
    uniq, inv = np.unique(g, return_inverse=True)
    rows_of = [np.where(inv == i)[0] for i in range(len(uniq))]
    diffs = {"deep-hybrid": [], "ensemble-hybrid": [], "hybrid-xgboost": []}
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[i] for i in pick])
        yy = yt[idx]
        if yy.min() == yy.max():
            continue
        ap = {k: average_precision(yy, P[k][idx]) for k in ("deep", "hybrid", "ensemble", "xgboost")}
        diffs["deep-hybrid"].append(ap["deep"] - ap["hybrid"])
        diffs["ensemble-hybrid"].append(ap["ensemble"] - ap["hybrid"])
        diffs["hybrid-xgboost"].append(ap["hybrid"] - ap["xgboost"])
    ci = {k: [round(float(np.mean(v)), 4), round(float(np.percentile(v, 2.5)), 4),
              round(float(np.percentile(v, 97.5)), 4), round(float(np.mean(np.array(v) > 0)), 3)]
          for k, v in diffs.items()}

    # do deep and hybrid make the same mistakes?
    rank = lambda a: pd.Series(a).rank().to_numpy()
    spearman = float(np.corrcoef(rank(P["deep"]), rank(P["hybrid"]))[0, 1])
    err_d = (P["deep"] >= 0.5) != yt
    err_h = (P["hybrid"] >= 0.5) != yt
    out[task] = {
        "n_test": int(len(yt)), "positives": int(yt.sum()), "authors": int(len(uniq)),
        "deep_weight": w, "pr_auc": {k: round(v, 4) for k, v in point.items()},
        "one_per_author_pr_auc": {k: (round(v, 4) if v is not None else None) for k, v in robust.items()},
        "paired_author_bootstrap": {"B": B, "fields": ["mean", "lo95", "hi95", "P(diff>0)"], **ci},
        "spearman_deep_vs_hybrid": round(spearman, 3),
        "errors_at_0.5": {"deep": int(err_d.sum()), "hybrid": int(err_h.sum()),
                          "both": int((err_d & err_h).sum()),
                          "only_deep": int((err_d & ~err_h).sum()),
                          "only_hybrid": int((~err_d & err_h).sum())},
    }
    print(task, json.dumps(out[task]["pr_auc"]), json.dumps(out[task]["one_per_author_pr_auc"]))
    print("   ", json.dumps(ci), "spearman", out[task]["spearman_deep_vs_hybrid"], out[task]["errors_at_0.5"])

dest = ROUND / "analysis" / "deep_dive.json"
dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote", dest)
