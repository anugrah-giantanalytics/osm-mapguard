"""Probe: does WHERE (distance from the author's usual area) improve the hybrid?

Same split, same features, same hybrid code (fit_hybrid) - the only change is extra tabular
columns from C:/osm-data/where/where_features.parquet. Prints PR-AUC, one-per-author, F1/Acc
at the validation threshold, and an author-cluster bootstrap of the difference.
"""
import json
import sys
from pathlib import Path

REPO = Path(r"C:\AyushGarg\projects\personal\osm_mapguard")
sys.path[:0] = [str(REPO / "train"), str(REPO / "train" / "01_baselines"), str(REPO / "src")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import scoring, stage  # noqa: E402
import baselines_train as bt  # noqa: E402
from mapguard.evaluation.metrics import average_precision  # noqa: E402

task = sys.argv[1] if len(sys.argv) > 1 else "ovid"
sys.argv = ["probe", "--set", f"task={task}", "--set", "tag=where_probe"]
args = stage.parse_args("probe", REPO / "train" / "01_baselines" / "config.json")
cfg, run, store, prep, masks = stage.setup(args)
tr, va, te = masks["train"], masks["val"], masks["test"]
y = prep.y.astype(int)
seed = int(cfg.get("seed", 0))
params = cfg.get("baselines", {}).get("xgboost_params", {})

wf = pd.read_parquet("C:/osm-data/where/where_features.parquet").set_index("cid")
cols = [c for c in wf.columns if c.startswith("where_")]
W = wf.reindex(prep.cids.astype(int))[cols].to_numpy(dtype="float64")
for j, c in enumerate(cols):
    if c.endswith("_km"):
        W[:, j] = np.log1p(W[:, j])
print("WHERE columns:", cols, "| rows with distance:", float(np.isfinite(W[:, cols.index("where_dist_home_km")]).mean()))

base_tab = prep.arrays["tabular"].copy()


def run_variant(extra):
    prep.arrays["tabular"] = base_tab if extra is None else np.column_stack([base_tab, extra])
    pv, pt, info = bt.fit_hybrid(store, prep, tr, va, te, y, seed, params, weighting=True)
    rep = scoring.report(y[va], pv, y[te], pt, seed=seed, test_groups=prep.uids[te])
    return pt, rep


out = {}
P = {}
for name, extra in (("hybrid", None), ("hybrid+where", np.nan_to_num(W, nan=-1.0))):
    pt, rep = run_variant(extra)
    P[name] = pt
    h = scoring.headline(rep)
    out[name] = {"pr_auc": h["test_pr_auc"], "one_per_author": h["test_author_robust"]["one_per_author_pr_auc"],
                 "ovid_table": h["test_ovid_table"]}
    print(name, json.dumps(out[name]))
prep.arrays["tabular"] = base_tab

yt = y[te]
g = np.array([str(u) for u in prep.uids[te]])
uniq, inv = np.unique(g, return_inverse=True)
rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
rng = np.random.default_rng(0)
d = []
for _ in range(2000):
    idx = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
    if yt[idx].min() == yt[idx].max():
        continue
    d.append(average_precision(yt[idx], P["hybrid+where"][idx]) - average_precision(yt[idx], P["hybrid"][idx]))
out["where_minus_hybrid"] = [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4),
                             round(float(np.percentile(d, 97.5)), 4), round(float(np.mean(np.array(d) > 0)), 3)]
print("WHERE - hybrid (author bootstrap) mean, lo95, hi95, P>0:", out["where_minus_hybrid"])
dest = Path("C:/osm-data/where") / f"probe_{task}.json"
dest.write_text(json.dumps(out, indent=1))
print("wrote", dest)
