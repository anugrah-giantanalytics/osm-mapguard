"""Multi-seed analysis of a round (round 4 onwards). Reads <round>/raw only.

    python scripts/analyse_seeds.py experiments/rounds/<round>

For every task and model: mean ± sd over seeds (PR-AUC, one-per-author PR-AUC, F1/accuracy on
Ovid's protocol), then a paired author-cluster bootstrap of the *seed-averaged* scores of each
model against the hybrid - the test a claimed gain has to pass. Writes analysis/seeds.json.

Models: hybrid, hybrid_emb (stage 01), deep (02), ensemble (07, recomputed from its own pair and
weight), stack_base, stack_emb (09). Seeds are read from run names (`_s<seed>`) or, for the
ensemble, from the deep run it blended.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "train"), str(REPO / "src")]
from common import scoring  # noqa: E402
from mapguard.evaluation.metrics import average_precision  # noqa: E402

TASKS = ["ovid", "ovid_modern", "T1", "T2", "revert_vandalism", "revert"]
SEED = re.compile(r"_s(\d{8})")
B = 2000


def load(npz):
    z = np.load(npz)
    return ({int(c): float(p) for c, p in zip(z["test_cids"], z["test_p"])},
            {int(c): int(y) for c, y in zip(z["test_cids"], z["test_y"])},
            {int(c): float(p) for c, p in zip(z["val_cids"], z["val_p"])},
            {int(c): int(y) for c, y in zip(z["val_cids"], z["val_y"])})


def seed_of(path: Path) -> str | None:
    m = SEED.search(path.name)
    return m.group(1) if m else None


def main() -> int:
    rnd = Path(sys.argv[1]).resolve()
    runs = rnd / "raw" / "experiments" / "runs"
    import pandas as pd
    uid = dict(pd.read_parquet("C:/osm-data/interim/changesets.parquet", columns=["cid", "uid"])
               .astype({"cid": int}).itertuples(index=False, name=None))
    out = {"source": str(rnd), "tasks": {}}
    for t in TASKS:
        preds = defaultdict(dict)     # model -> seed -> (test dict, y dict, val dict, val y)
        for d in glob.glob(str(runs / "baselines" / f"*_{t}_R1b_s*")):
            d = Path(d)
            for name in ("hybrid", "hybrid_emb"):
                f = d / f"predictions_{name}.npz"
                if f.exists():
                    preds[name][seed_of(d)] = load(f)
        for d in glob.glob(str(runs / "stage2_supervised" / f"*_{t}_R1b_s*")):
            d = Path(d)
            if "ovid_repro" not in d.name and (d / "predictions.npz").exists():
                preds["deep"][seed_of(d)] = load(d / "predictions.npz")
        for d in glob.glob(str(runs / "stack" / f"*_{t}_R1b_*")):
            d = Path(d)
            for name in ("stack_base", "stack_emb"):
                f = d / f"predictions_{name}.npz"
                if f.exists() and seed_of(d):
                    preds[name][seed_of(d)] = load(f)
        for d in glob.glob(str(runs / "ensemble" / f"*_{t}_R1b")):
            d = Path(d)
            cfg = json.loads((d / "config.json").read_text())
            res = json.loads((d / "results.json").read_text())
            dp, hp = Path(cfg["deep_run"]), Path(cfg["hybrid_run"])
            s = seed_of(dp)
            if s is None or s != seed_of(hp):
                continue                               # only same-seed pairs count
            dpred = load(runs / "stage2_supervised" / dp.name / "predictions.npz")
            hpred = load(runs / "baselines" / hp.name / "predictions_hybrid.npz")
            w = float(res["deep_weight"])
            preds["ensemble"][s] = ({c: w * dpred[0][c] + (1 - w) * hpred[0][c] for c in dpred[0]},
                                    dpred[1],
                                    {c: w * dpred[2][c] + (1 - w) * hpred[2][c] for c in dpred[2]},
                                    dpred[3])
        if "hybrid" not in preds:
            continue
        ids = sorted(next(iter(preds["hybrid"].values()))[1])
        yt = np.array([next(iter(preds["hybrid"].values()))[1][c] for c in ids])
        groups = [uid.get(c) for c in ids]
        task = {"n_test": len(ids), "models": {}}
        avg = {}
        for name, by_seed in preds.items():
            per = []
            for s, (tp, _, vp, vy) in sorted(by_seed.items()):
                p = np.array([tp[c] for c in ids])
                rep = scoring.author_robust(yt, p, groups)
                row = {"seed": s, "pr_auc": float(average_precision(yt, p)),
                       "one_per_author": rep.get("one_per_author_pr_auc")}
                if vp:                              # F1 / accuracy at the validation threshold
                    vids = sorted(vp)
                    full = scoring.report(np.array([vy[c] for c in vids]), np.array([vp[c] for c in vids]),
                                          yt, p, test_groups=groups)
                    row.update(f1=full["ovid_table"]["f1"], accuracy=full["ovid_table"]["accuracy"])
                per.append(row)
            arr = lambda k: np.array([r[k] for r in per if r.get(k) is not None], dtype=float)
            summ = {"seeds": len(per), "per_seed": per}
            for k in ("pr_auc", "one_per_author", "f1", "accuracy"):
                a = arr(k)
                if len(a):
                    summ[k] = [round(float(a.mean()), 4), round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4)]
            task["models"][name] = summ
            avg[name] = np.mean([[tp[c] for c in ids] for tp, *_ in by_seed.values()], axis=0)
        # paired author-cluster bootstrap of seed-averaged scores vs the hybrid
        g = np.array([str(u) if u is not None else f"u{i}" for i, u in enumerate(groups)])
        authors, inv = np.unique(g, return_inverse=True)
        rows_of = [np.flatnonzero(inv == i) for i in range(len(authors))]
        rng = np.random.default_rng(0)
        diffs = defaultdict(list)
        for _ in range(B):
            idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(authors), len(authors))])
            if yt[idx].min() == yt[idx].max():
                continue
            base = average_precision(yt[idx], avg["hybrid"][idx])
            for name in avg:
                if name != "hybrid":
                    diffs[name].append(average_precision(yt[idx], avg[name][idx]) - base)
        task["vs_hybrid_seed_averaged"] = {
            n: {"mean": round(float(np.mean(v)), 4), "ci95": [round(float(np.percentile(v, 2.5)), 4),
                                                              round(float(np.percentile(v, 97.5)), 4)],
                "p_gt_0": round(float(np.mean(np.array(v) > 0)), 3)} for n, v in diffs.items()}
        out["tasks"][t] = task
        print(f"\n== {t}")
        for name, s in task["models"].items():
            print(f"  {name:12s} seeds={s['seeds']}  PR {s.get('pr_auc')}  1/author {s.get('one_per_author')}  "
                  f"F1 {s.get('f1')}  acc {s.get('accuracy')}")
        for name, v in task["vs_hybrid_seed_averaged"].items():
            print(f"  {name:12s} - hybrid: {v['mean']:+.4f} {v['ci95']}  P>0 {v['p_gt_0']}")
    (rnd / "analysis").mkdir(exist_ok=True)
    (rnd / "analysis" / "seeds.json").write_text(json.dumps(out, indent=1, default=str))
    print("\nwrote", rnd / "analysis" / "seeds.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
