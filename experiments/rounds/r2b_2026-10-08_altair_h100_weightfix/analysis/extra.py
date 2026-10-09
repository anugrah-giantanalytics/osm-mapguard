"""Round 2b extras: ablation author-robust table, weighting effect, attention concentration.
Reads raw/ only; writes analysis/ablations_author_robust.json and analysis/attention.json."""
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(r"C:\AyushGarg\projects\personal\osm_mapguard")
ROUND = REPO / "experiments" / "rounds" / "r2b_2026-10-08_altair_h100_weightfix"
RUNS = ROUND / "raw" / "experiments" / "runs"

# --- ablations -----------------------------------------------------------------------
acc = defaultdict(list)
for d in sorted(glob.glob(str(RUNS / "stage2_ablations" / "*"))):
    if d.endswith("_quick"):
        continue
    h = json.load(open(Path(d) / "results.json"))["headline"]
    name = os.path.basename(d).split("ovid_R1b_")[1].rsplit("_s2026", 1)[0]
    acc[name].append((h["test_pr_auc"][0], h["test_author_robust"]["one_per_author_pr_auc"],
                      h["test_ovid_table"]["f1"], h["test_ovid_table"]["accuracy"]))
abl = {}
print(f"{'row':20s} {'PR-AUC':>15s} {'1/author':>15s} {'F1':>6s} {'Acc':>6s}")
for k, v in acc.items():
    a = np.array(v)
    abl[k] = {"seeds": len(v), "pr_auc": [round(a[:, 0].mean(), 4), round(a[:, 0].std(), 4)],
              "one_per_author": [round(a[:, 1].mean(), 4), round(a[:, 1].std(), 4)],
              "f1": round(a[:, 2].mean(), 4), "accuracy": round(a[:, 3].mean(), 4)}
    print(f"{k:20s} {a[:,0].mean():.4f}±{a[:,0].std():.4f} {a[:,1].mean():.4f}±{a[:,1].std():.4f} "
          f"{a[:,2].mean():.3f} {a[:,3].mean():.3f}")
(ROUND / "analysis" / "ablations_author_robust.json").write_text(json.dumps(abl, indent=1))

# --- attention concentration -----------------------------------------------------------
def concentration(path):
    z = np.load(path)
    if "test_attention" not in z:
        return None
    w = z["test_attention"].astype("float64")
    n = z["test_n_edits"].astype(int)
    y = z["test_y"].astype(int)
    keep = n >= 5                                    # only bags where "which edit" is a question
    w, n, y = w[keep], n[keep], y[keep]
    top = w.max(axis=1)
    uniform = 1.0 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(w > 0, w * np.log(w), 0.0)).sum(axis=1) / np.log(n)
    return {"bags_with_5plus_edits": int(keep.sum()),
            "median_edits": float(np.median(n)),
            "median_top_weight": round(float(np.median(top)), 3),
            "median_top_over_uniform": round(float(np.median(top / uniform)), 2),
            "share_top_edit_over_50pct": round(float(np.mean(top > 0.5)), 3),
            "median_normalised_entropy": round(float(np.median(ent)), 3),
            "positives_median_top_over_uniform": round(float(np.median((top / uniform)[y == 1])), 2),
            "negatives_median_top_over_uniform": round(float(np.median((top / uniform)[y == 0])), 2)}

att = {}
for label, pattern in (("main_ovid_mil", "stage2_supervised/*_ovid_R1b"),
                       ("ablation_full_mil", "stage2_ablations/*_full_s20260904"),
                       ("ablation_full_meanpool", "stage2_ablations/*_full_meanpool_s20260904")):
    ds = [d for d in glob.glob(str(RUNS / pattern)) if not d.endswith("_quick")]
    att[label] = concentration(Path(ds[0]) / "predictions.npz") if ds else None
    print(label, att[label])
(ROUND / "analysis" / "attention.json").write_text(json.dumps(att, indent=1))
