"""Round 3 analysis. Reads raw/ only; writes analysis/round3.json and prints tables.

Round 3 ran twice concurrently in one folder (two launches), so every task has two replicate
runs ("copy 1" = earlier timestamp). Both are reported; they share config, data and split.
"""
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(r"C:\AyushGarg\projects\personal\osm_mapguard")
ROUND = REPO / "experiments" / "rounds" / "r3_2026-10-12_altair_h100_v3explore"
RUNS = ROUND / "raw" / "experiments" / "runs"
TASKS = ["ovid", "ovid_modern", "T1", "T2", "revert_vandalism", "revert"]


def runs(stage, task, suffix=""):
    ds = sorted(d for d in glob.glob(str(RUNS / stage / f"*_{task}_R1b{suffix}")) if not d.endswith("_quick"))
    return [Path(d) for d in ds]


def hl(h):
    return {"pr_auc": h["test_pr_auc"][0], "ci": h["test_pr_auc"][1:],
            "one_per_author": h["test_author_robust"]["one_per_author_pr_auc"],
            "f1": h["test_ovid_table"]["f1"], "acc": h["test_ovid_table"]["accuracy"]}


out = {"tasks": {}, "ablations": {}, "provenance": {}}
for t in TASKS:
    rec = defaultdict(list)
    for d in runs("baselines", t):
        r = json.load(open(d / "results.json"))
        for k, v in r["models"].items():
            if k.startswith(("hybrid", "xgboost")):
                rec[k].append(hl(v["headline"]) if "headline" in v else {"skipped": v.get("skipped")})
    for d in runs("stage2_supervised", t):
        rec["deep_v2.1"].append(hl(json.load(open(d / "results.json"))["headline"]))
    for d in runs("ensemble", t):
        r = json.load(open(d / "results.json"))
        rec["ensemble"].append({**hl(r["headline"]), "deep_weight": r["deep_weight"],
                                "weight_spread": r.get("bootstrap_weight_spread"),
                                "deep_run": Path(r.get("evaluation", {}).get("deep_run", "") or "").name})
        cfg = json.load(open(d / "config.json"))
        out["provenance"].setdefault(t, []).append({"ensemble": d.name, "deep_run": Path(cfg["deep_run"]).name,
                                                    "hybrid_run": Path(cfg["hybrid_run"]).name})
    for d in runs("stack", t, "_stack"):
        r = json.load(open(d / "results.json"))
        for k, v in r["models"].items():
            rec[k].append(hl(v["headline"]) if "headline" in v else {"skipped": v.get("skipped")})
        rec["stack_folds_test_pr_auc"].append([f["test_pr_auc"][0] for f in r["folds"]])
    out["tasks"][t] = dict(rec)

# ablations: rows x seeds x two copies
abl = defaultdict(lambda: defaultdict(list))
for d in sorted(glob.glob(str(RUNS / "stage2_ablations" / "*"))):
    if d.endswith("_quick"):
        continue
    name = os.path.basename(d).split("ovid_R1b_")[1]
    row, seed = name.rsplit("_s", 1)
    h = json.load(open(Path(d) / "results.json"))["headline"]
    abl[row][seed].append((h["test_pr_auc"][0], h["test_author_robust"]["one_per_author_pr_auc"],
                           h["test_ovid_table"]["f1"]))
for row, seeds in abl.items():
    allv = np.array([v for s in seeds.values() for v in s])
    rep = [abs(s[0][0] - s[1][0]) for s in seeds.values() if len(s) == 2]
    out["ablations"][row] = {"runs": len(allv), "pr_auc_mean": round(float(allv[:, 0].mean()), 4),
                             "pr_auc_sd": round(float(allv[:, 0].std(ddof=1)), 4),
                             "one_per_author_mean": round(float(allv[:, 1].mean()), 4),
                             "f1_mean": round(float(allv[:, 2].mean()), 4),
                             "same_seed_replicate_abs_diff": [round(x, 4) for x in rep]}
grids = sorted(glob.glob(str(RUNS / "stage2_ablations_grid" / "*")))
out["ablation_rows_skipped"] = [json.load(open(Path(g) / "results.json")).get("skipped") for g in grids
                                if not g.endswith("_quick")]

(ROUND / "analysis").mkdir(exist_ok=True)
(ROUND / "analysis" / "round3.json").write_text(json.dumps(out, indent=1, default=str))

f = lambda xs, k: " / ".join(f"{x[k]:.4f}" if k in x else "skip" for x in xs)
print(f"{'task':17s} {'hybrid':>15s} {'hybrid_ctx':>15s} {'deep v2.1':>15s} {'ensemble':>15s} {'stack_base':>15s} {'hyb same split':>15s}")
for t, rec in out["tasks"].items():
    print(f"{t:17s} {f(rec['hybrid'],'pr_auc'):>15s} {f(rec.get('hybrid_ctx',[]),'pr_auc'):>15s} "
          f"{f(rec['deep_v2.1'],'pr_auc'):>15s} {f(rec['ensemble'],'pr_auc'):>15s} "
          f"{f(rec.get('stack_base',[]),'pr_auc'):>15s} {f(rec.get('hybrid_base_same_split',[]),'pr_auc'):>15s}")
print("\none-per-author:")
for t, rec in out["tasks"].items():
    print(f"{t:17s} {f(rec['hybrid'],'one_per_author'):>15s} {f(rec.get('hybrid_ctx',[]),'one_per_author'):>15s} "
          f"{f(rec['deep_v2.1'],'one_per_author'):>15s} {f(rec['ensemble'],'one_per_author'):>15s} "
          f"{f(rec.get('stack_base',[]),'one_per_author'):>15s}")
print("\nF1 (Ovid protocol):")
for t, rec in out["tasks"].items():
    print(f"{t:17s} {f(rec['hybrid'],'f1'):>15s} {f(rec.get('hybrid_ctx',[]),'f1'):>15s} "
          f"{f(rec['deep_v2.1'],'f1'):>15s} {f(rec['ensemble'],'f1'):>15s} {f(rec.get('stack_base',[]),'f1'):>15s}")
print("\nskipped variants:", {t: [k for k, v in rec.items() if isinstance(v, list) and v and isinstance(v[0], dict) and "skipped" in v[0]] for t, rec in out["tasks"].items()})
print("\nablations (ovid; 3 seeds x 2 copies):")
for row, v in out["ablations"].items():
    print(f"  {row:14s} PR {v['pr_auc_mean']:.4f} sd {v['pr_auc_sd']:.4f}  1/author {v['one_per_author_mean']:.4f}  F1 {v['f1_mean']:.4f}  same-seed |diff| {v['same_seed_replicate_abs_diff']}")
print("rows skipped:", out["ablation_rows_skipped"])
print("\nensemble pairing:", json.dumps(out["provenance"]["ovid"]))
