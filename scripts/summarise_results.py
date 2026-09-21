"""Summarise a folder of training runs into one table set - reading, never editing.

    python scripts/summarise_results.py experiments/rounds/<round>

Expects `<root>/raw/experiments/{runs,pipeline,splits}` exactly as `train/run_pipeline.py`
wrote them, and writes `<root>/analysis/summary.md` and `summary.json`. Runs whose directory
ends in `_quick` are smoke tests and are listed but never enter a table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

OVID = {"precision": 0.8035, "recall": 0.8302, "f1": 0.8166, "accuracy": 0.8137}


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def real_runs(runs: Path, stage: str) -> list[Path]:
    d = runs / stage
    return sorted(x for x in d.iterdir() if x.is_dir() and not x.name.endswith("_quick")) if d.exists() else []


def task_of(run: Path) -> str:
    return load(run / "config.json").get("task")


def fmt_ci(v) -> str:
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def curve(run: Path) -> dict:
    rows = list(csv.DictReader((run / "training_log.csv").open(encoding="utf-8")))
    return {"epochs": len(rows),
            "best_val_pr_auc": max(float(r["val_pr_auc"]) for r in rows),
            "best_epoch": 1 + max(range(len(rows)), key=lambda i: float(rows[i]["val_pr_auc"])),
            "final_train_pr_auc": float(rows[-1]["pr_auc"]),
            "final_val_pr_auc": float(rows[-1]["val_pr_auc"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    args = ap.parse_args()
    root = Path(args.root)
    runs = root / "raw" / "experiments" / "runs"
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    S: dict = {"source": str(runs), "quick_runs_excluded": []}
    for stage_dir in runs.iterdir() if runs.exists() else []:
        if stage_dir.is_dir():
            S["quick_runs_excluded"] += [x.name for x in stage_dir.iterdir() if x.name.endswith("_quick")]

    # provenance of the real runs
    sup = real_runs(runs, "stage2_supervised")
    prov = load(sup[0] / "provenance.json") if sup else {}
    S["provenance"] = {"git_sha": prov.get("git_sha"), "devices": prov.get("devices"),
                       "libraries": prov.get("libraries"), "platform": prov.get("platform")}

    # headline: baselines vs deep model, per task
    base = {task_of(r): load(r / "results.json")["models"] for r in real_runs(runs, "baselines")}
    S["headline"] = {}
    for r in sup:
        res = load(r / "results.json")
        tag = "ovid_repro" if "ovid_repro" in r.name else "full"
        t = task_of(r)
        S["headline"].setdefault(t, {})[tag] = {
            "run": r.name, "pr_auc": res["headline"]["test_pr_auc"],
            "roc_auc": res["headline"]["test_roc_auc"], "ovid_table": res["headline"]["test_ovid_table"],
            "ece": res["headline"]["test_ece"], "pr_auc_at_1pct": res["headline"]["test_pr_auc_at_1pct_prevalence"],
            "precision_at_k": res["evaluation"]["test"]["precision_at_k"],
            "masking": {k: v["max_delta"] for k, v in res["masking_checks"].items()},
            "curve": curve(r), "attribution": (res.get("attribution") or {}).get("branches")}
        if t in base:
            S["headline"][t]["baselines"] = {
                m: {"pr_auc": v["evaluation"]["test_intervals_95"]["pr_auc"],
                    "roc_auc": v["evaluation"]["test_intervals_95"]["roc_auc"],
                    "ovid_table": {k: v["evaluation"]["ovid_table"][k] for k in OVID}}
                for m, v in base[t].items() if "evaluation" in v}

    # ablations, stage 1, label efficiency, temporal
    grid = real_runs(runs, "stage2_ablations_grid")
    S["ablations"] = load(grid[-1] / "results.json")["table"] if grid else None
    pre = real_runs(runs, "stage1_pretrain")
    S["stage1_pretext"] = load(pre[-1] / "results.json")["pretext"] if pre else None
    zs = real_runs(runs, "stage1_zero_shot")
    S["stage1_zero_shot"] = load(zs[-1] / "results.json")["headline"] if zs else None
    le = real_runs(runs, "stage1_label_efficiency_curve")
    if le:
        c = load(le[-1] / "results.json")
        S["label_efficiency"] = {"curve": c["curve"], "pretrained_minus_scratch": c["headline"]}
    S["temporal"] = {}
    for r in real_runs(runs, "stage2_temporal"):
        res = load(r / "results.json")
        S["temporal"][task_of(r)] = {
            "cutoff": res["cutoff"], "train_window": res["train_window"], "test_window": res["test_window"],
            "pr_auc": res["headline"]["test_pr_auc"], "ovid_table": res["headline"]["test_ovid_table"],
            "drift_vs_R1b": (res.get("drift_vs_reference") or {}).get("change"),
            "worst_months": sorted([m for m in res["test_by_month"] if "roc_auc" in m],
                                   key=lambda m: m["roc_auc"])[:3]}

    (out / "summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")

    # markdown
    L = ["# Training results - summary", "",
         f"Source: `raw/experiments` (untouched). Code `{(S['provenance']['git_sha'] or '')[:10]}`, "
         f"devices {S['provenance']['devices']}, TF {S['provenance']['libraries'].get('tensorflow') if S['provenance']['libraries'] else '?'}.",
         f"Excluded smoke-test runs: {len(S['quick_runs_excluded'])}.", "",
         "## Headline (user-disjoint R1b test set)", "",
         "| Task | Model | PR-AUC [95% CI] | ROC-AUC | P | R | F1 | Acc | ECE |", "|---|---|---|---|---|---|---|---|---|"]
    for t, d in S["headline"].items():
        for name, v in [("deep (full)", d.get("full")), ("deep (Ovid-equivalent)", d.get("ovid_repro"))]:
            if v:
                o = v["ovid_table"]
                L.append(f"| {t} | {name} | {fmt_ci(v['pr_auc'])} | {v['roc_auc'][0]:.3f} | {o['precision']:.3f} | "
                         f"{o['recall']:.3f} | {o['f1']:.3f} | {o['accuracy']:.3f} | {v['ece']:.3f} |")
        for m, v in (d.get("baselines") or {}).items():
            o = v["ovid_table"]
            L.append(f"| {t} | {m} | {fmt_ci(v['pr_auc'])} | {v['roc_auc'][0]:.3f} | {o['precision']:.3f} | "
                     f"{o['recall']:.3f} | {o['f1']:.3f} | {o['accuracy']:.3f} | - |")
    L += ["", f"Ovid (published, 2021): P {OVID['precision']} R {OVID['recall']} F1 {OVID['f1']} Acc {OVID['accuracy']}.", ""]
    if S["ablations"]:
        L += ["## Ablations (ovid, 2 seeds)", "", "| Row | Branches | PR-AUC mean ± std | vs base |", "|---|---|---|---|"]
        for k, v in S["ablations"].items():
            L.append(f"| {k} | {'+'.join(v['branches'])} | {v['test_pr_auc_mean']:.4f} ± {v['test_pr_auc_std']:.4f} | {v['pr_auc_change_vs_base']:+.4f} |")
        L.append("")
    if S.get("label_efficiency"):
        L += ["## Label efficiency (ovid, 2 seeds)", "", "| Labels | Scratch | Pretrained | Diff |", "|---|---|---|---|"]
        sc = {p["fraction"]: p for p in S["label_efficiency"]["curve"]["scratch"]}
        pr = {p["fraction"]: p for p in S["label_efficiency"]["curve"]["pretrained"]}
        for f in sorted(sc):
            L.append(f"| {int(f * 100)}% | {sc[f]['test_pr_auc_mean']:.4f} | {pr[f]['test_pr_auc_mean']:.4f} | "
                     f"{pr[f]['test_pr_auc_mean'] - sc[f]['test_pr_auc_mean']:+.4f} |")
        L += ["", f"Stage 1 pretext val AUC {S['stage1_pretext']['val_auc']}; zero-shot ROC-AUC on real labels "
              f"{S['stage1_zero_shot']['zero_shot_roc_auc']}.", ""]
    if S["temporal"]:
        L += ["## Temporal (train past, test future)", "", "| Task | Train | Test | PR-AUC | F1 | Change vs R1b |", "|---|---|---|---|---|---|"]
        for t, v in S["temporal"].items():
            L.append(f"| {t} | {v['train_window'][0][:7]}→{v['train_window'][1][:7]} | {v['test_window'][0][:7]}→{v['test_window'][1][:7]} | "
                     f"{fmt_ci(v['pr_auc'])} | {v['ovid_table']['f1']:.3f} | {v['drift_vs_R1b']:+.4f} |")
        L.append("")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'summary.md'} and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
