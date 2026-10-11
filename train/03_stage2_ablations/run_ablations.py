"""[STAGE 2 · ablation grid] Train every row in rows/ under every seed, then tabulate.

Run order 3 of 6. Plan stage: Stage 2 ablations (PROJECT_PLAN §11, cut to 7 rows × 2 seeds
by §13). Heavy - intended for the cluster.

    python train/03_stage2_ablations/run_ablations.py
    python train/03_stage2_ablations/run_ablations.py --set rows='["full","plus_edit_meaning"]'
    python train/03_stage2_ablations/run_ablations.py --set pretrained_encoder=experiments/runs/stage1_pretrain/<dir>

**Seeds vary the initialisation, never the split.** Every row and every seed trains on the
frozen split for (task, split, seed=config.seed). If the split moved with the seed, a row
could look better because it drew an easier test fold, and the table would compare folds
instead of branches.

Every row goes through the same `train_stage2` as stage 02, so the rows differ only in
what their row file overrides.

Each (row, seed) writes its own run directory; a final grid directory holds the table with
mean ± std and the change against the base row.
"""

import json
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN.parent / "src")]

import numpy as np  # noqa: E402

from common import config as config_mod  # noqa: E402
from common import data_contract, preprocess, splits_io, stage  # noqa: E402
from common.fit import train_stage2  # noqa: E402
from common.run import Run, configure_device, seed_everything  # noqa: E402

ROWS = Path(__file__).with_name("rows")


def _missing_side_tables(row, store) -> list[str]:
    from common import extras

    m = row.get("overrides", {}).get("model", {})
    need = []
    if m.get("text_emb") and not extras.has_text_embeddings(store):
        need.append(extras.EMBED_FILE)
    if m.get("context_tabular") and not extras.has_object_context(store):
        need.append(extras.CONTEXT_FILE)
    return need


def load_rows(selected):
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ROWS.glob("*.json"))]
    if selected:
        names = {r["name"] for r in rows}
        unknown = set(selected) - names
        if unknown:
            raise SystemExit(f"unknown ablation rows {sorted(unknown)}; available {sorted(names)}")
        rows = [r for r in rows if r["name"] in selected]
    return rows


def main() -> int:
    args = stage.parse_args(__doc__.splitlines()[0], Path(__file__).with_name("config.json"))
    base = config_mod.load(args.config, args.set)
    devices = configure_device(base.get("device", "auto"))
    task, split, split_seed = base["task"], base["split"], int(base.get("split_seed", base["seed"]))
    seeds = [int(s) for s in base.get("seeds", [split_seed])]
    rows = load_rows(base.get("rows"))

    store = data_contract.load(base.get("data", {}).get("feature_store"), task=task)
    print("data:", store.describe())

    if args.dry_run:
        prep = preprocess.prepare(store, base)
        splits_io.get_split(prep, task=task, split=split, seed=split_seed,
                            cutoff=base.get("temporal", {}).get("cutoff"))
        print(f"dry run: {len(rows)} rows x {len(seeds)} seeds valid. Nothing trained.")
        return 0

    import tensorflow as tf

    grid, skipped = [], []
    for row in rows:
        if row.get("requires_pretrained_encoder") and not base.get("pretrained_encoder"):
            skipped.append({"row": row["name"], "reason": "no pretrained_encoder given"})
            print(f"skip {row['name']}: needs --set pretrained_encoder=<stage1 run dir>")
            continue

        missing = _missing_side_tables(row, store)
        if missing:
            skipped.append({"row": row["name"], "reason": f"side table missing: {missing}"})
            print(f"skip {row['name']}: needs {missing} in the feature store")
            continue

        row_cfg = config_mod.deep_merge(base, row.get("overrides", {}))
        row_cfg["stage"] = "stage2_ablations"
        if row.get("requires_pretrained_encoder"):
            row_cfg["init"] = {"pretrained_encoder": base["pretrained_encoder"], "freeze_encoder": False}

        prep = preprocess.prepare(store, row_cfg)
        masks, split_path, split_summary = splits_io.get_split(
            prep, task=task, split=split, seed=split_seed,
            cutoff=base.get("temporal", {}).get("cutoff"),
            resplit=bool(base.get("resplit", False)))

        for seed in seeds:
            tf.keras.backend.clear_session()
            seed_everything(seed, bool(base.get("deterministic", False)))
            run_cfg = {**row_cfg, "seed": seed, "split_seed": split_seed, "ablation_row": row["name"]}
            run = Run(run_cfg, tag=f"{row['name']}_s{seed}")
            run.provenance(devices=devices, data_files=store.files, split_file=split_path,
                           extra={"split_summary": split_summary, "ablation_row": row})
            print(f"\n=== {row['name']}  seed {seed} ===")
            res = train_stage2(run_cfg, prep, masks, run)
            res["ablation_row"] = row["name"]
            run.finish(res)
            grid.append({"row": row["name"], "seed": seed, "run_dir": str(run.dir),
                         "branches": res["branches"], "loss": res["training"]["loss"],
                         **res["headline"]})

    table = tabulate(grid)
    summary_cfg = {**base, "stage": "stage2_ablations_grid"}
    out = Run(summary_cfg, tag="grid")
    out.provenance(devices=devices, data_files=store.files,
                   extra={"rows": [r["name"] for r in rows], "seeds": seeds, "split_seed": split_seed})
    out.finish({"headline": table.get("full"), "table": table, "runs": grid,
                "skipped": skipped})
    print("\n" + json.dumps(table, indent=2))
    return 0


def tabulate(grid: list) -> dict:
    """mean ± std per row over seeds, plus the change against base_edits_tabular."""
    by_row: dict = {}
    for g in grid:
        by_row.setdefault(g["row"], []).append(g)
    table = {}
    for name, runs in by_row.items():
        pr = np.array([r["test_pr_auc"][0] for r in runs])
        roc = np.array([r["test_roc_auc"][0] for r in runs])
        pr1 = np.array([r["test_pr_auc_at_1pct_prevalence"] for r in runs])
        table[name] = {
            "n_seeds": len(runs),
            "test_pr_auc_mean": round(float(pr.mean()), 4),
            "test_pr_auc_std": round(float(pr.std(ddof=1)), 4) if len(pr) > 1 else None,
            "test_roc_auc_mean": round(float(roc.mean()), 4),
            "test_pr_auc_at_1pct_mean": round(float(pr1.mean()), 4),
            "branches": runs[0]["branches"],
        }
    base = table.get("base_edits_tabular")
    if base:
        for name, row in table.items():
            row["pr_auc_change_vs_base"] = round(row["test_pr_auc_mean"] - base["test_pr_auc_mean"], 4)
    return table


if __name__ == "__main__":
    raise SystemExit(main())
