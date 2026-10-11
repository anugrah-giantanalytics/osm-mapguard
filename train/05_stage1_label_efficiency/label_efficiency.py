"""[STAGE 1 -> STAGE 2 · R4] Pretrained vs scratch across label budgets.

Run order 5 of 6. Plan stage: Stage 1 transfer into Stage 2, result R4. Needs a finished
stage1_pretrain run whose zero-shot check was run. Cluster.

    python train/05_stage1_label_efficiency/label_efficiency.py \\
        --set pretrained_encoder=experiments/runs/stage1_pretrain/<dir>

The plan's single most important figure: if the pretrained curve sits above the scratch
curve, especially at 10% of labels, the self-supervision claim is demonstrated in one plot.
If it does not, that is also a finding - "label scarcity is not the bottleneck".

How the comparison is kept fair:

- **Same split, same validation, same test** for every point. Only the training rows shrink.
- **Label budgets are nested**: the 10% subset is inside the 25% subset, which is inside 50%.
  Independent draws would add sampling noise that has nothing to do with label count.
- **Subsets are stratified by class**, so a small budget does not accidentally change the
  class balance.
- **Scratch and pretrained see identical rows** at every (budget, seed).

The pretraining itself used unlabelled edits from the whole training fold. That is the
intended setting - unlabelled edits are abundant, labels are not - and the paper should
state it, because it is what makes "10% of labels" meaningful.
"""

import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN.parent / "src")]

import numpy as np  # noqa: E402

from common import config as config_mod  # noqa: E402
from common import data_contract, preprocess, splits_io, stage  # noqa: E402
from common.fit import train_stage2  # noqa: E402
from common.run import Run, configure_device, seed_everything  # noqa: E402


def nested_budgets(y: np.ndarray, train_mask: np.ndarray, fractions, seed: int) -> dict:
    """Class-stratified, nested subsets of the training rows, one per fraction."""
    rng = np.random.default_rng(seed)
    order = {}
    for cls in (0, 1):
        rows = np.flatnonzero(train_mask & (y == cls))
        order[cls] = rows[rng.permutation(len(rows))]
    out = {}
    for f in sorted(fractions):
        m = np.zeros(len(y), bool)
        for cls, rows in order.items():
            k = max(1, int(round(len(rows) * f)))
            m[rows[:k]] = True
        out[f] = m
    return out


def main() -> int:
    args = stage.parse_args(__doc__.splitlines()[0], Path(__file__).with_name("config.json"))
    base = config_mod.load(args.config, args.set)
    enc = base.get("pretrained_encoder")
    if not enc:
        raise SystemExit("set pretrained_encoder to a stage1_pretrain run directory")
    if not (Path(enc) / "edit_encoder.npz").exists():
        raise SystemExit(f"{enc} has no edit_encoder.npz")

    devices = configure_device(base.get("device", "auto"))
    task, split, split_seed = base["task"], base["split"], int(base.get("split_seed", base["seed"]))
    fractions = [float(f) for f in base.get("fractions", [0.1, 0.25, 0.5, 1.0])]
    seeds = [int(s) for s in base.get("seeds", [split_seed])]

    store = data_contract.load(base.get("data", {}).get("feature_store"), task=task)
    prep = preprocess.prepare(store, base)
    masks, split_path, split_summary = splits_io.get_split(
        prep, task=task, split=split, seed=split_seed, resplit=bool(base.get("resplit", False)))
    if args.dry_run:
        print(f"dry run: {len(fractions)} budgets x {len(seeds)} seeds x 2 inits valid.")
        return 0

    import tensorflow as tf

    points = []
    for seed in seeds:
        budgets = nested_budgets(prep.y.astype(int), masks["train"], fractions, seed)
        for frac, rows in budgets.items():
            for init in ("scratch", "pretrained"):
                tf.keras.backend.clear_session()
                seed_everything(seed, bool(base.get("deterministic", False)))
                cfg = {**base, "stage": "stage1_label_efficiency", "seed": seed,
                       "split_seed": split_seed,
                       "label_fraction": frac, "init_kind": init,
                       "init": {"pretrained_encoder": enc if init == "pretrained" else None,
                                "freeze_encoder": False}}
                run = Run(cfg, tag=f"{init}_f{int(frac * 100)}_s{seed}")
                run.provenance(devices=devices, data_files=store.files, split_file=split_path,
                               extra={"split_summary": split_summary,
                                      "n_train_rows_used": int(rows.sum())})
                print(f"\n=== {init}  {frac:.0%} labels  seed {seed}  ({int(rows.sum())} rows) ===")
                res = train_stage2(cfg, prep, masks, run, train_rows=rows)
                res.update({"label_fraction": frac, "init_kind": init})
                run.finish(res)
                points.append({"fraction": frac, "init": init, "seed": seed,
                               "n_train": int(rows.sum()), "run_dir": str(run.dir),
                               **res["headline"]})

    curve = {}
    for init in ("scratch", "pretrained"):
        curve[init] = []
        for frac in sorted(fractions):
            vals = [p["test_pr_auc"][0] for p in points if p["init"] == init and p["fraction"] == frac]
            curve[init].append({"fraction": frac, "test_pr_auc_mean": round(float(np.mean(vals)), 4),
                                "test_pr_auc_std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None,
                                "n_seeds": len(vals)})
    gap = {c["fraction"]: round(p["test_pr_auc_mean"] - c["test_pr_auc_mean"], 4)
           for c, p in zip(curve["scratch"], curve["pretrained"])}

    out = Run({**base, "stage": "stage1_label_efficiency_curve"}, tag="curve")
    out.provenance(devices=devices, data_files=store.files, split_file=split_path,
                   extra={"pretrained_encoder": enc})
    out.finish({"headline": {"pretrained_minus_scratch_pr_auc": gap}, "curve": curve,
                "points": points, "pretrained_encoder": enc})
    print("\npretrained minus scratch, test PR-AUC by label fraction:", gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
