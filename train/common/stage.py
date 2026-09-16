"""The start of every stage script, in one place.

Parse arguments -> load config -> pin device and seed -> validate data against the contract
-> assemble arrays -> load or create the frozen split -> open a run directory with
provenance. Every stage does exactly this, so every stage's results.json can be trusted to
the same standard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config as config_mod
from . import data_contract, preprocess, splits_io
from .run import Run, configure_device, seed_everything


def parse_args(description: str, default_config: Path) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(default_config))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set train.epochs=5")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate config, data and split, then stop before training")
    return ap.parse_args()


def setup(args: argparse.Namespace, *, needs_split: bool = True, tag: str | None = None):
    cfg = config_mod.load(args.config, args.set)
    devices = configure_device(cfg.get("device", "auto"))
    seed_everything(int(cfg.get("seed", 0)), bool(cfg.get("deterministic", False)))

    task = cfg.get("task", "all")
    store = data_contract.load(cfg.get("data", {}).get("feature_store"), task=task)
    print("data:", store.describe())
    prep = preprocess.prepare(store, cfg)

    masks = split_path = summary = None
    if needs_split:
        masks, split_path, summary = splits_io.get_split(
            prep, task=task, split=cfg["split"], seed=int(cfg.get("seed", 0)),
            cutoff=cfg.get("temporal", {}).get("cutoff"),
            resplit=bool(cfg.get("resplit", False)),
            exclude_test_users_from_train=bool(
                cfg.get("temporal", {}).get("exclude_test_users_from_train", True)),
        )
        print("split:", summary)

    if args.dry_run:
        print("\ndry run: config, data contract and split all valid. Nothing trained.")
        raise SystemExit(0)

    run = Run(cfg, tag=tag)
    run.provenance(devices=devices, data_files=store.files, split_file=split_path,
                   extra={"data_summary": store.describe(), "split_summary": summary})
    return cfg, run, store, prep, masks
