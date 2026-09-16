"""Run configuration: JSON files with command-line overrides.

JSON rather than YAML or TOML because it needs nothing installed. The cluster image is
Python 3.10 (no `tomllib`) and does not ship PyYAML, and a config format that works on one
machine but not another is exactly the kind of drift that produces unreproducible numbers.

Overrides use dotted paths and JSON-parsed values:

    --set train.epochs=5 --set model.branches='["B2","B5"]' --set task=T1
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

TASKS = ("T1", "T2", "ovid", "all")
SPLITS = ("R1a", "R1b", "temporal")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted: str, raw_value: str) -> None:
    """Apply one `a.b.c=value` override. The value is JSON if it parses, else a string."""
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def load(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        set_dotted(cfg, k.strip(), v.strip())
    validate(cfg)
    cfg["_config_path"] = str(path)
    return cfg


def validate(cfg: dict) -> None:
    """Catch the mistakes that would otherwise surface an hour into a run."""
    if "stage" not in cfg:
        raise SystemExit("config is missing 'stage'")
    task = cfg.get("task")
    if task is not None and task not in TASKS:
        raise SystemExit(f"task must be one of {TASKS}, got {task!r}")
    split = cfg.get("split")
    if split is not None and split not in SPLITS:
        raise SystemExit(f"split must be one of {SPLITS}, got {split!r}")
    if split == "temporal" and not cfg.get("temporal", {}).get("cutoff"):
        raise SystemExit("split 'temporal' needs temporal.cutoff, e.g. \"2026-01-01T00:00:00Z\"")
    if not isinstance(cfg.get("seed", 0), int):
        raise SystemExit("seed must be an integer")
