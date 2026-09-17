"""Run directories, seeding and provenance.

A number in the paper has to answer: which code, which data, which split, which seed,
which machine. A run directory records all five at the moment the run starts, so the
answer does not depend on anyone remembering.

    experiments/runs/<stage>/<YYYYmmdd-HHMMSS>_<task>_<split>_<tag>/
        config.json        the resolved config, overrides applied
        provenance.json    git sha, library versions, device, data fingerprint
        results.json       what the stage measured
        ...                weights, normaliser, logs - stage specific

`experiments/runs/index.jsonl` gets one line per finished run, so a results table can be
built by reading one file instead of walking directories.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TRAIN = Path(__file__).resolve().parents[1]
REPO = TRAIN.parent
RUNS = REPO / "experiments" / "runs"


def git_sha() -> str:
    """Commit of the code that ran. On the cluster there may be no .git, but a bundle is.

    An unpacked bundle is checked first: unzipped inside some other repository, `git
    rev-parse` would happily report that repository's commit.
    """
    bundle = REPO / "BUNDLE_MANIFEST.json"
    if bundle.exists():
        try:
            return json.loads(bundle.read_text()).get("git_sha", "unknown") + " (bundle)"
        except Exception:
            pass
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass
    for manifest in REPO.glob("*/BUNDLE_MANIFEST.json"):
        try:
            return json.loads(manifest.read_text()).get("git_sha", "unknown") + " (bundle)"
        except Exception:
            continue
    return "unknown"


def git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "train"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return None


def fingerprint_file(path: Path, limit: int = 16 * 1024 * 1024) -> dict:
    """Size, mtime and a hash of the first 16 MB. Enough to notice a swapped file cheaply."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(limit))
    st = path.stat()
    return {"bytes": st.st_size, "mtime": int(st.st_mtime), "sha256_16mb": h.hexdigest()[:16]}


def seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            # Slower, and some GPU kernels refuse to run under it - opt in per run.
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass


def configure_device(device: str) -> list[str]:
    """`cpu` hides GPUs before TensorFlow touches them; `auto` leaves them alone.

    Must run before any model is built - TensorFlow fixes the device list on first use.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return ["cpu (no tensorflow)"]
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    elif device != "auto":
        raise SystemExit(f"device must be 'auto' or 'cpu', got {device!r}")
    return [d.name for d in tf.config.get_visible_devices()]


def library_versions() -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("tensorflow", "keras", "sklearn", "xgboost", "pandas", "pyarrow"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            continue
    return out


class Run:
    """One run directory, created at start, finalised at the end."""

    def __init__(self, cfg: dict, tag: str | None = None, root: Path | None = None):
        self.cfg = cfg
        self.root = Path(root) if root else RUNS
        stage = cfg["stage"]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = [stamp, cfg.get("task") or "na", cfg.get("split") or "na"]
        if tag or cfg.get("tag"):
            parts.append(str(tag or cfg.get("tag")))
        self.dir = self.root / stage / "_".join(parts)
        self.dir.mkdir(parents=True, exist_ok=False)
        self.started = datetime.now(timezone.utc)
        self.path("config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def path(self, name: str) -> Path:
        return self.dir / name

    def write_json(self, name: str, obj) -> Path:
        p = self.path(name)
        p.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
        return p

    def provenance(self, *, devices: list[str], data_files: dict[str, Path] | None = None,
                   split_file: Path | None = None, extra: dict | None = None) -> dict:
        prov = {
            "started_utc": self.started.isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "git_dirty_src_or_train": git_dirty(),
            "command": " ".join(sys.argv),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "devices": devices,
            "libraries": library_versions(),
            "data": {k: fingerprint_file(v) for k, v in (data_files or {}).items() if v.exists()},
            "split_file": str(split_file) if split_file else None,
            "split_fingerprint": fingerprint_file(split_file) if split_file and split_file.exists() else None,
        }
        if extra:
            prov.update(extra)
        self.write_json("provenance.json", prov)
        return prov

    def finish(self, results: dict) -> Path:
        results = {
            "stage": self.cfg["stage"],
            "task": self.cfg.get("task"),
            "split": self.cfg.get("split"),
            "seed": self.cfg.get("seed"),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **results,
        }
        p = self.write_json("results.json", results)
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as fh:
            try:
                rel = str(self.dir.relative_to(REPO))
            except ValueError:
                rel = str(self.dir)
            fh.write(json.dumps({
                "run_dir": rel,
                "stage": results["stage"], "task": results["task"], "split": results["split"],
                "seed": results["seed"], "finished_utc": results["finished_utc"],
                "headline": results.get("headline"),
            }, default=_json_default) + "\n")
        print(f"\nrun directory: {self.dir}")
        return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
