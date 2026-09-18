"""Run the training stages in order, on any machine, with one command.

    python train/run_pipeline.py --feature-store data/interim --task ovid --quick
    python train/run_pipeline.py --feature-store data/interim --task ovid
    python train/run_pipeline.py --feature-store data/interim --task T1 --stages 01,02,06

The same command runs on the M4, the SVKM cluster (every notebook in `cluster/`), or
any Linux box with a GPU. It only calls the stage scripts under train/ - it adds no
training logic of its own - and wires one stage's output into the next:

    00 wiring_check            proves TensorFlow and the model work here; stops if it fails
    01 baselines               also freezes the split every later stage reuses
    02 stage2_supervised       R1b full model, plus the Ovid-equivalent reproduction (ovid task)
    04 stage1_pretrain         runs before 03 so the ablation grid can include row 07
       stage1_zero_shot        on the encoder 04 just wrote
    03 stage2_ablations        7 rows x 2 seeds, row 07 uses the stage 04 encoder
    05 stage1_label_efficiency uses the stage 04 encoder
    06 temporal_evaluation     reference = the stage 02 R1b run

`--quick` shrinks epochs, seeds and budgets so the whole chain finishes in minutes. Its
numbers are a smoke test and are tagged `quick` in every run directory name - never report
them.

Every stage writes its own run directory under experiments/runs/, and this script writes a
log of the whole chain to experiments/pipeline/<timestamp>_<task>.log plus a summary JSON
naming every run directory it produced.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TRAIN = Path(__file__).resolve().parent
REPO = TRAIN.parent
RUNS = REPO / "experiments" / "runs"
ORDER = ["00", "01", "02", "04", "03", "05", "06"]

# Temporal cutoffs chosen so both sides hold both classes (see RUNBOOK.md):
#   ovid: train 2014-2018, test 2019 (3,360 changesets, 41% damaging)
#   T1/T2: train 2023-01..2025-06, test 2025-07..2026-09
#   revert: train 2021-09..2024-12, test 2025-01..2026-06
#   ovid_modern: train 2014-2019 (Ovid), test 2021-09..2026-06 (corpus R) - same label rule
CUTOFFS = {"ovid": "2019-01-01T00:00:00Z", "T1": "2025-07-01T00:00:00Z",
           "T2": "2025-07-01T00:00:00Z", "revert": "2025-01-01T00:00:00Z",
           "revert_vandalism": "2025-01-01T00:00:00Z",
           "ovid_modern": "2020-01-01T00:00:00Z", "all": "2019-01-01T00:00:00Z"}


def index_entries() -> list[dict]:
    p = RUNS / "index.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def newest_run(stage: str, *, quick: bool, **match) -> Path | None:
    """Most recent finished run of `stage` for this task, from any earlier session.

    Stages can run in separate jobs (one notebook per stage on the cluster), so stage 05
    finds the encoder stage 04 wrote yesterday. Quick smoke-test runs and real runs never
    feed each other.
    """
    for e in reversed(index_entries()):
        if e["stage"] != stage or not all(e.get(k) == v for k, v in match.items()):
            continue
        d = Path(e["run_dir"])
        d = d if d.is_absolute() else REPO / d
        if d.name.endswith("_quick") == quick and d.exists():
            return d
    return None


class Pipeline:
    def __init__(self, args):
        self.args = args
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out = REPO / "experiments" / "pipeline"
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / f"{stamp}_{args.task}{'_quick' if args.quick else ''}.log"
        self.log = self.log_path.open("w", encoding="utf-8")
        self.summary = {"started": stamp, "task": args.task, "quick": args.quick,
                        "feature_store": str(args.feature_store), "stages": []}
        self.index_start = len(index_entries())

    def common(self) -> list[str]:
        s = ["--set", f"task={self.args.task}",
             "--set", f"data.feature_store={json.dumps(str(self.args.feature_store))}",
             "--set", f"device={self.args.device}"]
        if self.args.task == "ovid_modern":
            # train on A, test on R: R has no adiff geometry, so neither may the model
            s += ["--set", "model.exclude_columns=adiff_only"]
        if self.args.quick:
            s += ["--set", "tag=quick"]
        return s

    def quick(self, *pairs: str) -> list[str]:
        out = []
        if self.args.quick:
            for p in pairs:
                out += ["--set", p]
        return out

    def run(self, name: str, script: str, *extra: str) -> bool:
        cmd = [sys.executable, "-u", str(TRAIN / script), *extra]
        banner = f"\n{'=' * 80}\n[{datetime.now():%H:%M:%S}] {name}\n$ {' '.join(cmd)}\n{'=' * 80}"
        print(banner, flush=True)
        self.log.write(banner + "\n")
        t0 = time.monotonic()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            self.log.write(line)
        proc.wait()
        ok = proc.returncode == 0
        entry = {"stage": name, "ok": ok, "returncode": proc.returncode,
                 "minutes": round((time.monotonic() - t0) / 60, 1)}
        self.summary["stages"].append(entry)
        self.log.flush()
        print(f"[{'OK' if ok else 'FAILED'}] {name} in {entry['minutes']} min", flush=True)
        return ok

    def finish(self) -> int:
        self.summary["runs_created"] = [e["run_dir"] for e in index_entries()[self.index_start:]]
        self.summary["all_ok"] = all(s["ok"] for s in self.summary["stages"])
        path = self.log_path.with_suffix(".json")
        path.write_text(json.dumps(self.summary, indent=2), encoding="utf-8")
        print("\n" + json.dumps(self.summary, indent=2))
        print(f"\nlog: {self.log_path}\nsummary: {path}")
        self.log.close()
        return 0 if self.summary["all_ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feature-store", required=True,
                    help="directory holding changesets/edits/user_seq.parquet")
    ap.add_argument("--task", default="ovid",
                    choices=("ovid", "revert", "revert_vandalism", "ovid_modern", "T1", "T2"))
    ap.add_argument("--stages", default=",".join(ORDER),
                    help="comma list from 00,01,02,03,04,05,06 (always run in pipeline order)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu"))
    ap.add_argument("--quick", action="store_true", help="smoke test: tiny epochs, never report")
    ap.add_argument("--cutoff", default=None, help="override the temporal cutoff")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed stage")
    args = ap.parse_args()
    args.feature_store = Path(args.feature_store).resolve()
    wanted = {s.strip().zfill(2) for s in args.stages.split(",") if s.strip()}

    for name in ("changesets.parquet", "edits.parquet", "user_seq.parquet"):
        if not (args.feature_store / name).exists():
            raise SystemExit(f"{args.feature_store / name} not found - build the feature store "
                             "first (RUNBOOK.md step 2) or point --feature-store at it")

    p = Pipeline(args)
    t = args.task

    def stop(ok: bool) -> bool:
        return not ok and not args.keep_going

    encoder = newest_run("stage1_pretrain", quick=args.quick, task=t)
    ref_run = newest_run("stage2_supervised", quick=args.quick, task=t, split="R1b")
    for s in ORDER:
        if s not in wanted:
            continue
        if s == "00":
            if stop(p.run("00 wiring_check", "00_wiring_check/wiring_check.py", "--device", args.device)):
                break
        elif s == "01":
            if stop(p.run("01 baselines", "01_baselines/baselines_train.py", *p.common())):
                break
        elif s == "02":
            q = p.quick("train.epochs=2")
            if stop(p.run("02 stage2 R1b", "02_stage2_supervised/stage2_train.py", *p.common(), *q)):
                break
            ref_run = newest_run("stage2_supervised", quick=args.quick, task=t, split="R1b")
            if t == "ovid":
                p.run("02 Ovid reproduction (R1b, Ovid's features and architecture)",
                      "02_stage2_supervised/stage2_train.py",
                      "--config", str(TRAIN / "02_stage2_supervised" / "ovid_repro.json"),
                      *p.common(), *q)
        elif s == "04":
            q = p.quick("pretrain.epochs=1", "pretrain.max_edits=20000")
            if stop(p.run("04 stage1 pretrain", "04_stage1_pretrain/stage1_pretrain.py", *p.common(), *q)):
                break
            enc = newest_run("stage1_pretrain", quick=args.quick, task=t)
            if enc is not None:
                encoder = enc
                p.run("04 stage1 zero-shot", "04_stage1_pretrain/stage1_zero_shot.py", "--run", str(enc))
        elif s == "03":
            extra = ["--set", f"pretrained_encoder={json.dumps(str(encoder))}"] if encoder else []
            q = p.quick("train.epochs=1", "seeds=[20260904]")
            if stop(p.run("03 stage2 ablations", "03_stage2_ablations/run_ablations.py",
                          *p.common(), *extra, *q)):
                break
        elif s == "05":
            if encoder is None:
                print("skip 05: no finished stage 04 encoder for this task - run 04 first", flush=True)
                p.summary["stages"].append({"stage": "05 label efficiency", "ok": True,
                                            "skipped": "no encoder"})
                continue
            q = p.quick("train.epochs=1", "seeds=[20260904]", "fractions=[0.25,1.0]")
            if stop(p.run("05 label efficiency", "05_stage1_label_efficiency/label_efficiency.py",
                          *p.common(), "--set", f"pretrained_encoder={json.dumps(str(encoder))}", *q)):
                break
        elif s == "06":
            cutoff = args.cutoff or CUTOFFS[t]
            extra = ["--set", f"temporal.cutoff={cutoff}"]
            if ref_run is not None:
                extra += ["--set", f"reference_run={json.dumps(str(ref_run))}"]
            q = p.quick("train.epochs=2")
            if stop(p.run("06 temporal", "06_temporal_evaluation/temporal_train_eval.py",
                          *p.common(), *extra, *q)):
                break
    return p.finish()


if __name__ == "__main__":
    raise SystemExit(main())
