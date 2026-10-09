#!/bin/bash
# MapGuard on the SVKM Altair cluster - one training round per zip, plus the WHERE data job.
#
# Start from a Jupyter terminal in the folder holding mapguard_altair.zip:
#   cd /data/mpstme-tanay/deepLearning_osm
#   unzip -o -j mapguard_altair.zip altair/run_round2.sh -d .
#   MAPGUARD_BASE="$(pwd)" setsid nohup bash run_round2.sh > launch.out 2>&1 < /dev/null &
#
# Folder layout (one folder per round = per zip; re-running the same zip resumes it):
#   <BASE>/rounds/<round>/app/                 code + data + experiments/runs (all runs)
#   <BASE>/rounds/<round>/logs/job.log         every attempt, appended, live
#   <BASE>/rounds/<round>/logs/STATUS          what is running now / how it ended
#   <BASE>/rounds/<round>/logs/timing.csv      step, start, end, seconds, exit code
#   <BASE>/rounds/<round>/logs/env.txt         GPU, TensorFlow, packages
#   <BASE>/rounds/<round>/logs/steps/<step>.log
#   <BASE>/rounds/<round>/<round>_results.zip  send this back (runs + logs + WHERE output)
#   <BASE>/where/                              WHERE job: dump download, extract, features
#                                              (shared by all rounds; done once)
#
# Steps, in order. If smoke fails the job stops; later failures are recorded and skipped.
#   smoke      ovid --quick                 proves the model trains on this GPU (~7 min)
#   main       ovid 00,01,02,07             headline: deep, XGBoost, hybrid, ensemble
#   ablations  ovid 03                      which part helps
#   modern     ovid_modern 01,02,07,06
#   T1, T2     01,02,07,06
#   revert_vandalism, revert  01,02,07
#   where      CPU: planet changeset dump -> distance from the author's usual area (hours)

set -uo pipefail

STEPS="${MAPGUARD_STEPS:-smoke,main,ablations,modern,T1,T2,revert_vandalism,revert,where}"
ZIP_NAME="mapguard_altair.zip"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

START_DIR="$(pwd)"
JOB_ID="${PBS_JOBID:-local}"; JOB_ID="${JOB_ID%%.*}"

# ---------------------------------------------------------------- 1. where to work
find_base() {
    if [ -n "${MAPGUARD_BASE:-}" ]; then echo "$MAPGUARD_BASE"; return; fi
    for d in /data/mpstme-tanay/deepLearning_osm /data/*/deepLearning_osm \
             /data/*/*/deepLearning_osm "$HOME/deepLearning_osm"; do
        if [ -d "$d" ] && [ -w "$d" ]; then echo "$d"; return; fi
    done
    echo "$START_DIR"
}
BASE="$(find_base)"

PY=""
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys" >/dev/null 2>&1; then
        PY="$(command -v "$c")"; break
    fi
done
if [ -z "$PY" ]; then echo "no working python in this image"; exit 1; fi

ZIP=""
for z in "$BASE/$ZIP_NAME" "$START_DIR/$ZIP_NAME"; do
    if [ -f "$z" ] && { [ -z "$ZIP" ] || [ "$z" -nt "$ZIP" ]; }; then ZIP="$z"; fi
done
if [ -z "$ZIP" ]; then echo "$ZIP_NAME not found in $BASE or $START_DIR"; exit 1; fi

ROUND="$("$PY" - "$ZIP" <<'EOF'
import json, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
if "altair/ROUND.txt" in names:
    print(z.read("altair/ROUND.txt").decode().strip())
else:
    print("round_" + json.loads(z.read("BUNDLE_MANIFEST.json"))["generated"][:16].replace(":", "").replace("T", "_"))
EOF
)"
RDIR="$BASE/rounds/$ROUND"
APP="$RDIR/app"
LOGS="$RDIR/logs"
mkdir -p "$LOGS/steps" "$RDIR/.done" "$BASE/where"
exec > >(tee -a "$LOGS/job.log") 2>&1

status() { echo "$(ts)  $*" > "$LOGS/STATUS"; say "STATUS: $*"; }
[ -f "$LOGS/timing.csv" ] || echo "step,start,end,seconds,exit_code,job" > "$LOGS/timing.csv"
JOB_T0=$(date +%s)

echo
echo "=================================================================================="
say "MapGuard | round $ROUND | job $JOB_ID | host $(hostname)"
say "base   : $BASE"
say "zip    : $ZIP"
say "round  : $RDIR"
say "steps  : $STEPS"

# ---------------------------------------------------------------- 2. unpack once per round
"$PY" - "$ZIP" "$APP" <<'EOF' || { status "FAILED: could not unpack $ZIP"; exit 1; }
import json, pathlib, sys, zipfile
zpath, app = sys.argv[1], pathlib.Path(sys.argv[2])
with zipfile.ZipFile(zpath) as z:
    fresh = json.loads(z.read("BUNDLE_MANIFEST.json"))
    cur = app / "BUNDLE_MANIFEST.json"
    if cur.exists() and json.loads(cur.read_text()).get("generated") == fresh.get("generated"):
        print("code + data already unpacked for this round")
    else:
        z.extractall(app)                 # experiments/runs is never in the zip: runs survive
        print("unpacked ->", app)
print("bundle", fresh.get("generated"), "git", str(fresh.get("git_sha"))[:12])
EOF
cd "$APP" || exit 1

# ---------------------------------------------------------------- 3. python packages
export PYTHONUSERBASE="$BASE/.pydeps"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PIP_CACHE_DIR="$BASE/.pipcache"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1
NC="${NCPUS:-$(nproc 2>/dev/null || echo 4)}"
export OMP_NUM_THREADS="$NC"

missing() {
    "$PY" - <<'EOF'
import importlib.util
pk = (("pyarrow", "pyarrow"), ("pandas", "pandas"), ("sklearn", "scikit-learn"), ("xgboost", "xgboost"))
print(" ".join(p for m, p in pk if importlib.util.find_spec(m) is None))
EOF
}
MISSING="$(missing)"
if [ -n "$MISSING" ]; then
    NUMPY="$("$PY" -c 'import numpy; print(numpy.__version__)')"
    say "installing: $MISSING (numpy pinned at $NUMPY)"
    "$PY" -m pip install --user -q $MISSING "numpy==$NUMPY" || true
    MISSING="$(missing)"
    if [ -n "$MISSING" ]; then status "FAILED: could not install $MISSING"; exit 1; fi
fi

{
    echo "== date";   ts
    echo "== host";   hostname
    echo "== cpus";   echo "NCPUS=${NCPUS:-unset} nproc=$(nproc 2>/dev/null)"
    echo "== gpu";    nvidia-smi 2>&1 || echo "nvidia-smi not available"
    echo "== python"; "$PY" --version
    echo "== tensorflow"
    "$PY" -c "import tensorflow as tf; print(tf.__version__, [g.name for g in tf.config.list_physical_devices('GPU')])" 2>&1 | tail -1
    echo "== bundle"; head -12 BUNDLE_MANIFEST.json
    echo "== packages"; "$PY" -m pip list 2>/dev/null
} > "$LOGS/env.txt" 2>&1

# ---------------------------------------------------------------- 4. results packaging
package() {
    "$PY" - "$APP" "$RDIR" "$ROUND" "$BASE/where" <<'EOF' || true
import pathlib, sys, zipfile
app, rdir, name, where = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4])
dest = rdir / f"{name}_results.zip"
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
    for top in ("experiments/runs", "experiments/pipeline"):
        for f in sorted((app / top).rglob("*")) if (app / top).exists() else []:
            if f.is_file():
                z.write(f, f.relative_to(app).as_posix())
    for f in sorted((rdir / "logs").rglob("*")):
        if f.is_file():
            z.write(f, "logs/" + f.relative_to(rdir / "logs").as_posix())
    for n in ("where_features.parquet", "where.log", "STATUS", "source.json"):
        if (where / n).exists():
            z.write(where / n, f"where/{n}")
print(f"results: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
EOF
}
FINAL="unknown"
on_exit() {
    local total=$(( $(date +%s) - JOB_T0 ))
    say "this job: $((total / 3600))h $(((total % 3600) / 60))m"
    echo "JOB_TOTAL,,,${total},,$JOB_ID" >> "$LOGS/timing.csv"
    [ "$FINAL" = "unknown" ] && status "STOPPED before finishing (killed or error) - run the same command again to resume"
    package
}
trap on_exit EXIT
trap 'say "received TERM"; exit 143' TERM

# ---------------------------------------------------------------- 5. the steps
FAILED=()
record() { echo "$1,$2,$(ts),$3,$4,$JOB_ID" >> "$LOGS/timing.csv"; }
run_step() {
    local name="$1"; shift
    if [ -f "$RDIR/.done/$name" ]; then
        say "skip $name - finished earlier in this round"
        return 0
    fi
    status "running $name: $*"
    local t0 start rc
    t0=$(date +%s); start="$(ts)"
    "$@" 2>&1 | tee -a "$LOGS/steps/$name.log"
    rc=${PIPESTATUS[0]}
    local secs=$(( $(date +%s) - t0 ))
    record "$name" "$start" "$secs" "$rc"
    if [ "$rc" -eq 0 ]; then
        date > "$RDIR/.done/$name"
        say "done $name in $((secs / 60)) min"
    else
        FAILED+=("$name")
        say "FAILED $name (exit $rc) after $((secs / 60)) min - see logs/steps/$name.log"
    fi
    return "$rc"
}
pipeline() { "$PY" -u train/run_pipeline.py --feature-store data/interim "$@"; }

IFS=',' read -r -a STEP_LIST <<< "$STEPS"
for step in "${STEP_LIST[@]}"; do
    case "$step" in
        smoke)
            if ! run_step smoke pipeline --task ovid --stages 00,01,02,07,03,06 --quick; then
                FINAL="failed"
                status "FAILED at smoke test - nothing else was run. Send logs/steps/smoke.log"
                exit 1
            fi ;;
        main)             run_step main pipeline --task ovid --stages 00,01,02,07 ;;
        ablations)        run_step ablations pipeline --task ovid --stages 03 ;;
        modern)           run_step modern pipeline --task ovid_modern --stages 01,02,07,06 ;;
        T1)               run_step T1 pipeline --task T1 --stages 01,02,07,06 ;;
        T2)               run_step T2 pipeline --task T2 --stages 01,02,07,06 ;;
        revert_vandalism) run_step revert_vandalism pipeline --task revert_vandalism --stages 01,02,07 ;;
        revert)           run_step revert pipeline --task revert --stages 01,02,07 ;;
        where)
            package       # training results are safe before the long CPU job starts
            if [ -f "$APP/altair/where_targets.parquet" ]; then
                run_step where "$PY" -u altair/where_job.py \
                    --targets altair/where_targets.parquet --workdir "$BASE/where"
            else
                say "skip where - this zip has no altair/where_targets.parquet"
            fi ;;
        *) say "unknown step '$step' - skipped" ;;
    esac
done

if [ ${#FAILED[@]} -eq 0 ]; then
    FINAL="ok"
    status "FINISHED - all steps ok. Send rounds/$ROUND/${ROUND}_results.zip"
else
    FINAL="partial"
    status "FINISHED with failures: ${FAILED[*]}. Send rounds/$ROUND/${ROUND}_results.zip"
fi
exit 0
