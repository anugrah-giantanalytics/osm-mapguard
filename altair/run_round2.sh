#!/bin/bash
# MapGuard - round 2 on the SVKM Altair cluster (PBS, image tensorflow_pbs:23.07-tf2-py3).
#
# Upload THIS file as "Job Script" and mapguard_altair.zip as "Job Files" (or put the zip in
# /data/mpstme-tanay/deepLearning_osm first). Nothing else to type: the script finds the
# folder, unpacks the code + data once, installs what the image lacks, and runs every step.
#
# What it runs, in order (each step = one call to train/run_pipeline.py):
#   smoke      ovid, --quick              proves the v2 model trains on this GPU (~10 min)
#   main       ovid    00,01,02,07        headline: deep v2, XGBoost, hybrid, ensemble
#   ablations  ovid    03                 which part helps: B6, MIL pooling, author weighting
#   modern     ovid_modern 01,02,07,06    2014-19 -> 2021-26 under the same label rule
#   T1         T1      01,02,07,06        human-reviewed triage
#   T2         T2      01,02,07,06        detection vs random traffic
#   revert_vandalism / revert  01,02,07   the two revert tasks (cheap on a GPU)
#
# If smoke fails the job stops (nothing else would work). Any later step that fails is
# recorded and the job moves on.
#
# Resumable: a finished step leaves a marker, so re-submitting the same job after a crash or
# a walltime kill skips what is already done and continues. A new zip (new code) re-runs all.
#
# Everything lands in <BASE>/outputs/<job>_<time>/ :
#   job.log            the whole job, live (tail -f it)
#   STATUS             what is running right now / how it ended
#   timing.csv         step, start, end, seconds, exit code
#   env.txt            GPU, python, tensorflow, packages
#   steps/<step>.log   full output of each step
#   round2_results.zip experiments/ (every run: config, metrics, predictions, best.weights.h5
#                      checkpoints, training_log.csv per epoch) + these logs - send this back
# The runs themselves live in <BASE>/mapguard_v2/experiments/runs/ and survive across jobs.

set -uo pipefail

STEPS="${MAPGUARD_STEPS:-smoke,main,ablations,modern,T1,T2,revert_vandalism,revert}"
ZIP_NAME="mapguard_altair.zip"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

START_DIR="$(pwd)"
JOB_ID="${PBS_JOBID:-local}"; JOB_ID="${JOB_ID%%.*}"
JOB_NAME="${PBS_JOBNAME:-mapguard}"

# ---------------------------------------------------------------- 1. where to work
find_base() {
    if [ -n "${MAPGUARD_BASE:-}" ]; then echo "$MAPGUARD_BASE"; return; fi
    for d in /data/mpstme-tanay/deepLearning_osm /data/*/deepLearning_osm \
             /data/*/*/deepLearning_osm "$HOME/deepLearning_osm"; do
        if [ -d "$d" ] && [ -w "$d" ]; then echo "$d"; return; fi
    done
    for z in "$START_DIR/$ZIP_NAME" "${PBS_O_WORKDIR:-/nonexistent}/$ZIP_NAME"; do
        if [ -f "$z" ]; then dirname "$z"; return; fi
    done
    echo "$START_DIR"
}
BASE="$(find_base)"
APP="$BASE/mapguard_v2"
OUT="$BASE/outputs/${JOB_NAME}_${JOB_ID}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT/steps"
exec > >(tee -a "$OUT/job.log") 2>&1

status() { echo "$(ts)  $*" > "$OUT/STATUS"; say "STATUS: $*"; }
echo "step,start,end,seconds,exit_code" > "$OUT/timing.csv"
JOB_T0=$(date +%s)

say "MapGuard round 2 | job $JOB_ID ($JOB_NAME) | host $(hostname)"
say "start dir : $START_DIR"
say "base      : $BASE"
say "outputs   : $OUT"
say "steps     : $STEPS"

PY=""
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys" >/dev/null 2>&1; then
        PY="$(command -v "$c")"; break
    fi
done
if [ -z "$PY" ]; then status "FAILED: no working python in this image"; exit 1; fi

# ---------------------------------------------------------------- 2. unpack the zip
ZIP=""
for z in "$START_DIR/$ZIP_NAME" "${PBS_O_WORKDIR:-/nonexistent}/$ZIP_NAME" "$BASE/$ZIP_NAME"; do
    if [ -f "$z" ] && { [ -z "$ZIP" ] || [ "$z" -nt "$ZIP" ]; }; then ZIP="$z"; fi
done
if [ -n "$ZIP" ]; then
    say "zip       : $ZIP"
    "$PY" - "$ZIP" "$APP" <<'EOF' || { echo "unpack failed"; exit 1; }
import json, pathlib, sys, zipfile
zpath, app = sys.argv[1], pathlib.Path(sys.argv[2])
with zipfile.ZipFile(zpath) as z:
    fresh = json.loads(z.read("BUNDLE_MANIFEST.json"))
    cur = app / "BUNDLE_MANIFEST.json"
    if cur.exists() and json.loads(cur.read_text()).get("generated") == fresh.get("generated"):
        print("code + data already unpacked (same bundle)")
    else:
        # experiments/runs is never inside the zip, so earlier results survive an update
        z.extractall(app)
        print("unpacked ->", app)
print("bundle", fresh.get("generated"), "git", str(fresh.get("git_sha"))[:12],
      "schema", fresh.get("feature_schema"))
EOF
    if [ ! -f "$APP/train/run_pipeline.py" ]; then status "FAILED: could not unpack $ZIP"; exit 1; fi
elif [ ! -f "$APP/train/run_pipeline.py" ]; then
    status "FAILED: $ZIP_NAME not found (looked in $START_DIR, \$PBS_O_WORKDIR, $BASE)"
    exit 1
fi
cd "$APP" || exit 1
chmod +x altair/*.sh 2>/dev/null || true
BUNDLE_ID="$("$PY" -c "import json;print(json.load(open('BUNDLE_MANIFEST.json'))['generated'].replace(':','-'))")"
DONE_DIR="$APP/.done/$BUNDLE_ID"
mkdir -p "$DONE_DIR"

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
    say "installing: $MISSING (numpy pinned at $NUMPY - the image's TensorFlow needs it)"
    "$PY" -m pip install --user -q $MISSING "numpy==$NUMPY" || true
    MISSING="$(missing)"
    if [ -n "$MISSING" ]; then
        status "FAILED: could not install $MISSING (no internet on this node?)"
        exit 1
    fi
fi

{
    echo "== date";      ts
    echo "== host";      hostname
    echo "== cpus";      echo "NCPUS=${NCPUS:-unset} nproc=$(nproc 2>/dev/null)"
    echo "== memory";    free -g 2>/dev/null || true
    echo "== gpu";       nvidia-smi 2>&1 || echo "nvidia-smi not available"
    echo "== python";    "$PY" --version
    echo "== tensorflow"
    "$PY" -c "import tensorflow as tf; print(tf.__version__, [g.name for g in tf.config.list_physical_devices('GPU')])" 2>&1 | tail -1
    echo "== bundle";    cat BUNDLE_MANIFEST.json | head -12
    echo "== packages";  "$PY" -m pip list 2>/dev/null
} > "$OUT/env.txt" 2>&1
say "environment written to env.txt:"
sed -n '/== tensorflow/,/== bundle/p' "$OUT/env.txt" | head -3
if ! grep -q "GPU" <(sed -n '/== tensorflow/,/== bundle/p' "$OUT/env.txt"); then
    say "WARNING: TensorFlow sees no GPU - this will run on CPU and be slow. Check 'Number of GPUs per Node' = 1."
fi

# ---------------------------------------------------------------- 4. results packaging
package() {
    "$PY" - "$APP" "$OUT" <<'EOF' || true
import pathlib, sys, zipfile
app, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dest = out / "round2_results.zip"
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
    for top in ("experiments/runs", "experiments/pipeline"):
        for f in sorted((app / top).rglob("*")) if (app / top).exists() else []:
            if f.is_file():
                z.write(f, f.relative_to(app).as_posix())
    for f in sorted(out.rglob("*")):
        if f.is_file() and f != dest:
            z.write(f, "logs/" + f.relative_to(out).as_posix())
print(f"results: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
EOF
    # a copy in the job's own directory, so Altair's "Output Directory" receives it too
    if [ "$START_DIR" != "$OUT" ] && [ -w "$START_DIR" ]; then
        mkdir -p "$START_DIR/mapguard_output"
        cp -r "$OUT"/. "$START_DIR/mapguard_output/" 2>/dev/null || true
    fi
}
FINAL="unknown"
on_exit() {
    local total=$(( $(date +%s) - JOB_T0 ))
    say "total job time: $((total / 3600))h $(((total % 3600) / 60))m"
    echo "TOTAL,,,${total}," >> "$OUT/timing.csv"
    [ "$FINAL" = "unknown" ] && status "STOPPED before finishing (killed, walltime, or error) - resubmit to resume"
    package
}
trap on_exit EXIT
trap 'say "received TERM (walltime?)"; exit 143' TERM

# ---------------------------------------------------------------- 5. the steps
FAILED=()
run_step() {
    local name="$1"; shift
    if [ -f "$DONE_DIR/$name" ]; then
        say "skip $name - already finished in an earlier job for this bundle"
        echo "$name,skipped,,,0" >> "$OUT/timing.csv"
        return 0
    fi
    status "running $name: run_pipeline.py $*"
    local t0 start rc
    t0=$(date +%s); start="$(ts)"
    "$PY" -u train/run_pipeline.py --feature-store data/interim "$@" 2>&1 | tee "$OUT/steps/$name.log"
    rc=${PIPESTATUS[0]}
    local secs=$(( $(date +%s) - t0 ))
    echo "$name,$start,$(ts),$secs,$rc" >> "$OUT/timing.csv"
    if [ "$rc" -eq 0 ]; then
        date > "$DONE_DIR/$name"
        say "done $name in $((secs / 60)) min"
    else
        FAILED+=("$name")
        say "FAILED $name (exit $rc) after $((secs / 60)) min - see steps/$name.log"
    fi
    return "$rc"
}

IFS=',' read -r -a STEP_LIST <<< "$STEPS"
for step in "${STEP_LIST[@]}"; do
    case "$step" in
        smoke)
            if ! run_step smoke --task ovid --stages 00,01,02,07,03,06 --quick; then
                FINAL="failed"
                status "FAILED at smoke test - nothing else was run. Send steps/smoke.log"
                exit 1
            fi ;;
        main)             run_step main --task ovid --stages 00,01,02,07 ;;
        ablations)        run_step ablations --task ovid --stages 03 ;;
        modern)           run_step modern --task ovid_modern --stages 01,02,07,06 ;;
        T1)               run_step T1 --task T1 --stages 01,02,07,06 ;;
        T2)               run_step T2 --task T2 --stages 01,02,07,06 ;;
        revert_vandalism) run_step revert_vandalism --task revert_vandalism --stages 01,02,07 ;;
        revert)           run_step revert --task revert --stages 01,02,07 ;;
        *) say "unknown step '$step' - skipped" ;;
    esac
done

if [ ${#FAILED[@]} -eq 0 ]; then
    FINAL="ok"
    status "FINISHED - all steps ok. Send round2_results.zip"
else
    FINAL="partial"
    status "FINISHED with failures: ${FAILED[*]}. Send round2_results.zip"
fi
exit 0
