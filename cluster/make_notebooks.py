"""Generate the cluster notebooks.

Notebooks are build artefacts, not source. The rule in CONTRIBUTING is that they stay
thin - unzip, import from `src/`, run, write a results file - and the cheapest way to keep
that true is to generate them from a script that has nowhere to put a 2000-line cell.

Hand-editing a generated notebook is allowed while debugging on the cluster. Regenerating
overwrites it, so move anything worth keeping into `src/mapguard/` first, which is where
it belonged.

    python cluster/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

CLUSTER = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip()}


def notebook(cells: list[dict]) -> dict:
    out = []
    for c in cells:
        lines = c["source"].split("\n")
        out.append({**c, "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]})
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


UNZIP = '''
# Unzip the bundle uploaded through the Altair portal, then work from it.
# Everything real lives in src/mapguard/ - see CONTRIBUTING.md.
import glob, json, os, sys, zipfile, pathlib

BUNDLE = next(iter(sorted(glob.glob("*.zip")) + sorted(glob.glob("/data/**/*.zip", recursive=True))), None)
if BUNDLE and not pathlib.Path("src/mapguard").exists():
    with zipfile.ZipFile(BUNDLE) as z:
        z.extractall(".")
    print("unpacked", BUNDLE)

sys.path.insert(0, "src")
TASK = [p for p in glob.glob("*/BUNDLE_MANIFEST.json")]
if TASK:
    manifest = json.load(open(TASK[0]))
    print(json.dumps({k: manifest[k] for k in ("task","git_sha","git_dirty","generated")}, indent=2))
    if manifest.get("git_dirty"):
        print("WARNING: bundle was built from a dirty tree; git_sha does not fully describe it")
'''.strip()


def nb_env_check() -> dict:
    return notebook([
        md("""
# 00 - environment check

**Run this first, and again whenever the container image changes.** It is cheap and it
catches the one thing that would waste a day.

`tensorflow_pbs:23.07-tf2-py3` is NGC 23.07, which ships **TensorFlow 2.13 / Keras 2.x**.
Gate 1d was verified locally on Keras **3.15**, and the two APIs differ in exactly the
places this model uses them - `keras.ops`, masked attention keywords, pooling signatures.
`src/mapguard/models/stage2.py` is written to build on both. This notebook proves it on
the image that will actually train.

It needs no data and no credentials.
"""),
        code(UNZIP),
        md("## GPU and versions\n\nAsk for **1 GPU**. It is a 40 GB MIG slice of an H100 and the others belong to other people."),
        code('''
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
'''),
        code('''
import sys, platform
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("python  ", platform.python_version())
print("tf      ", tf.__version__)
print("keras   ", keras.__version__)
print("numpy   ", np.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs    ", gpus)
assert gpus, "no GPU visible - resubmit the job with Number of GPUs per Node = 1"
'''),
        md("""
## Does the model build here?

Every branch, then a few subsets, since the ablation grid is the same function with a
different `branches` tuple.
"""),
        code('''
from mapguard.models.stage2 import build_model, focal_loss, verify_masking

CFG = dict(max_edits=32, edit_features=14, text_len=128, vocab_size=56,
           user_steps=32, user_features=4, tabular_features=41)

for branches in [("B5",), ("B2","B5"), ("B3","B5"), ("B4","B5"), ("B1","B2","B3","B4","B5")]:
    m = build_model(branches=branches, raster=16 if "B1" in branches else None, **CFG)
    print(f"{str(branches):34} inputs={len(m.inputs)}  params={m.count_params():>9,}")
'''),
        md("""
## Does it train, and does masking hold?

The masking check is the one that matters. A model that silently attends to padding trains
fine, scores fine, and is wrong - and the failure presents three weeks later as
"geometry did not help".

We perturb **only the padded slots** by 50x noise. A correct model does not move at all.
"""),
        code('''
import numpy as np
model = build_model(branches=("B1","B2","B3","B4","B5"), raster=16, **CFG)
model.compile(optimizer="adam",
              loss={"p_damaging": focal_loss()},
              metrics={"p_damaging": ["accuracy"]})

rng = np.random.default_rng(20260908)
n, K = 64, CFG["max_edits"]
real = rng.integers(1, K + 1, n)
mask = (np.arange(K)[None, :] < real[:, None]).astype("float32")

x = dict(
    edit_features=(rng.normal(size=(n, K, CFG["edit_features"])) * mask[:, :, None]).astype("float32"),
    edit_mask=mask,
    geometry=(rng.normal(size=(n, K, 16, 16, 2)) * mask[:, :, None, None, None]).astype("float32"),
    text=rng.integers(0, CFG["vocab_size"], (n, CFG["text_len"])).astype("int32"),
    user_sequence=rng.normal(size=(n, CFG["user_steps"], 4)).astype("float32"),
    user_mask=np.ones((n, CFG["user_steps"]), "float32"),
    tabular=rng.normal(size=(n, CFG["tabular_features"])).astype("float32"),
)
order = [t.name.split(":")[0] for t in model.inputs]
y = rng.integers(0, 2, n).astype("float32")

hist = model.fit([x[k] for k in order],
                 {"p_damaging": y, "edit_attention_weight": np.zeros((n, K), "float32")},
                 epochs=1, batch_size=16, verbose=0)
print("trained one epoch, loss =", hist.history["loss"][0])

check = verify_masking(model, x)
print("masking:", check)
assert check["masking_holds"], "MASKING LEAKS - do not train on this image until fixed"
'''),
        md("""
## Three outputs

`p(damaging)`, per-edit attention weight, and (at training time) branch attribution.
Output 2 is what tells a reviewer *which edit* to open.
"""),
        code('''
p, w = model.predict([x[k] for k in order], verbose=0)
print("p(damaging)      ", p.shape)
print("edit weights     ", w.shape)
row = 0
print(f"row {row}: {int(real[row])} real edits, most suspicious index = {int(w[row][:real[row]].argmax())}")
'''),
        md("## Save the evidence\n\nDownload `results/cluster_env_check.json` and commit it. A gate with no file did not run."),
        code('''
import json, os, subprocess, datetime
os.makedirs("results", exist_ok=True)
out = {
    "generated": datetime.datetime.now().isoformat(timespec="seconds"),
    "where": "SVKM AI/ML cluster, Altair Access",
    "python": platform.python_version(),
    "tensorflow": tf.__version__,
    "keras": keras.__version__,
    "gpus": [d.name for d in gpus],
    "model_built": True,
    "trained_one_epoch": True,
    "masking": check,
    "verdict": "PASS" if check["masking_holds"] else "FAIL",
}
json.dump(out, open("results/cluster_env_check.json", "w"), indent=2)
print(json.dumps(out, indent=2))
'''),
    ])


def nb_baselines() -> dict:
    return notebook([
        md("""
# 01 - baselines

B0, logistic regression and XGBoost on the real feature store, under both splits.
CPU-only really, but it runs here so every number in the paper comes from one environment.

**Read `results/gate1c_b0_tripwire.json` before trusting anything below.** B0 is the
leakage tripwire: comment length plus editor string, nothing about what was edited. It
passed locally at AUC 0.656, which is a weak prior and not a leak. If it ever climbs
above 0.85, stop.
"""),
        code(UNZIP),
        code('''
import json
import numpy as np
import pyarrow.parquet as pq

changesets = pq.read_table("data/interim/changesets.parquet").to_pandas()
print(len(changesets), "changesets x", changesets.shape[1], "columns")
print(changesets["label"].value_counts(dropna=False).to_dict())
'''),
        md("""
## Splits

**R1a** is Ovid's protocol: a random split. It is user-leaky and every table it appears in
must say so. **R1b** is user-disjoint *and class-stratified* - without the stratification
the folds came out 63/23/32 percent positive against 50 overall, because vandals have one
or two changesets each while prolific mappers are overwhelmingly clean.
"""),
        code('''
from mapguard.evaluation.splits import random_split, user_disjoint_split, assert_user_disjoint

y = changesets["label"].to_numpy(dtype="float32", na_value=0.0)
uids = changesets["uid"].tolist()

r1b = user_disjoint_split(uids, y)
assert_user_disjoint(r1b, uids)
r1a = random_split(len(changesets))

print("R1b user-disjoint:", r1b.summary(y))
print("R1a random (LEAKY):", r1a.summary(y))
'''),
        md("## Baselines\n\nEvery run records split, seed and git SHA. A number without those cannot go in the paper."),
        code('''
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mapguard.evaluation.metrics import summary as metric_summary, bootstrap_ci
from mapguard.training.dataset import TABULAR_FEATURES

X = np.nan_to_num(changesets[[c for c in TABULAR_FEATURES if c in changesets.columns]]
                  .to_numpy(dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)

results = {}
for name, split in (("R1b", r1b), ("R1a_leaky", r1a)):
    tr, te = split.train | split.val, split.test
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    lr.fit(X[tr], y[tr])
    p = lr.predict_proba(X[te])[:, 1]
    res = metric_summary(y[te], p)
    res["pr_auc_ci95"] = bootstrap_ci(y[te], p)
    results[f"logreg_{name}"] = res
    print(name, "PR-AUC", res["by_prevalence"]["0.5"]["pr_auc"], "CI", res["pr_auc_ci95"])
'''),
        code('''
try:
    import xgboost as xgb
except ImportError:
    !pip install -q xgboost
    import xgboost as xgb

for name, split in (("R1b", r1b), ("R1a_leaky", r1a)):
    tr, te = split.train | split.val, split.test
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                            subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr")
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    res = metric_summary(y[te], p)
    res["pr_auc_ci95"] = bootstrap_ci(y[te], p)
    results[f"xgboost_{name}"] = res
    print(name, "PR-AUC", res["by_prevalence"]["0.5"]["pr_auc"], "CI", res["pr_auc_ci95"])
'''),
        md("""
## The number that matters

**R1a minus R1b is a finding, not a nuisance.** It measures how much of a published score
is account memorisation rather than edit understanding - a question this literature has
not asked.
"""),
        code('''
import datetime, os, subprocess
os.makedirs("results", exist_ok=True)
manifest = json.load(open([p for p in __import__("glob").glob("*/BUNDLE_MANIFEST.json")][0]))
out = {
    "generated": datetime.datetime.now().isoformat(timespec="seconds"),
    "git_sha": manifest["git_sha"],
    "feature_schema": manifest.get("feature_schema"),
    "n_changesets": int(len(changesets)),
    "splits": {"R1b": r1b.summary(y), "R1a_leaky": r1a.summary(y)},
    "results": results,
    "leak_note": "R1a is user-leaky and exists only for comparability with Ovid.",
}
json.dump(out, open("results/baselines.json", "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
'''),
    ])


def nb_stage2() -> dict:
    return notebook([
        md("""
# 02 - Stage 2 training

The full five-branch model on the real feature store. This is the research result.

Order matters: `00_env_check` must have passed on this image, and `01_baselines` gives the
number this has to beat. Training first and checking later is how you end up with a
result you cannot explain.
"""),
        code(UNZIP),
        code('''
import json, glob, os, datetime
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from tensorflow import keras

print("tf", tf.__version__, "| keras", keras.__version__, "| GPU", tf.config.list_physical_devices("GPU"))

changesets = pq.read_table("data/interim/changesets.parquet").to_pandas()
edits      = pq.read_table("data/interim/edits.parquet").to_pandas()
user_seq   = pq.read_table("data/interim/user_seq.parquet").to_pandas()
print(len(changesets), "changesets |", len(edits), "sampled edits |", len(user_seq), "history steps")
'''),
        md("""
## Assemble the padded arrays

`build_batch` is where masking bugs are born - each changeset holds a different number of
edits, and if the mask does not line up with the padding the model attends to zeros. The
alignment is unit-tested locally (`tests/test_dataset.py`); the assertion below is the
belt to that braces.
"""),
        code('''
from mapguard.training.dataset import build_batch, standardise, EDIT_FEATURES, TABULAR_FEATURES

batch = build_batch(changesets, edits, user_seq, max_edits=256, max_user_steps=200)
print("edit_features", batch.edit_features.shape, "| edit_mask", batch.edit_mask.shape)
print("mean real edits per changeset:", batch.edit_mask.sum(axis=1).mean().round(2))
assert np.all(batch.edit_features[batch.edit_mask == 0] == 0), "padded slots carry signal"
'''),
        code('''
from mapguard.evaluation.splits import user_disjoint_split, random_split, assert_user_disjoint

y = batch.y
r1b = user_disjoint_split(list(batch.uids), y)
assert_user_disjoint(r1b, list(batch.uids))
print("R1b:", r1b.summary(y))

tr, va, te = r1b.train, r1b.val, r1b.test
tab_tr, tab_va, tab_te = standardise(batch.tabular[tr], batch.tabular[va], batch.tabular[te])
'''),
        md("## Build, verify masking on the REAL model, then train"),
        code('''
from mapguard.models.stage2 import build_model, focal_loss, verify_masking

CFG = dict(max_edits=batch.edit_features.shape[1],
           edit_features=batch.edit_features.shape[2],
           text_len=batch.text.shape[1], vocab_size=int(batch.text.max()) + 2,
           user_steps=batch.user_sequence.shape[1], user_features=batch.user_sequence.shape[2],
           tabular_features=batch.tabular.shape[1])

model = build_model(branches=("B2","B3","B4","B5"), raster=None, **CFG)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss={"p_damaging": focal_loss(), "edit_attention_weight": None},
              metrics={"p_damaging": [keras.metrics.AUC(curve="PR", name="pr_auc")]})
model.summary()

order = [t.name.split(":")[0] for t in model.inputs]
def pack(mask, tab):
    d = dict(batch.inputs()); d["tabular"] = tab
    return [np.asarray(d[k])[mask] if k != "tabular" else tab for k in order]

check = verify_masking(model, {k: v[:64] for k, v in dict(batch.inputs()).items()})
print("masking on the real model:", check)
assert check["masking_holds"]
'''),
        code('''
os.makedirs("experiments/checkpoints", exist_ok=True)
cb = [
    keras.callbacks.ModelCheckpoint("experiments/checkpoints/stage2_epoch{epoch:02d}.weights.h5",
                                    save_weights_only=True),
    keras.callbacks.EarlyStopping(monitor="val_p_damaging_pr_auc", mode="max",
                                  patience=5, restore_best_weights=True),
]
# Early stopping on validation PR-AUC, never accuracy. The classes are balanced here but
# the operating point is not, and accuracy would happily pick the wrong model.

hist = model.fit(
    pack(tr, tab_tr), {"p_damaging": y[tr]},
    validation_data=(pack(va, tab_va), {"p_damaging": y[va]}),
    epochs=40, batch_size=64, callbacks=cb, verbose=2,
)
'''),
        code('''
from mapguard.evaluation.metrics import summary as metric_summary, bootstrap_ci

p, w = model.predict(pack(te, tab_te), verbose=0)
p = p.ravel()
res = metric_summary(y[te], p)
res["pr_auc_ci95"] = bootstrap_ci(y[te], p)
print(json.dumps(res, indent=2, default=float))
'''),
        md("## Save everything a result needs to be reproducible"),
        code('''
manifest = json.load(open(glob.glob("*/BUNDLE_MANIFEST.json")[0]))
model.save("experiments/checkpoints/stage2_final.keras")
out = {
    "generated": datetime.datetime.now().isoformat(timespec="seconds"),
    "git_sha": manifest["git_sha"],
    "feature_schema": manifest.get("feature_schema"),
    "branches": list(model.branch_config),
    "config": CFG,
    "split": r1b.summary(y),
    "seed": 20260904,
    "checkpoint": "experiments/checkpoints/stage2_final.keras",
    "masking_check": check,
    "epochs_run": len(hist.history["loss"]),
    "test": res,
}
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/stage2_r1b.json", "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "test"}, indent=2, default=float))
print("\\nDownload results/stage2_r1b.json and experiments/checkpoints/stage2_final.keras")
'''),
    ])


TASKS = {
    "00_env_check": nb_env_check,
    "01_baselines": nb_baselines,
    "02_stage2_train": nb_stage2,
}


def main() -> int:
    for task, fn in TASKS.items():
        d = CLUSTER / task
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{task}.ipynb"
        path.write_text(json.dumps(fn(), indent=1), encoding="utf-8")
        print(f"{task:20} {len(fn()['cells']):>2} cells -> {path.relative_to(CLUSTER.parent)}")
    print("\nRegenerate with: python cluster/make_notebooks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
