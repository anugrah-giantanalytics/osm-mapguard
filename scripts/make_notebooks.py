"""Generate the thin Colab/Kaggle notebooks.

Notebooks are generated, not hand-edited. CONTRIBUTING says they stay thin - git pull,
install, import from `src/`, run - and the cheapest way to keep that true is to make the
notebook a build artefact of a script that has no room for a 2000-line cell.

Hand-editing one is allowed; regenerating overwrites it, so move any logic worth keeping
into `src/mapguard/` first, which is where it belonged anyway.

    python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/ayushmgarg/osm_mapguard.git"   # update once the remote exists
OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().split("\n"),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": [
            {**c, "source": [line + "\n" for line in c["source"][:-1]] + [c["source"][-1]]}
            for c in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = f"""
# Thin by design. Everything real lives in src/mapguard/ - see CONTRIBUTING.md.
!git clone -q {REPO} mapguard 2>/dev/null || (cd mapguard && git pull -q)
%cd mapguard
!pip install -q -r requirements.txt
import sys; sys.path.insert(0, "src")
print(open("results/gate1a_adiff_batch.json").read()[:200])
"""

DRIVE = """
from google.colab import drive
drive.mount("/content/drive")

import os
from pathlib import Path

# Bulk data never goes in the repo. On Colab it goes to Drive so it survives the
# 12 h session limit and the 90 min idle kill.
DATA = Path("/content/drive/MyDrive/mapguard")
DATA.mkdir(parents=True, exist_ok=True)
os.environ["MAPGUARD_DATA"] = str(DATA)
os.environ["MAPGUARD_CONTACT"] = "ayushmgarg10@gmail.com"   # goes in the User-Agent
print("MAPGUARD_DATA =", DATA)
"""

SECRETS = """
# OSMCha token. Put it in Colab Secrets (key icon, left sidebar) as OSMCHA_TOKEN.
# Never paste it into a cell - notebooks get committed, shared and screenshotted.
from google.colab import userdata
import os
os.environ["OSMCHA_TOKEN"] = userdata.get("OSMCHA_TOKEN")
print("token loaded:", bool(os.environ.get("OSMCHA_TOKEN")))
"""


def nb_fetch() -> dict:
    return notebook([
        md("""
# 01 - Corpus A and D collection

Runs the fetchers on Colab rather than the laptop: better network, and the gigabytes stay
off a machine with 0.77 GB of free RAM.

Everything here is **resumable**. Colab kills sessions at 12 h and after 90 min idle;
rerun the same cell and it picks up where it stopped.

Rate limits are not negotiable - the OSM API is donated infrastructure. The shared token
bucket in `mapguard.data.http` enforces them globally, so do not raise `--rate` to
"go faster".
"""),
        code(SETUP),
        code(DRIVE),
        code(SECRETS),
        md("## Corpus A - metadata\n\n~18,276 changesets at 3/s. About 100 minutes."),
        code("!python scripts/fetch_corpus_a_meta.py --rate 3 --workers 3"),
        md("## Corpus A - contents (osmChange)\n\nThe edits themselves. Measured at 1.2 s per call."),
        code("!python -m mapguard.data.fetch_corpus_a --rate 3 --workers 3"),
        md("""
## Corpus A - geometry via Overpass adiff

**Check `results/gate1b_overpass_capacity.json` first.** Overpass runs two slots per
client and is much slower than the OSM API; the gate says whether the full corpus is
affordable or whether to subset.

If subsetting: sample by changeset id, never by size or user, or the subset correlates
with the label.
"""),
        code("!python scripts/fetch_corpus_a_geometry.py"),
        md("## Corpus A - user histories, point-in-time\n\nOne call per user, not per changeset."),
        code("!python scripts/fetch_user_histories.py --rate 3 --workers 3"),
        md("## Corpus D - live 2026\n\nThe temporal test set. State the exact window; it goes in the paper."),
        code("!python scripts/harvest_corpus_d.py --minutes 1440"),
        code("!python scripts/fetch_osmcha_labels.py --since 2026-01-01 --window-days 7"),
        md("## Build the feature store\n\nThree Parquet tables. Schema frozen end of week 1."),
        code("!python -m mapguard.features.build_store"),
        code("""
import pyarrow.parquet as pq, os
t = pq.read_table(os.environ["MAPGUARD_DATA"] + "/interim/changesets.parquet")
print(t.num_rows, "changesets x", t.num_columns, "features")
print(open("results/../" + "data/interim/build_report.json").read() if False else "")
"""),
    ])


def nb_train() -> dict:
    return notebook([
        md("""
# 02 - Stage 2 training

The changeset classifier: five branches, attention fusion. Runs on a Colab T4.

**Before the first real run**, confirm `results/gate1d_framework_spike.json` says PASS.
That gate checks masking actually masks - a model that attends to padding trains fine,
scores fine, and is wrong, and the failure looks like "geometry did not help".

Checkpoint every epoch. Sessions die at 12 h and after 90 min idle, and an unresumable
run is a run you will do twice.
"""),
        code(SETUP),
        code(DRIVE),
        code("""
import tensorflow as tf
print("TF", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
"""),
        md("## Gate 1d - framework spike\n\nCheap. Run it here if it has not run locally."),
        code("!python scripts/gate1d_framework_spike.py"),
        md("""
## Load the feature store

Built by notebook 01. If you are iterating on the model and not the data, mount Drive and
read the Parquet - do not refetch.
"""),
        code("""
import os
import pyarrow.parquet as pq

INTERIM = os.environ["MAPGUARD_DATA"] + "/interim"
changesets = pq.read_table(INTERIM + "/changesets.parquet").to_pandas()
edits      = pq.read_table(INTERIM + "/edits.parquet").to_pandas()
user_seq   = pq.read_table(INTERIM + "/user_seq.parquet").to_pandas()
print(len(changesets), "changesets |", len(edits), "sampled edits")
changesets["label"].value_counts(dropna=False)
"""),
        md("""
## Splits

**R1a** is Ovid's protocol - a random split. It is user-leaky and must be labelled as such
in every table it appears in. **R1b** is user-disjoint and is the honest number. Report
both; the gap between them is itself a result.
"""),
        code("""
from mapguard.evaluation.splits import assert_user_disjoint, user_disjoint_split, random_split
from mapguard.evaluation.metrics import summary, bootstrap_ci

uids = changesets["uid"].tolist()
r1b = user_disjoint_split(uids)
assert_user_disjoint(r1b, uids)          # raises if any user spans folds
r1a = random_split(len(changesets))      # Ovid protocol - LEAKY, label it everywhere

print("R1b", r1b.summary(changesets["label"].values))
print("R1a", r1a.summary(changesets["label"].values), "<- leaky, for comparability only")
"""),
        md("""
## Weights and Biases

Every run records config, split, seed and git SHA. An ablation grid without that is a
spreadsheet of numbers nobody can reproduce.
"""),
        code("""
import wandb
wandb.login()
run = wandb.init(project="mapguard", config={"split": "R1b", "seed": 20260904})
"""),
        md("## Train\n\nEarly stopping on validation **PR-AUC**, never accuracy - the classes are imbalanced."),
        code("""
# from mapguard.training.stage2 import build_model, train      # WS3, week 2-3
# model = build_model(...)
# train(model, ..., checkpoint_dir=os.environ["MAPGUARD_DATA"] + "/checkpoints")
"""),
        md("""
## Metrics

`summary()` reports the balanced number and the prevalence-corrected ones in **one**
table. The collapse between them is the finding; splitting them across two tables is how
a reader misses it. Recall will not move - only positives carry it - while precision
falls off a cliff. That asymmetry is the operational story of this problem.
"""),
        code("""
# scores = model.predict(x_test).ravel()
# print(json.dumps(summary(y_test, scores), indent=2))
# print("PR-AUC 95% CI:", bootstrap_ci(y_test, scores))
"""),
        md("## Results go to results/*.json, not to stdout"),
        code("""
# import json, subprocess
# sha = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
# json.dump({"git_sha": sha, ...}, open("results/stage2_r1b.json","w"), indent=2)
"""),
    ])


def nb_pretrain() -> dict:
    return notebook([
        md("""
# 03 - Stage 1 pretraining (Kaggle, week 4, conditional)

**Run this only if the week-3 Stage 2 model is stable.** Plan section 13 makes Stage 1
conditional: it is the first thing cut if the timeline slips, and Stage 2 was always
designed to stand alone.

Kaggle, not Colab - 30 GPU-h/week is an explicit quota and this is the longest job.

One pretext task at this timeline: **corruption contrast**. Masked tag modelling and
geometry reconstruction are future work.
"""),
        code(SETUP),
        md("""
## Gate 1e - measure throughput before setting a budget

Edits per second on one P100 at the real batch size. The pretraining corpus size is then
derived, not guessed. 30 GPU-h/week is a ceiling, not a target.
"""),
        code("!python scripts/gate1e_throughput.py    # WS4, week 4"),
        md("""
## Corpus C - disjointness

Corpus C must exclude every changeset id **and every uid** that appears in a Corpus D
evaluation split. Pretraining on the test set would make R4 - the most important figure
in the paper - meaningless.
"""),
        code("!python -m pytest tests/test_pretrain_eval_disjoint.py -q"),
        md("""
## Zero-shot transfer check - before any fine-tuning

Score the frozen corruption head on the real labelled set. A model that separates
real-from-synthetic need not separate real-from-vandalism. If the AUC is near 0.5 the
corruptions are the wrong distribution and Pretext C needs redesigning.

Report the number either way. It is a finding, not a diagnostic.
"""),
        code("# from mapguard.training.stage1 import zero_shot_transfer   # WS4"),
    ])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, nb in (
        ("01_collect_data.ipynb", nb_fetch()),
        ("02_train_stage2.ipynb", nb_train()),
        ("03_pretrain_stage1.ipynb", nb_pretrain()),
    ):
        path = OUT / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"{name:26} {len(nb['cells']):>2} cells")
    print("\nRegenerate with: python scripts/make_notebooks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
