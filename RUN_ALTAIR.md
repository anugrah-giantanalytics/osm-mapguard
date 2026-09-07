# Running MapGuard on the SVKM AI/ML cluster

Everything GPU-bound lives in [`cluster/`](cluster/), one folder per task, each with a
single notebook and a `README.md` saying what it needs and what it produces. Nothing in
`cluster/` invents data — every notebook reads artefacts produced by the local fetchers
under `src/mapguard/` and `scripts/`.

## The hardware

| | |
|---|---|
| Compute node | 2× Intel Xeon Gold 6438Y+, 32 cores, **512 GB RAM** |
| GPU | NVIDIA **H100 80 GB**, partitioned by MIG into **4 slices of 40 GB** |
| Master node | 2× Xeon Gold 5420+, 28 cores, 256 GB RAM |
| Portal | Altair Access — `https://10.126.1.10:4443` |
| Scheduler | Altair PBS Professional, queue `workq` |

A 40 GB MIG slice is far more than this project needs. Stage 2 is a small model; the
constraint is data loading and epochs, not memory. Ask for **1 GPU**, not more — the other
slices belong to other people.

## Submitting a job

1. **Portal** → `https://10.126.1.10:4443`, log in.
2. **Jobs → Actions → List Docker Images (hpccluster)**. Copy the image name you want:
   - `tensorflow_pbs:23.07-tf2-py3` — what we use
   - `pytorch_pbs:23.06-py3` — fallback only
3. **Submit Job Using → Jupyter**. Paste the image into **Container Image**.
4. Toggle **Show All** and set:

   | Field | Value |
   |---|---|
   | Job Name | e.g. `mapguard_stage2` |
   | Number of Nodes | `1` |
   | Number of Processors per Node | `8` |
   | **Number of GPUs per Node** | `1` |
   | Queue | `workq` |
   | Amount of Memory (MB) | `32768` |
   | Execution Platform | `linux` |

5. **Job Files** → drag in the bundle (see below).
6. **Submit Job** → note the job id.
7. **Custom Actions → Show Jupyter URL** → open the Jupyter Notebook link.
8. Open the notebook and run it top to bottom.

Files land under `/data/<your-user>/`. `pip install <pkg>` works inside a notebook cell.

### Keeping an environment

Installing packages every session gets old. **Jobs → Actions → Save Docker Container
(hpccluster)**, give it the running job id, a new image name and `/data` as the working
directory. Next submission, use your saved image instead.

## The version trap — read this before writing model code

`tensorflow_pbs:23.07-tf2-py3` is NGC 23.07, which ships **TensorFlow 2.13 and Keras 2.x**.
Gate 1d was verified locally on Keras **3.15**. The two APIs differ in exactly the places
this model uses them.

`src/mapguard/models/stage2.py` is therefore written to build under both:

- imports go through `tf.keras`, never a bare `import keras`
- tensor ops are `tf.*`, never `keras.ops.*` (Keras 3 only)
- no Keras-3-only layer keywords

**`cluster/00_env_check` exists to prove that on the actual image before anything trains.**
Run it first, every time the image changes. If it fails, the fix is to adjust
`stage2.py`, not to pin a different Keras inside the notebook — the model has to keep
working on both machines.

## Preparing a bundle

```bash
python cluster/bundle.py --task 02_stage2_train
```

That writes `cluster/_bundles/02_stage2_train.zip` containing the notebook, `src/`, the
feature Parquet, and a manifest recording the git SHA, feature-schema version and row
counts. Upload the zip through **Files → +Upload**, then unzip it in the first notebook
cell.

Bundles are gitignored. They are build artefacts, and some of them carry data.

## Rules that still apply on the cluster

- **No secrets in a notebook.** The OSMCha token is only needed by the local fetchers.
  Nothing in `cluster/` should ever want it.
- **Metrics go to `results/*.json`**, and every run records config, split, seed,
  checkpoint id, git SHA and data window. Download the results file; do not screenshot
  the output.
- **Checkpoint every epoch.** A job that dies at hour six with nothing on disk is six
  hours gone.
- **Do not fetch OSM data from the cluster.** The polite rate limiter is tuned for one
  machine, and the cluster shares an institutional IP with everyone else.

## Task folders

| Folder | What it runs | Needs |
|---|---|---|
| `00_env_check` | GPU visible, TF/Keras versions, model builds + trains + masking holds | nothing |
| `01_baselines` | B0, logistic regression, XGBoost, Ovid reproduction | `changesets.parquet` |
| `02_stage2_train` | the full five-branch model, R1a and R1b | all three Parquet tables |
| `03_ablations` | six-row grid, two seeds | as above, plus a working 02 |
| `04_stage1_pretrain` | corruption contrast, conditional on week 4 | Corpus C |

Run them in order. Each folder's `README.md` states its inputs, outputs and expected
runtime.
