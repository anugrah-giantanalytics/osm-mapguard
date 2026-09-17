# MapGuard — v2 build checklist

**Purpose:** execution checklist after reviewing `PROJECT_PLAN.md`, `CHECKLIST.md`,
`docs/PLAN_REVIEW.md`, the current source tree, and the tracked gate artifacts.

**Scope:** preserve the research contribution, make the evidence trail truthful, and
ship a visual, inspectable demo alongside the paper. This file is additive; the v1
checklist remains unchanged for historical reference.

## Current baseline

- [x] Plan review recorded the original data and evaluation flaws and their fixes.
- [x] Gate 1a passed: 93/100 sampled changesets yielded usable geometry.
- [x] Gate 1b measured the real constraint: public Overpass projects to 96.2 hours;
  choose a mirror or an id-sampled subset.
- [x] Gate 1d passed: Keras 3 masked multi-input spike trained and padding perturbation
  was zero.
- [x] Corpus D replication harvest, OSMCha labels, mock data, parser, geometry helpers,
  splits, and padded dataset assembly exist.
- [x] Gate 1c has a tracked result and an explicit leakage decision.
- [~] Corpus A has been fully fetched, geometry-enriched, featurised, and schema-frozen.
  *Fetched: yes, 18,276/18,276 metadata and contents. Geometry: 1,055 adiff + 65 skipped of the 5,972-changeset subset. Featurised: no. Schema frozen: no.*
- [ ] A real trainable model, checkpoint, inference path, and experiment results exist.
  *Training code exists in `train/00`–`06`; none of the TensorFlow code has run yet.*
- [ ] A visual demo exists and scores the same input through the same inference path as
  the research results.

## Audit 2026-09-17 — what blocks the first real training run

Legend used from this audit on: `[x]` done and verified on real data · `[~]` code exists
but has not been run on real data or is partial · `[ ]` nothing yet.

How it was checked: row counts read directly from every store in `C:\osm-data\raw`. The test suite
passed (103). Every training script byte-compiled. The feature builder was run on 400 real
changesets. `01_baselines` ran end to end on that probe, and `--dry-run` passed for stages
02, 03 and 04. All of this ran in an isolated copy, so no split or run was written to the repo.

### A. Make Corpus A trainable (the `ovid` task) — blocks everything
- [ ] Fix `build_store.py`: read table `changeset` keyed `id`, write `corpus="A"`, and fix
  the `labelled` count in `build_report.json` (it counts cids, not labels).
- [ ] Fetch `users.sqlite` (account creation dates) for the 8,768 Corpus A uids.
  `account_age_days` is currently null everywhere, so B0 and B5 each lose a feature.
- [ ] Fetch point-in-time histories for those 8,768 users (20 done). Without them B4
  is empty on 99% of rows, and the B4 ablation row measures nothing.
- [ ] Geometry decision: at 122 s per changeset, the full 5,972 subset needs ~160 more
  hours. Either stop at a fixed count (e.g. 2,000) and report B1 only on that subset, or
  run parallel streams against a mirror. Write the decision down.
- [ ] Build the full store into `C:\osm-data\interim`, commit `build_report.json` to
  `results/`, and write the schema manifest.
- [ ] Change the default `task` in `train/*/config.json` from `T2` to `ovid` until
  corpora M and N exist. T2 is unreachable today.

### B. Prove the TensorFlow code runs — blocks every neural result
- [ ] Set up Python 3.11 or 3.12 with TensorFlow on the M4 or the cluster. This laptop is on
  3.14 with no TensorFlow, so no line of the rewritten model has ever executed.
- [ ] Run `train/00_wiring_check` on both Keras 2 (cluster) and Keras 3 (M4), and fix what fails.
- [ ] Regenerate `cluster/` notebooks to call the `train/` scripts. `02_stage2_train.ipynb`
  still fits the two-output model and saves `.keras` — the pattern that breaks.

### C. Modern tasks T1/T2 and the temporal test — optional, decide now
- [ ] No OSMCha changeset has contents fetched: 90,725 labels, 0 osmChange downloads.
  Fetcher and builder for corpus M do not exist (4k contents at 3/s ≈ 25 min — cheap).
- [ ] No corpus N sampler exists (random unreviewed changesets from the same window).
- [ ] 318 OSMCha gap windows are unresolved (2023–2026, both classes).
- [ ] 2026 labels are 8,209 harmful vs 1,187 cleared, so a 2026 temporal test is ~87%
  positive. Report prevalence-corrected metrics, or move the cutoff earlier.
- [ ] Modern geometry at 122 s per changeset is not achievable for 4k. T1/T2 would run
  without B1.

### D. Claims that have to match the evidence
- [ ] "Self-supervised": there is no unlabelled Corpus C. Stage 1 pretrains on training-fold
  edits without their labels. The only evidence for the claim is the label-efficiency
  curve, so word it that way.
- [ ] "Geometry-aware": geometry covers ~6% of Corpus A today. B1 must be evaluated on
  the geometry-covered subset, or its ablation delta will read as zero.
- [ ] Ovid comparison: no reproduction of the published protocol exists yet. Without it,
  the paper cannot say "beats Ovid".

### E. Demo and paper — nothing started
- [ ] Phases 3–5 are all open. With 15 days to 2026-10-02, scope the demo to replay mode on
  committed fixtures through the same inference function. Live mode is a stretch goal.

## V2 operating rules

- [ ] Treat every gate as a stop/go decision. A red gate changes scope before more
  downstream work is built.
- [ ] Keep research truth separate from product polish: no UI claim is accepted unless
  it is backed by a saved score, model version, input changeset id, and feature schema.
- [ ] Do not claim "real-time" unless the demo shows its polling interval, fetch time,
  inference time, and stale/error state.
- [ ] Do not expose raw user names or uid values in screenshots, logs, or hosted output.
- [ ] The demo must work with a recorded fixture when external OSM services, Overpass,
  Google Maps, or credentials are unavailable.
- [~] Every result records config, split, seed, model/checkpoint id, git SHA, and data
  window.
  *`train/common/run.py` records config, seed, split file, git SHA, devices and a data summary. It does not record the data date window explicitly.*

## Phase 0 — reconcile the repo before modelling

- [x] Update the status table in `README.md` from actual `results/*.json` artifacts;
  distinguish verified, running, stale, and not-started work.
- [x] Decide the Gate 1b route in writing: mirror, id-sampled Corpus A, or another
  reproducible route. Record the sampling rule and its coverage consequences.
- [x] Run and save Gate 1c. If the B0 tripwire is high, stop and diagnose leakage before
  training anything.
- [x] Confirm the Python environment is reproducible. The checked-in `.venv311` points
  to a missing Python 3.11 executable; document the supported interpreter and a clean
  setup path.
- [x] Run the no-network test suite in that supported environment and save the command,
  interpreter version, and result.
- [x] Remove or explicitly track stale assumptions in docstrings and docs, especially
  the node-only Corpus A geometry description.
- [~] Reconcile schema names across `build_corpus_a.py`, `build_store.py`,
  `training/dataset.py`, mock Parquet, and the plan. One field name must mean one thing.
- [x] Decide the canonical artifact layout: raw, interim, features, checkpoints,
  fixtures, results. Ensure no large data or secrets enter git.

## Phase 1 — data and feature contract

- [x] Fetch Corpus A metadata and contents with resumability, status accounting, and
  retry/backoff evidence.
  *18,276/18,276 metadata and 18,276/18,276 contents, all status 200.*
- [~] Enrich Corpus A geometry through the selected Gate 1b route and fallback; report
  coverage by object type and by label.
  *1,055 adiff + 65 skipped of 5,972 (the id-hash subset). Averages 122 s per changeset, so the remaining ~4,850 would take ~160 h single-stream. No coverage report yet.*
- [~] Build exact all-edit changeset summaries; never erase scale information.
  *Audit 2026-09-17: code and unit tests exist, but it has never produced a feature store from real data. `build_store.py` crashes on the real Corpus A store (queries table `contents`; the table is `changeset`, keyed `id`).*
- [~] Build the K=256 stratified per-edit sample: displacement, area change, text-bearing
  edits, then random fill. Persist `K/N` and truncation status.
  *Same builder - see above.*
- [~] Build point-in-time user history using only changesets strictly earlier than the
  target; add the explicit future-event regression test.
  *Code and test exist. Histories fetched for 20 of 8,768 Corpus A users. `users.sqlite` (account creation dates) was never fetched, so `account_age_days` is null on every row.*
- [x] Build text features from the agreed tag keys and document the treatment of casing,
  Unicode, empty values, and personal data.
- [~] Freeze and version the feature schema; write a schema manifest with dtypes, units,
  missing-value policy, and provenance for every field.
  *`train/common/data_contract.py` fixes the required tables and columns and refuses anything else. There is no manifest of dtypes, units or missing-value policy.*
- [x] Add a deterministic fixture set containing at least: normal edit, text vandalism,
  geometry displacement, mass change, deleted-account/anonymous case, no-bbox case, and
  a changeset whose geometry requires fallback.
- [~] Verify Corpus C pretraining is disjoint from every Corpus D evaluation changeset
  and uid; save the proof artifact.
  *Corpus C was never fetched. Stage 1 pretrains on training-fold edits minus evaluation authors, and writes `disjointness.json` when it runs.*

## Phase 2 — baselines and research model

- [x] Produce R1a random/user-leaky and R1b user-disjoint splits with persisted ids.
- [~] Produce B0, logistic regression, and XGBoost baselines with bootstrap intervals,
  PR-AUC, precision@k, calibration, and prevalence-corrected curves.
  *`train/01_baselines`. Verified end to end on a 400-changeset real-data probe after patching the builder bug. No reportable run yet.*
- [ ] Reproduce the Ovid feature protocol before claiming improvement; document any
  feature or label mismatch that explains a gap from the published number.
- [~] Implement the smallest end-to-end B5 tabular classifier first and establish a
  working checkpoint/inference contract.
  *`train/03_stage2_ablations/rows/01_base_edits_tabular`. Weights-only checkpoints and a rebuild from `model_config.json`. Passes `--dry-run`; never trained.*
- [~] Implement B2 text, B1 geometry, B3 edit sequence, and B4 point-in-time user
  history branches incrementally; test each branch in isolation.
  *All five are in `src/mapguard/models/stage2.py`. `train/00_wiring_check` tests each branch set but has not been run.*
- [~] Implement masked attention over sampled edits and verify padded edits cannot
  change the output.
  *Gate 1d verified the spike model. The rewritten `stage2.py` checks masking on every run, but it has not run yet.*
- [~] Emit the three promised outputs: damage probability, suspicious edit index/weight,
  and branch/signal attribution.
  *Written: `p_damaging`, `edit_attention_weight` and occlusion in `train/common/attribution.py`. Not yet run.*
- [ ] Train the full supervised model before attempting Stage 1 self-supervision.
  *Script ready: `train/02_stage2_supervised`.*
- [ ] Run the planned ablations and at least two seeds for headline comparisons.
  *Script ready: `train/03_stage2_ablations`, 7 rows x 2 seeds, passes `--dry-run`.*
- [ ] Run Corpus D temporal evaluation and separate temporal drift from label-source
  drift in both metrics and discussion.
  *Script ready: `train/06_temporal_evaluation`. It has no data: no modern changeset has contents or features.*
- [ ] If Stage 1 survives the compute gate, run corruption contrast, zero-shot transfer,
  and label-efficiency experiments. Otherwise record the cut explicitly and remove
  self-supervised claims from the headline conclusion.
  *Scripts ready: `train/04_stage1_pretrain` (corruptions, pretrain, zero-shot) and `train/05_stage1_label_efficiency`.*

## Phase 3 — visual demo definition

### Product shape

- [ ] Choose a small dashboard surface, preferably a local Streamlit/Gradio or static
  web app with a thin Python inference endpoint; avoid building a second ML pipeline.
- [ ] Support two entry modes:
  - [ ] **Replay mode:** choose a saved fixture or changeset id and reproduce a known
    result deterministically.
  - [ ] **Live mode:** enter a changeset id or poll the OSM changeset replication stream
    for a recent candidate, with visible loading, rate-limit, timeout, and stale states.
- [ ] Make the first screen answer, at a glance: what changed, where, how suspicious it
  is, and why a reviewer should inspect it.

### Map experience

- [ ] Show before/after geometry on an interactive map with a clear legend:
  unchanged, created, modified, deleted, and top-ranked suspicious edit.
- [ ] Add zoom-to-change, fit-to-bounds, object-type filters, and a before/after toggle.
- [ ] On selecting an edit, synchronize the map highlight with the detail panel and
  attention/attribution output.
- [ ] Include a side-by-side provider comparison only if it is legally and technically
  available. Default to an OSM-compatible map layer; make Google Maps an optional
  reference panel requiring a user-supplied key and current attribution/usage review.
- [ ] Never scrape, copy, cache, or merge Google tiles/data into OSM output. The demo
  must remain useful with Google disabled.
- [ ] Display required map/data attribution in the UI and in every exported screenshot.

### Dashboard panels

- [ ] **Decision card:** `p(damaging)`, threshold, confidence/calibration note, model
  version, and timestamp.
- [ ] **Change summary:** changeset id, anonymised author identifier if needed, editor,
  operation mix, object counts, bbox, and fetch status.
- [ ] **Why flagged:** geometry/text/user/scale signal bars tied to actual model outputs,
  not hand-written explanations.
- [ ] **Suspicious edits:** ranked table with object type, action, score/attention weight,
  geometry delta, and a map-selection control.
- [ ] **Text diff:** safe old/new tag display with redaction/truncation rules for hostile
  or sensitive content.
- [ ] **History/evaluation panel:** show OSMCha/revert label only when it exists and label
  it as external review evidence, not as model truth.
- [ ] **System status:** data source, cache/fixture/live mode, request latency, inference
  latency, and failure reason.

## Phase 4 — visual demo implementation and evidence

- [ ] Define the demo API contract: input changeset id, normalized feature payload,
  prediction payload, edit ranking, attribution, geometry payload, and provenance.
- [ ] Make CLI inference and dashboard inference call the same function and checkpoint.
- [ ] Add golden-fixture tests that assert the displayed score, ranked edit, and map
  geometry match saved expected output within defined tolerances.
- [ ] Add an offline/demo mode with no network and no API key; it must launch from a
  clean checkout using committed small fixtures only.
- [ ] Add live-mode safeguards: bounded polling, shared request limiter, timeout,
  retry/backoff, response caching, and a visible "data may be stale" indicator.
- [ ] Include at least three scripted scenarios:
  - [ ] clean ordinary edit;
  - [ ] obvious geometry or mass-change case;
  - [ ] semantic/text case with a clear explanation.
- [ ] Capture a short walkthrough showing input → fetch → map before/after → score →
  suspicious edit → explanation. Keep a recorded version for service outages.
- [ ] Export a reproducible demo evidence bundle: screenshots, fixture ids, model id,
  request timestamps, dependency versions, and a one-command launch instruction.
- [ ] Test desktop and narrow-screen layouts, empty states, slow responses, malformed
  changesets, missing geometry, missing model checkpoint, and disabled Google Maps.
- [ ] Do not call the demo production-ready. Label it as a research prototype and make
  false-positive/false-negative limitations visible.

## Phase 5 — paper, presentation, and handoff

- [ ] Make the visual demo a named contribution/artefact in the paper, not an unlinked
  recording.
- [ ] Add one architecture figure and one end-to-end dashboard figure generated from
  the actual implementation.
- [ ] Report model quality separately from demo responsiveness; do not use UI polish as
  evidence of detection quality.
- [ ] Include limitations: incomplete geometry coverage, relations scope, presumed
  negative contamination, OSMCha/revert label differences, calibration drift, and map
  provider/licensing constraints.
- [ ] Verify ODbL/map attribution, Google Maps key handling, privacy redaction, and no
  account-level claims before publishing screenshots or a URL.
- [ ] Re-run the demo from a clean environment and record the exact commit, checkpoint,
  feature-schema version, and fixture manifest.
- [ ] Final acceptance: a reviewer can launch offline, inspect a known result, then run
  one live changeset if credentials/services are available, and obtain the same score
  from the CLI and dashboard.

## Explicitly deferred unless the core path is green

- [ ] Public hosted service with continuous polling.
- [ ] Google Maps as a required dependency.
- [ ] Full planet-scale live monitoring.
- [ ] Mobile-native client.
- [ ] Extra self-supervised pretexts beyond corruption contrast.
- [ ] Relations-specific modelling.
- [ ] Account-level user-risk rankings.

## Final v2 go/no-go checklist

- [ ] One defensible supervised result exists on R1b.
- [ ] The temporal result on Corpus D is reproducible and label-source differences are
  disclosed.
- [ ] No leakage, future information, or pretraining/evaluation overlap remains
  untested.
- [ ] The demo uses the real inference path and exposes the actual evidence behind a
  flag.
- [ ] Offline replay works without secrets; live mode degrades gracefully.
- [ ] Paper, README, results, demo screenshots/recording, and repository status agree.
