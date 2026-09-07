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
- [ ] A real trainable model, checkpoint, inference path, and experiment results exist.
- [ ] A visual demo exists and scores the same input through the same inference path as
  the research results.

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
- [ ] Every result records config, split, seed, model/checkpoint id, git SHA, and data
  window.

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

- [~] Fetch Corpus A metadata and contents with resumability, status accounting, and
  retry/backoff evidence.
- [ ] Enrich Corpus A geometry through the selected Gate 1b route and fallback; report
  coverage by object type and by label.
- [x] Build exact all-edit changeset summaries; never erase scale information.
- [x] Build the K=256 stratified per-edit sample: displacement, area change, text-bearing
  edits, then random fill. Persist `K/N` and truncation status.
- [x] Build point-in-time user history using only changesets strictly earlier than the
  target; add the explicit future-event regression test.
- [x] Build text features from the agreed tag keys and document the treatment of casing,
  Unicode, empty values, and personal data.
- [ ] Freeze and version the feature schema; write a schema manifest with dtypes, units,
  missing-value policy, and provenance for every field.
- [x] Add a deterministic fixture set containing at least: normal edit, text vandalism,
  geometry displacement, mass change, deleted-account/anonymous case, no-bbox case, and
  a changeset whose geometry requires fallback.
- [ ] Verify Corpus C pretraining is disjoint from every Corpus D evaluation changeset
  and uid; save the proof artifact.

## Phase 2 — baselines and research model

- [x] Produce R1a random/user-leaky and R1b user-disjoint splits with persisted ids.
- [~] Produce B0, logistic regression, and XGBoost baselines with bootstrap intervals,
  PR-AUC, precision@k, calibration, and prevalence-corrected curves.
- [ ] Reproduce the Ovid feature protocol before claiming improvement; document any
  feature or label mismatch that explains a gap from the published number.
- [~] Implement the smallest end-to-end B5 tabular classifier first and establish a
  working checkpoint/inference contract.
- [~] Implement B2 text, B1 geometry, B3 edit sequence, and B4 point-in-time user
  history branches incrementally; test each branch in isolation.
- [x] Implement masked attention over sampled edits and verify padded edits cannot
  change the output.
- [x] Emit the three promised outputs: damage probability, suspicious edit index/weight,
  and branch/signal attribution.
- [ ] Train the full supervised model before attempting Stage 1 self-supervision.
- [ ] Run the planned ablations and at least two seeds for headline comparisons.
- [ ] Run Corpus D temporal evaluation and separate temporal drift from label-source
  drift in both metrics and discussion.
- [ ] If Stage 1 survives the compute gate, run corruption contrast, zero-shot transfer,
  and label-efficiency experiments. Otherwise record the cut explicitly and remove
  self-supervised claims from the headline conclusion.

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
