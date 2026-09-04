# MapGuard — build checklist

Every step, in order, with a pass criterion. Nothing here is optional unless it says so.
Tick as you go. If a gate fails, stop and re-plan — do not build past a red gate.

**Conventions**

- `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked · `[-]` dropped, with a reason
- **A/C** = acceptance criterion. An item is not done until its A/C is demonstrably met.
- Owner is the workstream (WS1 data · WS2 eval · WS3 Stage 2 · WS4 Stage 1) or a name.
- Every gate writes a JSON file to `results/`. If there is no file, the gate did not run.

Reference: [`PROJECT_PLAN.md`](PROJECT_PLAN.md) · [`docs/PLAN_REVIEW.md`](docs/PLAN_REVIEW.md)

---

## Phase 0 — Repository and environment hygiene (day 1)

Do this before any code. It is 90 minutes and it prevents a month of mess.

### 0.1 Repository state

- [ ] **0.1.1** Merge the plan-review branch into the mainline, then delete it.
- [ ] **0.1.2** Reconcile the diverged branches. `master`, `setup`, `claude/project-plan-checkpoints-1ec64b` and `claude/project-plan-evaluation-2a0729` currently hold parallel commits with identical messages and different SHAs. Pick one mainline, rebase or cherry-pick the rest onto it, delete the others.
  **A/C:** `git branch -a` lists exactly the mainline plus any live feature branch.
- [ ] **0.1.3** Decide the mainline name — `main` or `master` — and make the remote default match. Right now the local default is `master` while tooling assumes `main`.
- [ ] **0.1.4** Remove all git worktrees and delete `.claude/worktrees/`.
  ```bash
  git worktree list && git worktree remove --force .claude/worktrees/project-plan-evaluation-2a0729 && git worktree prune && rm -rf .claude/worktrees
  ```
  **A/C:** `git worktree list` shows only the main checkout; `.claude/worktrees` does not exist.
- [ ] **0.1.5** Add `.claude/` to `.gitignore` so local agent state never enters history.
- [ ] **0.1.6** Branch naming convention: `ws1/…`, `ws2/…`, `ws3/…`, `ws4/…`, or `fix/…`. **No tool-named branches.**
- [ ] **0.1.7** Commit convention: Conventional Commits, no `Co-Authored-By` trailers for tooling, no generated-by footers.
- [ ] **0.1.8** Set up the GitHub repository (private), add all four members, enable branch protection on the mainline.
- [ ] **0.1.9** Add `CONTRIBUTING.md`: branch names, commit format, "no data in the repo", "metrics go to `results/*.json`".

### 0.2 Local environment

- [ ] **0.2.1** Create the 3.12 venv. TensorFlow does not support the machine's 3.14.
  ```bash
  py -3.12 -m venv .venv
  ```
- [ ] **0.2.2** `pip install -r requirements.txt`; record the resolved versions to `requirements.lock`.
  **A/C:** `python -c "import tensorflow, duckdb, polars, shapely; print('ok')"` succeeds.
- [ ] **0.2.3** Create `C:\osm-data\` and confirm it is outside OneDrive.
  **A/C:** the path contains no `OneDrive` segment and OneDrive is not syncing it.
- [ ] **0.2.4** Add `MAPGUARD_DATA` env var pointing at `C:\osm-data\`; make every module read it rather than hard-coding `ROOT/data`.
  **A/C:** no module writes bulk data inside the repository.
- [ ] **0.2.5** Add `pytest` and a `tests/` directory with one trivial passing test, so the harness exists before it is needed.

### 0.3 Accounts and access

- [ ] **0.3.1** **Ayush:** create an OSM account (free, instant).
- [ ] **0.3.2** **Ayush:** log into OSMCha with it and copy the API token from the user page.
  **A/C:** `GET osmcha.org/api/v1/changesets/?harmful=True&page_size=1` returns 200, not 401.
- [ ] **0.3.3** Store the token in a `.env` that is gitignored. Never commit it.
- [ ] **0.3.4** Create Kaggle accounts for all four (30 GPU-h/week each — that is the real compute budget).
- [ ] **0.3.5** Create a shared Google Drive folder `mapguard/` for Parquet and checkpoints.
- [ ] **0.3.6** Create the Weights & Biases project and share it with all four.
- [ ] **0.3.7** Agree and document the User-Agent string used for every OSM request, including a contact address.

### 0.4 Kick-off decisions

- [ ] **0.4.1** Confirm the four workstream owners by name in `README.md`.
- [ ] **0.4.2** Confirm the official course calendar and replace the assumed dates in §13.
- [ ] **0.4.3** Read `docs/PLAN_REVIEW.md` as a team. Everyone must know why B1 now runs on Corpus A.

---

## Phase 1 — Gate 1: prove the plan before building it (week 1)

**Nothing in Phase 2 onward starts until 1a–1e are green.**

### Gate 1a — `adiff` at batch scale · WS1

- [ ] **1a.1** Write `scripts/gate1a_adiff_batch.py`: sample 100 random Corpus A changeset ids, fetch metadata, build the per-changeset `adiff` query, parse `<old>`/`<new>`.
- [ ] **1a.2** Handle the two known gotchas: `[bbox:…]` in the setting block does not parse with `adiff` (put the bbox on each statement); a global or missing bbox must fall back to versioned multi-fetch.
- [ ] **1a.3** Record per-changeset: HTTP status, wall-clock, action count, how many actions carry the target changeset id, whether ways came back with resolved coordinates.
- [ ] **1a.4** **A/C: ≥90 of 100 return usable before/after geometry.** Median wall-clock recorded. → `results/gate1a_adiff_batch.json`
- [ ] **1a.5** If below 90%: characterise the failures (empty bbox? relation-only? too large?) and decide the fallback split before proceeding.

### Gate 1b — Overpass capacity · WS1

- [ ] **1b.1** Run `adiff` queries continuously for 30 minutes against `overpass-api.de`, logging 200/429/504 and slot waits.
- [ ] **1b.2** Extrapolate: can 18,276 queries complete inside the P1 window at a polite rate?
- [ ] **1b.3** **A/C:** a written decision — full corpus, a mirror instance, or a documented geometry subset with the sampling rule stated. → `results/gate1b_overpass_capacity.json`
- [ ] **1b.4** If subsetting, ensure the subset is **not** correlated with the label (e.g. sample by changeset id, never by size or user).

### Gate 1c — label-leakage tripwire · WS2

- [ ] **1c.1** Fetch changeset **metadata only** for all 18,276 (fast, small): comment, `created_by`, account age at edit time.
- [ ] **1c.2** Train baseline **B0** — logistic regression on comment length + editor string + account age. Nothing else.
- [ ] **1c.3** **A/C: F1 < 0.70 under the user-disjoint split.** → `results/gate1c_b0_tripwire.json`
- [ ] **1c.4** If F1 ≥ 0.70: inspect which feature carries it. Revert-related text inside the changeset record would void every downstream number — find it now, not in the paper.

### Gate 1d — framework spike · WS3

- [ ] **1d.1** Build a toy Keras 3 model: 5 heterogeneous inputs (2-D raster, char sequence, edit set, user sequence, tabular), a ragged/padded set of edits with masking, masked multi-head attention pooling, one sigmoid output.
- [ ] **1d.2** Train one epoch on random data. Confirm masking actually masks (padded rows must not change the output).
- [ ] **1d.3** **A/C: it trains, masking is verified, and the code is legible.** → `results/gate1d_framework_spike.json`
- [ ] **1d.4** **Decision by day 5.** Pass → Keras 3 (Decision 2 confirmed). Fail → PyTorch, and update `requirements.txt`, §10 and §15.

### Gate 1e — Stage-1 throughput · WS4

- [ ] **1e.1** Build a stub edit encoder at the intended width (128-d output) and measure edits/sec on one Kaggle P100 at the intended batch size.
- [ ] **1e.2** Multiply out: how many edits fit inside 30 GPU-h/week? Set the pretraining corpus size from that number.
- [ ] **1e.3** **A/C:** a stated pretraining budget in edits and GPU-hours, derived not guessed. → `results/gate1e_throughput.json`

### Gate 1 review

- [ ] **1.R** All five gates green, all five JSON files committed. Team sign-off before Phase 2.

---

## Phase 2 — Corpus A data pipeline (weeks 2–4) · WS1

### 2.1 Contents fetch

- [ ] **2.1.1** Update `fetch_corpus_a.py` defaults against the **measured 1.2 s** latency, not the 0.2 s the old plan claimed.
- [ ] **2.1.2** Verify resume works: kill mid-run, restart, confirm no refetch and no gap.
- [ ] **2.1.3** Verify HTTP 509 backoff by forcing it, or by unit-testing the backoff path.
- [ ] **2.1.4** Run the full fetch for all 18,276.
  **A/C:** ≥98% HTTP 200; every non-200 recorded with its status. → `results/corpus_a_fetch_report.json`
- [ ] **2.1.5** Mirror the SQLite checkpoint to Drive; confirm the live DB is never on a FUSE mount.

### 2.2 Geometry enrichment (the fix from the plan review)

- [ ] **2.2.1** Write `src/mapguard/data/adiff.py`: per-changeset `adiff` → before/after geometry per edited object.
- [ ] **2.2.2** Write `src/mapguard/data/multifetch.py`: versioned multi-fetch (`?nodes=<id>v<ver>`, `?ways=<id>v<ver>`), batching up to 725 ids per request, as the fallback route.
- [ ] **2.2.3** Route selection: `adiff` primary; multi-fetch when the bbox is missing, global, or the query fails.
- [ ] **2.2.4** Store geometry alongside contents, keyed by (changeset, type, id).
- [ ] **2.2.5** Run enrichment across the corpus (or the Gate-1b-approved subset).
  **A/C:** geometry coverage rate reported per object type. → `results/corpus_a_geometry_coverage.json`
- [ ] **2.2.6** **Fix the false docstrings** in `osmchange.py` and `build_corpus_a.py` that say Corpus A geometry is node-only.

### 2.3 Parser hardening

- [ ] **2.3.1** Fix the memory leak in `osmchange.parse()`: `elem.clear()` empties children but leaves the husk attached to the root, so a 50,000-object changeset still accumulates 50,000 elements. Detach from the parent.
- [ ] **2.3.2** Unit-test the parser against: a create-only changeset, a mixed create/modify/delete, a deleted node, a way with `<nd>` refs, a relation with members, and a truncated document.
- [ ] **2.3.3** Confirm document order is preserved — B3 consumes commit order and must not be sorted.
- [ ] **2.3.4** Confirm the `num_changes` / `changes_count` attribute name difference between the changeset API and the replication stream is handled in both readers.

### 2.4 User histories — point-in-time

- [ ] **2.4.1** Decide the source: changeset replication stream for recent windows, `changesets-latest.osm.bz2` (**8.72 GB**, not 8.1) for the 2014–2019 era.
- [ ] **2.4.2** Stream-parse to a per-uid ordered changeset table. Constant memory — verify peak RSS on Colab.
- [ ] **2.4.3** **Implement the point-in-time rule:** every user feature and sequence uses only changesets with `created_at < target.created_at`.
- [ ] **2.4.4** Write `test_user_features_are_point_in_time` — construct a user with a known later event and assert it never appears in the features of an earlier target. **This test is not optional.**
- [ ] **2.4.5** Cap the user sequence length (e.g. last 200 changesets) and record the truncation rate.

### 2.5 Presumed-negative pool for R2

- [ ] **2.5.1** Sample ~50,000 changeset ids uniformly from the Ovid era, excluding all 18,276 labelled ids.
- [ ] **2.5.2** Fetch metadata plus contents (~4.2 h at 1.2 s and 4 workers).
- [ ] **2.5.3** Document the contamination assumption: true vandalism prevalence among them is well under 1%; state the bound in the paper.
- [ ] **2.5.4** **A/C:** a sampled 1:100 evaluation set exists and its construction is reproducible from a seed.

---

## Phase 3 — Feature store (weeks 3–5) · WS1

- [ ] **3.1** Extend `build_corpus_a.py` to emit three tables, not two: `changesets.parquet`, `edits.parquet`, `geometry.parquet`.
- [ ] **3.2** Implement the **exact** whole-changeset summary statistics over all N edits (op mix, type mix, tag-key histogram, geometry moments). These must never truncate — this is what makes the G4 claim honest.
- [ ] **3.3** Implement the **stratified edit sample**, K = 256: top displacement, top area change, all text-bearing edits, then random fill. Store `K/N` as a feature.
- [ ] **3.4** Geometry features per edit: node displacement (haversine), way length delta, polygon area delta, bbox diagonal delta, vertex count delta, Hausdorff distance where both sides exist.
- [ ] **3.5** Text features: extract text-key values (`name`, `name:*`, `description`, `note`, `operator`, …), old and new, as character sequences for B2.
- [ ] **3.6** Tabular features: a documented superset of Ovid's, so the reproduction is honest.
- [ ] **3.7** Build the DuckDB view layer over the Parquet files.
  **A/C:** a full-corpus aggregate query runs on the laptop with peak RSS under 1 GB.
- [ ] **3.8** Write a **mock Parquet generator** — 200 synthetic changesets with the identical schema — and publish it in week 1 so WS2/3/4 are not blocked on WS1.
- [ ] **3.9** Freeze and document the schema. Any later change bumps a version field.
- [ ] **3.10** Upload the featurised Parquet to Drive once.

---

## Phase 4 — Baselines and evaluation harness (weeks 3–7) · WS2

### 4.1 Splits

- [ ] **4.1.1** Implement **R1a**: random 70/10/20, seeded. Ovid's protocol. Label it user-leaky in every output.
- [ ] **4.1.2** Implement **R1b**: user-disjoint 70/10/20 — no uid crosses splits.
- [ ] **4.1.3** Assert disjointness in a test, and report the class balance of each split.
- [ ] **4.1.4** Persist split assignments to disk so every model sees identical splits.

### 4.2 Metrics

- [ ] **4.2.1** Implement F1, precision, recall, ROC-AUC, **PR-AUC**, precision@k.
- [ ] **4.2.2** Implement **prevalence-corrected PR curves**: reweight negatives by `w = ((1-p)/p) * (n_pos/n_neg)`.
- [ ] **4.2.3** Unit-test the correction against a synthetic set with known prevalence.
- [ ] **4.2.4** Implement calibration: reliability diagram plus Brier score plus Platt/isotonic recalibration.
- [ ] **4.2.5** Bootstrap confidence intervals on every headline number. A point estimate without a CI does not go in the paper.

### 4.3 Baselines

- [ ] **4.3.1** **B0 tripwire** (from Gate 1c), re-run on the final feature store.
- [ ] **4.3.2** Logistic regression on the full tabular feature set.
- [ ] **4.3.3** Random forest.
- [ ] **4.3.4** XGBoost, lightly tuned.
- [ ] **4.3.5** **Ovid reproduction** — Ovid's feature set only, under R1a.
  **A/C:** within a few F1 points of 81.66. If far off, say so and diagnose before claiming any improvement over it.
- [ ] **4.3.6** All baselines write to `results/baselines.json` with split, seed, metric and CI.

### 4.4 Label audit

- [ ] **4.4.1** Draw 100 labelled changesets at random; two reviewers read each in OSMCha independently.
- [ ] **4.4.2** Report the agreement rate and the estimated label noise rate. → `results/label_audit.json`
- [ ] **4.4.3** Write the limitations paragraph from the measured number, not from an assumption.

---

## Phase 5 — Stage 2 model (weeks 5–10) · WS3

- [ ] **5.1** **B5 tabular MLP** first — it is the parity baseline and the simplest end-to-end path.
- [ ] **5.2** **B2 char-CNN** over tag text, multi-width parallel kernels (the Inception idea).
- [ ] **5.3** **B3 edit-sequence BiLSTM** over the sampled edits in commit order.
- [ ] **5.4** **B4 user-history GRU** over the point-in-time user sequence.
- [ ] **5.5** **U5 deliverable:** train a vanilla RNN on 200-step user histories deliberately, plot gradient-norm collapse, then justify LSTM/GRU. This is a report section, not a detour.
- [ ] **5.6** **B1 geometry CNN** — 2-channel before/after raster per edit, plus scalar geometry features.
- [ ] **5.7** Decide and document the rasterisation: extent, resolution, normalisation, how a single node is rendered.
- [ ] **5.8** **Multi-head attention pooling** over the K sampled edits, with masking verified.
- [ ] **5.9** Fusion head: concatenate branch outputs → dense → dropout → LayerNorm → sigmoid.
- [ ] **5.10** Three outputs wired: `p(damaging)`, per-edit attention weights, branch attribution.
- [ ] **5.11** **Focal loss** implemented and compared against BCE.
- [ ] **5.12** Training loop with early stopping on **validation PR-AUC**, never accuracy.
- [ ] **5.13** Checkpoint to Drive every epoch; verify a killed run resumes.
- [ ] **5.14** W&B logging: config, metrics, split, seed, git SHA on every run.
- [ ] **5.15** **U3 deliverable:** optimiser comparison (SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam) on the real task, reusing the lab expt 6 protocol.
- [ ] **5.16** **U2 deliverable:** activation ablation (ReLU / LeakyReLU / GELU).
- [ ] **5.17** Parameter count per branch, tabulated — U4 asks for it explicitly.

---

## Phase 6 — Corpora B, C and D (weeks 6–11) · WS1 + WS2

### 6.1 Corpus B — modern bulk

- [ ] **6.1.1** Bulk-fetch `real-changesets` JSONs across 147M–164M (82% density, ~14M available).
- [ ] **6.1.2** Parse the `old` block: before/after tags and node coordinates come free here.
- [ ] **6.1.3** Featurise into the same frozen schema as Corpus A.

### 6.2 Labels for B and D

- [ ] **6.2.1** OSMCha route: pull `harmful=True` changesets over the target windows; store id, reviewer, timestamp.
- [ ] **6.2.2** Revert-mining route: harvest the changeset **replication stream** (~125 MB gz/month) and match revert comment patterns (`revert`, `reverting changeset`, `mass revert`, tool signatures).
- [ ] **6.2.3** Validate revert mining against the OSMCha set — what fraction of known-harmful changesets does it recover, and at what precision? → `results/label_route_agreement.json`
- [ ] **6.2.4** Build the final Corpus B/D label sets and document the prevalence in each.
- [ ] **6.2.5** If **both** routes fail: fall back to metadata-only drift analysis for R3 and say so plainly. Corpus B still serves pretraining.

### 6.3 Corpus C — pretraining set

- [ ] **6.3.1** Sample unlabelled changesets from the bucket window up to the Gate-1e budget.
- [ ] **6.3.2** **Enforce the disjointness rule:** exclude every changeset id **and every uid** appearing in a Corpus B or D evaluation split.
- [ ] **6.3.3** Write `test_pretrain_eval_disjoint` and run it before every pretraining job. → `results/corpus_c_disjointness.json`

### 6.4 Corpus D — live 2026

- [ ] **6.4.1** Write `src/mapguard/data/replication.py`: read `state.yaml`, walk sequence numbers, stream `.osm.gz` changeset files.
- [ ] **6.4.2** Harvest a defined 2026 window (state the exact dates; they go in the paper).
- [ ] **6.4.3** Sample, fetch contents per changeset from the API, enrich geometry with `adiff`.
- [ ] **6.4.4** Featurise into the frozen schema.
- [ ] **6.4.5** Verify the temporal boundary: **no Corpus D changeset may predate the Corpus A training cut-off.**

---

## Phase 7 — Stage 1 pretraining (weeks 8–12) · WS4

- [ ] **7.1** Edit encoder: tag embedding + char-CNN + geometry conv + dense → `z ∈ R^128`.
- [ ] **7.2** **Pretext A — masked tag modelling.** Mask one tag value, predict it from the rest.
- [ ] **7.3** **Pretext B — geometry reconstruction.** Autoencode the polyline/polygon; keep reconstruction error as a standalone anomaly score.
- [ ] **7.4** **Pretext C — corruption contrast.** Corruption generator: geometry displacement, name scrambling, ×100 area, chunk deletion, tag-value swap.
- [ ] **7.5** Document each corruption and why it mimics a real vandalism mode.
- [ ] **7.6** Combined loss `L = L_mask + λ₁·L_recon + λ₂·L_contrast`; λ values are an ablation, not a guess.
- [ ] **7.7** Pretrain to the Gate-1e budget on Kaggle. Checkpoint every epoch.
- [ ] **7.8** **Zero-shot corruption transfer check:** score the frozen Pretext-C head on the real labelled set before any fine-tuning. Report the AUC. → `results/pretext_c_zeroshot.json`
- [ ] **7.9** If that AUC is near 0.5, redesign the corruptions before Phase 8. Discovering this during the paper write-up is too late.
- [ ] **7.10** Freeze / partial-unfreeze / full fine-tune comparison — the U4 transfer-learning deliverable (lab expt 8 procedure, off-image).
- [ ] **7.11** Embedding sanity check: t-SNE or UMAP of `z`, coloured by object type and by label.

---

## Phase 8 — Evaluation (weeks 10–14) · WS2

- [ ] **8.1** **R1a** — balanced, Ovid protocol. Report alongside the 81.66 reference, flagged as user-leaky.
- [ ] **8.2** **R1b** — balanced, user-disjoint. Report the R1a–R1b gap explicitly; it is a finding.
- [ ] **8.3** **R2** — prevalence-corrected PR curves at 1% and 0.1%, plus the sampled 1:100 set. PR-AUC and precision@k only; no F1 headline.
- [ ] **8.4** **R3a** — train Corpus A, test Corpus B.
- [ ] **8.5** **R3b** — train Corpus A, test Corpus D (live 2026).
- [ ] **8.6** **R4** — label efficiency at 10/25/50/100%, pretrained vs scratch, ≥3 seeds each. **This is the single most important figure in the paper.**
- [ ] **8.7** **Per-vandalism-type breakdown** — geometric vs semantic vs destructive. Requires a type taxonomy applied to the test set; do the labelling work.
- [ ] **8.8** Calibration curves for the final model on every regime.
- [ ] **8.9** Threshold analysis: precision at the operating points a real review queue would use.
- [ ] **8.10** Error analysis: 20 false positives and 20 false negatives read by hand, grouped into causes.
- [ ] **8.11** Every number written to `results/`, keyed by config, split, seed and git SHA.

---

## Phase 9 — Ablation grid (weeks 11–14) · WS3 + WS4

- [ ] **9.1** B0 tripwire · **9.2** Ovid-repro · **9.3** +geometry · **9.4** +text · **9.5** +sequences · **9.6** Full-scratch · **9.7** Full-pretrained
- [ ] **9.8** λ ablation for the Stage-1 loss weights.
- [ ] **9.9** LSTM vs GRU vs BiLSTM for B3/B4.
- [ ] **9.10** Focal loss vs BCE.
- [ ] **9.11** Optimiser grid (U3) and activation grid (U2).
- [ ] **9.12** ≥3 seeds per config; report mean ± std. A single-seed ablation row is not evidence.
- [ ] **9.13** Batch overnight on Kaggle; track everything in W&B.
- [ ] **9.14** Export the grid to a single `results/ablations.json` and generate the paper table from it programmatically.

---

## Phase 10 — Demo and figures (weeks 12–14) · WS4

- [ ] **10.1** Folium map demo: score a live changeset, highlight the flagged edit, show the branch attribution.
- [ ] **10.2** Pull genuinely current changesets from the minute replication stream for the demo — the plan's freshness claim must be true on the day.
- [ ] **10.3** Figure — architecture diagram (from the existing mermaid, exported clean).
- [ ] **10.4** Figure — **R4 label-efficiency curve**, pretrained vs scratch. The headline plot.
- [ ] **10.5** Figure — PR curves at balanced and realistic prevalence on one axis.
- [ ] **10.6** Figure — gradient-norm collapse for the vanilla RNN (U5).
- [ ] **10.7** Figure — per-vandalism-type breakdown.
- [ ] **10.8** Figure — attention weights on a worked vandalism example.
- [ ] **10.9** Every figure regenerable by one script from `results/`. No hand-edited images.

---

## Phase 11 — Paper (weeks 12–15) · all four

- [ ] **11.1** Overleaf project, IEEE conference template, all four as authors.
- [ ] **11.2** Related work — from §2 and §3, with the gap table.
- [ ] **11.3** Method — Stages 1 and 2, the five branches, the hierarchical encoding rule, the point-in-time rule.
- [ ] **11.4** Data — the four corpora, with the `adiff` and replication routes described so others can reproduce them.
- [ ] **11.5** Evaluation — R1a/R1b/R2/R3a/R3b/R4, and why the protocol is a contribution.
- [ ] **11.6** Results — tables generated from `results/`, never retyped.
- [ ] **11.7** Limitations — measured label noise, presumed-negative contamination, relations out of scope, geometry subsetting if Gate 1b forced it.
- [ ] **11.8** Ethics and licensing section (§16): ODbL attribution, salted-hash uids, no account-level claims.
- [ ] **11.9** Reproducibility appendix: seeds, split files, git SHA, `requirements.lock`.
- [ ] **11.10** Plagiarism check.
- [ ] **11.11** Internal review — each member reviews a section they did not write.
- [ ] **11.12** Submit.

---

## Phase 12 — Course deliverables

- [ ] **12.1** Topic approval — end of week 3.
- [ ] **12.2** Phase-1 evaluation — week 7. Needs: Corpus A pipeline, B0 + Ovid reproduction, B2/B4/B5, R1 and R2 results.
- [ ] **12.3** Phase-2 evaluation — week 14/15. Needs: Stage 1, B1, Corpus B and D, R3 and R4, the ablation grid, the demo.
- [ ] **12.4** Syllabus coverage audit against §6 — every unit U2–U5 demonstrably used, with the artefact named.
- [ ] **12.5** Final repository tidy: README current, no data committed, `results/` complete, no stale docstrings.

---

## Standing rules — check at every merge

- [ ] No data in the repository. Bulk data lives in `C:\osm-data\`.
- [ ] No secrets in the repository. The OSMCha token stays in the gitignored `.env`.
- [ ] Metrics go to `results/*.json`, not to stdout and not to a screenshot.
- [ ] Every run records config, split, seed and git SHA.
- [ ] Notebooks stay thin: `git pull`, install, import, run.
- [ ] OSM requests: 1 req/s per worker, honest User-Agent, back off on 509.
- [ ] No `Co-Authored-By` trailers, no generated-by footers, no tool-named branches.
- [ ] A red gate stops the phase. Re-plan; do not build past it.
