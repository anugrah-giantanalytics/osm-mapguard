# MapGuard v2 — why round 1 plateaued, and what changed

Round 1 (M4, 2026-09-21, `experiments/rounds/r1_2026-09-21_m4/`) trained the five-branch
network and it **tied a plain XGBoost** (ovid PR-AUC 0.844 vs 0.857, F1 78.1 both), below Ovid's
published 81.66. Before changing the model we asked *why*. Every number below was measured on
2026-09-22 with cheap CPU probes, not guessed.

## 1. What is vandalism, as data?

A changeset is damaging because of three kinds of evidence:

| Kind | Question | Example signals |
|---|---|---|
| **WHO** | is this someone who knows the map? | first edits, no history, web editor |
| **WHAT** | what did the edit *do*? | added a fake park, deleted a landmark, turned a track into a motorway |
| **TEXT** | what did they write? | abuse in a name, junk comment, promotional text |

A fourth, **WHERE** (famous places, far from the author's usual area), is not in the data yet.

Probe: one model per family, Corpus A, counted one changeset per author (see §3):

| Family | ovid | T1 (human-reviewed) |
|---|---|---|
| WHO only | 0.603 | 0.673 |
| WHAT as round-1 *counts* only | 0.610 | 0.813 |
| TEXT only (char n-grams) | 0.626 | 0.787 |

No single family is enough; they are complementary. Round 1 was weakest on WHAT: it saw
"3 tags added", never *which* tag.

## 2. The missing information: what each edit did

Turning every edit into tokens — `+leisure=park`, `-historic`, `~highway:track>motorway`,
`+name` — and letting a linear model read them, the most damaging tokens were:

`+leisure=park`, `+highway=footway`, `+tourism=artwork`, `+highway=motorway`,
`+highway=primary`, `-historic`

That is the documented **Pokémon-Go vandalism** (fake parks and paths to create game
spawns; Juhász et al. 2020, cited by Ovid) and classic fake-road / landmark-deletion
vandalism. The most benign tokens were `+source`, `+access=private`, `+fixme`: the habits of
careful mappers. **This is the "common trait" the network should learn**, and it was not in
its input.

Adding those tokens and the text to the tree model:

| ovid | PR-AUC | one per author | F1 | Acc |
|---|---|---|---|---|
| round-1 deep model | 0.844 | — | 78.1 | 76.3 |
| XGBoost on counts | 0.858 | 0.764 | 78.1 | 77.0 |
| + edit tokens + text | 0.892 | 0.815 | 80.6 | 79.7 |
| **+ author-balanced training** | **0.898** | **0.836** | **81.1** | **81.3** |
| Ovid (published) | — | — | 81.66 | 81.37 |

The gain came from *information*, not from a deeper network.

## 3. The benchmark flatters everyone: author concentration

In Ovid's labels, 56 accounts supply 37% of the positives (bulk-reverted accounts, e.g. one
university mapathon with 206 changesets). A model that learns one such account's style
collects hundreds of easy points. XGBoost: **0.858 normally, 0.764 counting one changeset
per author**. Every result now also reports:

- `one_per_author_pr_auc` — the first changeset of each (author, class);
- `author_weighted_pr_auc` — each changeset weighted 1 / (changesets by its author).

Training uses author-balanced weights, 1/√n per author, so a bulk-reverted account counts
like ~14 changesets instead of 200.

## 4. The architecture fix: a changeset is a bag of edits

One bad edit among 200 good ones makes a changeset bad. That is **multiple-instance
learning**. Round 1 *averaged* the edit vectors, which dilutes that one edit 200×, and its
attention weights came out uniform (~1/n): it was not looking at any edit in particular.

v2 pools with **gated attention** (Ilse, Tomczak & Welling 2018) *plus* a **masked max**, so
one strong edit survives pooling. The attention weights become the "which edit" explanation.

## 5. v2 model

```
per edit:  numbers ──► edit_encoder ─┐                      (Stage 1 weights still load here)
           tokens  ──► embed ► mean ─┼─► edit_fuse ─► [B3 BiLSTM] ─► self-attention
           old>new text ► char-CNN ──┘        (B6 "edit meaning", new)       │
                                                            gated-attention MIL + max
changeset: comment + tag text ► char-CNN (B2) ─┐                         │
           user history ► GRU (B4) ────────────┼──────── fusion ◄─────────┘
           tabular ► MLP (B5) ─────────────────┘            │
                                                        p(damaging)
```

Training: author-balanced weights, AdamW (weight decay 1e-4), dropout 0.3. The round-1 model
peaked at epoch 5 and then memorised (train 0.97 vs validation 0.86).

## 6. The final system: deep + trees (stage 07)

Moderation systems at scale have converged on a common design, including Wikipedia's ORES and
the Wikidata vandalism detector (WDVD, which Ovid compares against). Gradient-boosted trees do
the structured part: reputation, counts, rules like OSMCha's "new mapper" and "mass deletion".
Neural encoders do only the unstructured part: text, and here edit semantics. We do the same:

- **Hybrid** (stage 01): XGBoost on tabular features + out-of-fold edit-token and text model
  scores, author-balanced. CPU only.
- **Ensemble** (stage 07): a blend of the deep model and the hybrid, with the weight chosen on
  validation only.

## 7. What each ablation row now proves

| Row | Question |
|---|---|
| base_edits_tabular / plus_geometry / plus_text / plus_sequences | same as round 1 |
| **plus_edit_meaning** | does the WHAT branch help on its own? |
| **full** | the v2 headline |
| **full_meanpool** | does MIL pooling beat averaging? |
| **full_unweighted** | does author weighting help the one-per-author score? |
| full_pretrained | does Stage 1 help? (round 1: no) |

## 8. Order of work, and why supervised first

Self-supervision cannot fix missing information. Stage 1's pretext is "spot synthetic damage
in edit numbers", and round 1 showed it transfers weakly (zero-shot 0.61) and adds nothing
(±0.003). If Stage 1 is tried again, it should pretrain on the new edit tokens — for example
masked-token prediction over millions of unlabelled edits — and only after the supervised
model is right.

## 9. Honest limits

- Revert labels are noisy: a revert is not always vandalism.
- WHERE is still missing. Adding it needs author locations from history, which means a refetch.
- The hybrid already matches Ovid's F1 on Ovid's protocol. The one-per-author number is the
  one to beat, and the one to lead with.
