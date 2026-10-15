# Error analysis — best ovid model (stack + embeddings, seed-averaged), 2026-10-15

Source: `error_analysis_ovid_stack_emb.csv` (same folder): the 30 most confident false positives
(label clean, highest score) and the 30 most confident false negatives (label reverted, lowest
score) on the R1b test set (3,607 changesets). Threshold chosen on validation: 0.254, giving
382 false positives and 258 false negatives in total. Fields come from the feature store;
`user_n_prior` is capped at 100 (the history fetch keeps the last 100 changesets). Every
statement below is a count over these 60 rows.

## False positives: clean changesets scored as damaging

- **All 30 come from different authors**, so this is not one account dominating the errors.
  29 of 30 were made in iD, the beginner web editor.
- **Mostly newer accounts:** the median is 11 prior changesets; 11 of the 30 had at most 2.
- **What they add** is exactly the pattern the model learned for vandalism: new parks
  (`+leisure=park`: "Park", "nombrando los parques…", "Contribución de parques y lagos"),
  footpaths and cycle paths (`+highway=footway/path/cycleway`: "Added missing footpath",
  "Foot path", "Cycle path", "Пешеходная дорожка"), and new POIs.
- **Reading:** genuine newcomers adding small local features look like Pokémon-Go-style
  vandalism (fake parks and paths) on every signal the model has: a new account, iD, created
  parks and footpaths. Telling them apart needs ground truth the data lacks (imagery, local
  knowledge).
- **At least two look like unreverted vandalism, i.e. label noise:** comment "pokemonㅡ"
  (2 edits, 0 prior changesets, adds a playground) and comment "rjgumz" (adds a name).

## False negatives: reverted changesets scored as clean

- **Experienced mappers:** median prior changesets ≥ 100 (the cap); 19 of 30 in JOSM, the
  expert editor.
- **Concentrated:** 14 of 30 come from authors with at least 5 changesets in the test set.
  One mapathon account accounts for 9 rows (`#hotosm-project-1318 … #Map4Tg`, 5 building
  edits each).
- **What was reverted:**
  - organised mapping (Humanitarian OSM Team tasking projects, `#hotosm-…`, hurricane Irma);
  - bulk building imports ("Add buildings", 533 and 669 edits);
  - routine technical edits ("Moved source to relation", maxspeed fixes, "Removed tag with
    typo", bus-stop tagging);
  - one comment is itself a revert ("revert von voreiligem …"), so this is an edit war.
- **Reading:** these are reverts for quality, import or dispute reasons, not vandalism. They
  are the "a revert is not always vandalism" noise in Ovid's labels, made by exactly the
  experienced, organised mappers the model has learned to trust.

## What this means

1. **The model's errors line up with the label definition, not with a blind spot in the
   features.** False negatives are reverted good-faith work. False positives are newcomer
   edits shaped like known vandalism, plus some likely unreverted vandalism.
2. **This supports the ceiling seen in rounds 2–4** (hybrid ~0.90 whatever cheap signal was
   added). Part of the remaining error is label noise that no model can remove.
3. **For a moderation tool, the false positives are the useful kind.** A reviewer would
   check them, and a newcomer adding parks deserves a look. The false negatives are mostly
   cases a vandalism tool *should* ignore.
