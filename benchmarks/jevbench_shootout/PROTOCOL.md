# JevBench Shootout Protocol (LOCKED before any test evaluation)

Goal: honestly determine whether an anydecision entrant beats real `wfzyx/von`
(von-sdk 1.3.x, native weights) on JevBench public items. No hardcoded wins;
whatever the numbers say is reported, including a von victory.

## Data
- Source: `fstandhartinger/jevbench` `datasets/public/{easy,original,hard}.jsonl`
  (231 items, MIT harness + original items; third-party items keep their licence,
  used for evaluation only, never redistributed).
- Split: stratified by file, seed 0 → train 60% / dev 20% / test 20%
  (`splits.json`, id lists). Test ids are locked: no training decisions,
  no early stopping, no prompt tuning may use them. Single final scoring run.

## Entrants (identical items, identical label sets)
- **von-1.3**: native `von.decide` / `von.judge` / `von.rate`, default chains,
  `von-sdk==1.3.1`. Frozen weights; never trained on JevBench (per its model card).
- **ours-lm** (zero-shot reference): anydecision L0 + `Qwen/Qwen2.5-0.5B-Instruct`
  via TransformersBackend, fixed prompt template (recorded in artifact).
- **ours-head** (trained): frozen `answerdotai/ModernBERT-large` embeddings
  (premise + option texts) → `NonAutoregressiveDecisionHead` (bilinear,
  CE + Brier objective, AdamW, early stop on dev ECE/acc). Trains on train
  split ONLY. Options for choice items: label + criteria description.

## Item mapping
- choice: von `decide(state, choices={label: description}, instructions)`;
  ours `Question.choice("{instructions}\n{state}\nOptions:\n{label} - {desc}...")`.
- noul: von `judge(state, instructions)`; ours binary yes/no mapped to
  true/false.
- score: von `rate(state, criteria=[ordered levels])`; ours ordinal ranks
  0..K-1 over the label set in criteria order.

## Metrics (per entrant, overall + per tier file)
- accuracy (argmax over exact label set), majority-class floor per tier
- ECE top-label, 10 equal-width bins; multi-class Brier over label set
- mean latency ms/decision (caller wall time, serial)
- McNemar exact two-sided p on paired correct/incorrect outcomes for every
  head-to-head delta + MDE (smallest accuracy delta significant at α=0.05
  given observed discordant count). p ≥ 0.05 → UNRESOLVABLE, never a win.

## Decision rule
- ours-head vs von on locked test accuracy: significant (p < 0.05) and
  positive → "beats von on <tier> (Δ=…, p=…)". Else UNRESOLVABLE or von wins.
- Artifacts: `results/<entrant>_<split>.json` with model revisions, seed,
  per-item outcomes, latencies, commit hash. Nothing typed by hand.

## Anti-cheating
- No test-label training, no test-driven prompt/model selection.
- No von-output distillation into our head.
- Calibration (if any) fit on train only.
