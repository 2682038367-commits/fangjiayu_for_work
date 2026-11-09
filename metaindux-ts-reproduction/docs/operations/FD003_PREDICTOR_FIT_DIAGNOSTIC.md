# FD003 saved-predictor fit versus transfer diagnostic

Run ID: `diagnostic-fd003-predictor-fit-20261026`; scope: diagnostic/supporting.
User authorized continuing the proposed train/dev/real-test error check.
No training, parameter changes, selection sweeps or zero-label experiments.

## Frozen scope and comparisons

1. Primary: all25 existing full-data FD003 w48 predictors (5 generator seeds ×
   5 evaluator seeds). Use their original fixed70/30 train/dev indices and
   their own original generated dataset. Replay saved real-test predictions.
2. Real-trained reference:5 original FD003 predictors, evaluate original real
   train/dev windows and real test. These train/dev indices are the same as the
   synthetic experiments, but overlapping windows from the same real engine
   cross this random split. Real dev is NOT engine-independent validation.
3. Supporting followup:10 existing fixed35epoch local-anchor control/anchor10
   predictors. Reconstruct their actual training mixture from frozen209
   target/donor indices, with unchanged synthetic dev. Keep separate from the
   historical headline25. Do not retrain or evaluate the newly proposed guided
   anchor plans, which failed their coverage gate.
4. For synthetic-trained predictors also evaluate real training-source windows
   at the same train/dev indices as a transfer bridge. Those real windows were
   seen by the original generator. They were not direct predictor training data
   in the headline arm, but209 donor windows were direct training inputs in the
   anchor10 arm. Clearly distinguish this reference from official real test.

## Metrics and aggregation

- Bins:0–30,31–60 (primary),61–90,91–124,125 and all. True labels capped125;
  continuous predictor outputs remain unclipped. RUL125 merges raw>=125.
- Compute per-model MSE/RMSE/MAE/bias and exact-integer-RUL errors. Save raw
  per-sample predictions so every number can be reconstructed.
- Main displayed RMSE: sqrt(mean of per-model MSE), no prediction ensembling.
  Also save mean per-model RMSE and per-generation-seed summaries. Treat the
  5 evaluation seeds within a generator as dependent, not25 independent runs.
- Report native distributions and additionally reweight per-integer-RUL MSE
  to the real test RUL histogram inside each bin. Missing target labels produce
  an explicit undefined matched value, never silently omit label mass.
- Exact-RUL matching controls label composition, not operating/fault modes or
  within-label signal difficulty. Overlapping train/dev windows can make dev
  optimistic. Do not interpret native train/test gaps as causal evidence alone.

## Checks and stopping

- Hash source, data, splits, all40 weights, original result manifests/predictions
  and protocol before execution. Use CPU inference in eval/inference mode.
- Saved real-test prediction replay must agree within1e-3 for every model;
  reject mismatched weights/splits instead of accepting new values.
- Verify exact labels, original count/shape, original train/dev disjointness,
  unchanged dev and frozen donor/target reconstruction. Store sample and engine
  counts and aggregate from predictions with no training-loop loss estimates.
- Stop after this one predefined40-checkpoint audit. No automatic additional
  generator runs, architectures, epochs, anchor ratios or checkpoint selection.

## Interpretation, not a causal classification rule

- Small synthetic train AND dev error but large real-reference/test error
  favors transfer mismatch over a simple failure to fit generated inputs.
- High synthetic train error warrants checking fitting limitations, ambiguous
  generated signal/labels and stage weighting; it does not by itself prove
  optimizer failure. Low train/high dev also admits overfit/selection effects.
- Existing real predictors are an empirical comparison, not a known Bayes-error
  bound or proof of trainability for the synthetic distribution.
- No paper has corresponding diagnostic targets. Strict results and the
  paused zero-label route remain unchanged. Further intervention needs a new
  controlled design; this diagnostic alone cannot uniquely separate support,
  bias and optimization.
