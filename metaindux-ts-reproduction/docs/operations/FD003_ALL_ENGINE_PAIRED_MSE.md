# FD003 all100-engine paired MSE audit

Run `diagnostic-fd003-all-engine-pairs-20261027`; scope diagnostic/supporting.
User explicitly asks for ALL100paired results BEFORE deciding model changes.
No inference, retraining, architecture, weighting or data-selection changes.

## Frozen definition

For every real test engine i, use its SAME saved last48-step input and SAME
cappedRUL y_i in both model families. The original loader repeats the initial
record for padding if fewer than48actual cycles exist; record padding metadata.

Primary: delta_i = mean_{5generationseeds x5evaluationseeds}
(synthetic_prediction_i - y_i)^2 - mean_{5evaluationseeds}
(real_prediction_i - y_i)^2. Positive means synthetic-trained is worse.
This is a difference of mean squared errors, in cycles², NOT the square of
mean prediction, NOT a difference of ensemble errors, NOT a trajectory RMSE.

Also compute5vs5 per generationseed and evaluator-seed paired differences:
delta_ige=(synthetic_pred_ige-y_i)^2-(real_pred_ie-y_i)^2.
Their mean over g,e must equal primary delta_i. Shared real models are not
25independent real fits. Count sign stability over5generatorseeds and separately
over5evaluationseeds, without treating2500rows as2500independent engines.

## Required outputs and checks

- Complete100engine table in ID order: RUL/rawRUL/windowmetadata, both model
  counts, both MSE, both per-engine RMSerrors, delta, seed-sign stability.
- Keep2500paired rows and500generationseed summaries for reconstruction.
- Check exactly IDs1–100 and full seed Cartesianproducts, unique records,
  identical labels against original truth file, finite continuous outputs,
  source hashes and original30model test-RMSE replay tables.
- Summarize mean/median delta,25/75quantiles, positive/negative/zero enginecounts,
  pooled RMSEs, and5generationseed sensitivities.
- Pairwiseenginebootstrap:20000resamples of100engine indices,
  RandomState(20261027), fixed model families; percentile95%CI for mean and
  median delta. This is conditional on fitted models and empirical engines,
  not full algorithm/generation-seed uncertainty.
- Predetermined five RULbins0–30,31–60,61–90,91–124,125; report counts/signs,
  mean delta and contribution sum(delta)/100 to overallmean. Reconcile with
  existing segmented-error source; do not tune boundaries.
- Concentration: top1/5/10positive deltas as fractions of grosspositive and
  nettotal delta (net shares can exceed100% due to negative cancellation).
  Describe remaining100-kengines and leave75out sensitivities; these are
  changed subsets for explanation, never proposed deletions from evaluation.
- Save full signed engineplot and empirical paired-distribution/concentration
  visualization; do not only show handpicked75or worst cases.

## Decision and stop

No decision rule authorizes model edits from this test-set audit. Results can
rank hypotheses and motivate a controlled experiment, not identify causal
coverage/bias/optimizer mechanisms. Never use testdelta to define training
weights or remove hard testengines. Stop after all100results/report. Existing
strictmain/A/B claims unchanged,Cdeferred,zero-label route paused; previous
prototype/anchor/lossweight proposals remain unexecuted.
