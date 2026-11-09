# FD003 anchor selection: frozen diagnostic protocol

Scope: diagnostic/supporting followup of extension EXT-001, not strict reproduction.
Authorization: user said “继续” after the recommendation to define selection on
training/validation engines. This run constructs and validates a selection rule;
it does not train a predictor or generator and does not access official test data.

## Fixed design, before execution

- Run ID: `diagnostic-fd003-anchor-selection-20261026`.
- Reuse FD003 w48 generation seed3, original predictor train/dev indices, and
  the previous 209 replacement positions. Keep their exact RUL histogram.
- Split the 100 real training engines using RandomState(20261026).permutation:
  first30 are validation, remaining70 selection. Never use validation features
  or distances to select donors or choose hyperparameters.
- Real donors: original predictor-train indices from the70 selection engines,
  RUL31–60 only. Neither official test records nor synthetic dev are candidates.
- Reconstruct the original14-sensor train-fit MinMax normalization. This scaler
  and the already trained generator used all100 training engines; this is NOT
  an independent holdout from generator training. Descriptor standardization
  is fitted on selection engines only and frozen before validation.
- Random arm: draw unique real donors for each exact RUL with
  RandomState(20261026); assign to the previous same-RUL replacement positions.
- Guided arm (one rule, no sweep): start from the synthetic train pool with all
  209 targets removed. Selection queries are real windows RUL31–60 from the70
  selection engines. At each of209 steps choose an unused donor giving the
  largest reduction in the mean5-NN LEVEL distance over selection queries,
  constrained by the remaining exact-RUL quotas. A donor can only serve queries
  within RUL±5 (also within31–60). When scoring a donor, exclude queries from
  that donor's own engine, preventing direct self/overlapping-window rewards.
  Previously selected same-engine donors are likewise excluded from the fitting
  objective. Equal query weights; aggregate validation results by engine.
  Resolve equal gains by ascending donor window index. No ratio/seed tuning.
- Freeze both plans and their SHA256 before reading validation descriptors.
- Validate on ALL RUL31–60 windows of the30 disjoint validation engines.
  Compare control, random and guided mixed pools using the same train candidate
  counts, same-RUL±5 and mean5-NN distances. Level is the primary representation;
  dynamics and unscaled cross-channel correlations are secondary checks.
- Also compute a real-donor-pool reference. This is a reference distance, not a
  lower bound: candidate counts and geometry differ.
- Primary gate (before results): guided must reduce engine-mean level distance
  by at least5% relative to random AND improve more than half the30 engines.
  Report both components, not only a pass/fail. Paired engine bootstrap interval
  is descriptive, 10000 samples, RandomState(20261026), no window-level inference.
- Stop after this one selection split and rule. If the gate fails, do not scan
  seeds/ratios or train a predictor automatically. If it passes, coverage alone
  still does not establish RMSE benefit; any next training must compare random
  versus guided under otherwise identical frozen settings.

## Limits and retained state

This is a data-selection diagnostic, not a missing-coverage-vs-signal-bias causal
test, nor validation of a paper result. No predictive RMSE is measured here.
Historical strict results stay frozen; zero-label repair remains paused.
