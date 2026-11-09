# Experiment2: seed13 epoch64/70 fixed condition readout

Diagnostic/supporting, `scope-20261102-fd001-seed13-condition-segments`. User now explicitly authorizes experiment2 after previously deferring it. Keep experiment1 frozen. Compare only the SAME seed13 rerun's best64 versus final70, using existing generation and readout artifacts from `diagnostic-fd001-final70-two-seeds-20261030`. No generator/predictor training, new generation, new checkpoint selection or model changes.

## Exact artifact identity

Verify original run input/output hashes for both generator checkpoints, both full15931-window sample archives, per-arm sampling records, condition_predictions.npz and condition_metrics.csv, the frozen reference index file, and the same five real-trained predictors (seeds3/13/23/33/43). Confirm selected generator epochs64/70, original state tensors' weight hashes, same15931 labels/order and dimensions, retain-condition sampling, explicit null CFG, DDPM1000/CFG3/batch256, sampling103. Verify paired initial and step-noise audit:63000 actual normal tensors/conditional uniform draws,0 dropped conditional windows, same pre-construction RNG and full noise start/end/schedule fingerprint. No new sampling at a different3000-window batch size.

Use the exact dev_matched3000 generated indices saved in BOTH arms. They must equal each other and the preceding experiment's common frozen seed23-indexed matched list (`FD001_predictions.npz`, position where generation_seeds==23). This common index list was used for both seed13 arms already; do not substitute an older seed13-specific list. Save exact3000 indices and labels before computing comparison summaries. Each of0–30/31–60/61–90/91–124/125 contains600 windows. Real dev indices are not generated indices; do not confuse the two.

[Observed] Reuse the existing five real-trained predictors' complete per-window readouts, hash-bound to the prior run, and recompute all statistics from their saved3000 outputs. This is an auditable retrospective analysis of existing observations, not new independent readout or new training. No additional inference is needed unless provenance/hash checks fail, in which case stop and report rather than silently replacing artifacts.

## Metrics

For each predictor and each bin, error=unclipped readout−capped generation condition. Save n,RMSE,signed mean error, and condition/readout means for both epochs. Positive error means the predictor reads a larger RUL, negative a smaller RUL; readouts are a diagnostic proxy rather than true generated-sample life. Condition125 is the capped125 category, grouping reference RUL>=125; generated windows have no independently observed uncapped true RUL.

Aggregate RMSE=sqrt(mean over5 predictors and600 windows of squared error), not RMSE of averaged predictions. Aggregate signed error=mean over the same5×600 errors. Also show mean predictor RMSE and all5 individual results so aggregation is unambiguous. Report absolute/relative RMSE change, signed shift and5-model direction counts. Do not add an arbitrary significance threshold or treat the five predictors as independent generator repetitions. Keep91–124 and125 separate. Optional91–125 combined summary is only context and must not hide the two directions.

Link the already-completed REAL-validation downstream contrasts for these same generator weights as context, keeping their different domain/metric explicit. These are original30-real-engine/4875-real-window engine-balanced results, not readouts of generated samples. No new downstream fits or official test evaluation.

## Interpretation and stop

If a bin's condition RMSE increases consistently across five fixed real predictors, this supports weakened compatibility of that generated condition with this diagnostic. If125 improves while91–124 worsens, report the mixed outcome, not uniform high-RUL deterioration. Improved condition readout cannot rule out coverage, conditional-distribution shape or downstream learning issues. Do not infer causality from covariance or treat this proxy as generated ground truth. Do not automatically run subsequent sensor/coverage interventions.

Budget: one cached two-arm comparison,50 per-predictor/bin metric rows,5 aggregate paired rows; no new models/samples or hyperparameter search. Stop on any identity/label/noise/metric audit failure, otherwise close with CSVs, all5 model tables, figure, report, frozen run/config/source/environment and updated state/ledger. Existing artifacts remain unchanged. Current diagnostic→no matching paper target→paper gap not applicable→five fixed predictor readouts conditional on one generator and sampling seed→strict reproduction conclusions unchanged. Numeric-zero repair remains paused, EXT-002weight1, main/A/B unchanged,C deferred.
