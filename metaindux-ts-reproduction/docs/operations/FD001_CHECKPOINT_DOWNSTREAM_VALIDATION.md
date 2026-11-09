# FD001 checkpoint age: fixed downstream validation

- Scope: diagnostic/supporting, `scope-20261030-fd001-checkpoint-downstream-validation`.
- User authorization: compare epochs15/70 with five identical downstream seeds; epoch64 is an auxiliary comparison. Do not change networks or training parameters. Do not select individual generator epochs using official test results.
- Baseline: completed `diagnostic-fd001-seed23-checkpoint-age-20261030`, a single original-setting generator training seed23 rerun. Original historical late weights are unavailable.
- Single changed factor: generator checkpoint age (15 versus70, auxiliary64). Reuse the existing three full15931-window sampling103 datasets with identical conditions and audited initial/step noise. No new generator training, sampling, discriminator fits, or other generator seeds.

## Data and validation

[Code] Preserve the original FD001 window48/14 sensors/global training MinMax/capped125 target, original synthetic11151/4780 train/dev indices, and original predictor training entry point `upstream/eva_regressor.predictive_score_metrics`.

Before any fit, select30 of100 real training engines with `np.random.RandomState(20261030).permutation(sorted_ids)[:30]`, sort their IDs, freeze ALL their windows and window identities. This selection uses no predictions, no test files, and no error-based engine selection. Use identical validation windows and labels for all15 fits. Save engine IDs, end cycles, capped and uncapped RUL. Official test files, test predictions and test metrics are not used.

[Observed] The frozen generator and training MinMax have already seen all100 training engines, including these30. The validation is exploratory downstream evaluation; it is NOT an independent validation of the complete generation pipeline. Synthetic samples are RUL-conditioned independent windows, not attributed engine trajectories. Retaining the existing synthetic split does not create a whole-pipeline engine holdout. All windows from the selected engines are used, so the lifespan distribution differs from official last-window testing.

## Predictor controls

[Code] Same public LSTM (input14, internal LSTM hidden28, one layer, flattened48 steps, FC64 then1), MSE, Adam LR.002, batch256, original decay at zero-based epochs15/25/35, maximum35 epochs. The ORIGINAL best synthetic-dev RMSE checkpoint rule (epoch>5, evaluate every `len(train_loader)//5` batches) and10-epoch patience remain unchanged. Real validation is supplied to the original reporting-only `test_iter` argument; it NEVER determines optimizer updates, saved predictor checkpoints, early stopping, or training parameters. This argument name does not make it the official test set.

Seeds3/13/23/33/43 are paired across all three datasets. Keep original deterministic predictor settings. Audit identical initial tensors and consumed training-index order prefixes across checkpoint ages for each seed. Record actual training updates, epochs and selected checkpoint updates; actual lengths can differ under the SAME pre-existing synthetic-dev early-stop rule. Do not silently replace that rule with fixed-final models or force equal actual update counts.

No output clipping or ensembles for the primary metrics. Replay each saved predictor and calculate all metrics from saved predictions. Preserve synthetic train/dev predictions as fitting diagnostics.

## Metrics and interpretation fixed before results

- Primary comparison70 minus15 on fixed real validation. Primary aggregate: average per-engine MSE, average across five predictors, then square root. Also report mean individual predictor engine-balanced RMSE, conventional window-weighted RMSE, paired per-seed changes and per-engine changes. Never confuse these aggregates.
- Segments using capped real labels:0–30,31–60,61–90,91–125; additionally separate91–124 and125 to expose the cap. Save uncapped labels for context, but no uncapped-error metric substituted for the capped training target.
- A practical clear improvement flag requires at least5% reduction in pooled engine-balanced overall RMSE and at least4/5 paired seeds improving. This is an exploratory diagnostic criterion, not statistical significance. Always show all segment tradeoffs, including regressions, regardless of this flag.
- Epoch64 is auxiliary. Its results cannot be used to quietly redefine the predeclared primary70-versus15 contrast. No default checkpoint-selection rule is adopted by this single generator seed/sampling seed experiment.
- If downstream also improves, selection contributes to this rerun's utility difference; if not, improved DS/condition readout has not established downstream utility. Multiple generator training seeds would require a separately authorized next experiment; do not run them automatically.

## Audit and stop

Snapshot code/config/data/checkpoint hashes, source archive identity, Git state/diff, environment, seeds and data split before training. Validate shapes/finiteness/label identity/noise records and reconstruct raw-training windows against the earlier frozen tensor. Verify exact saved-initialization and paired consumed order prefixes, count optimizer updates, and replay metric identities. Existing source/defaults/checkpoints/strict reproduction results remain unchanged; numeric batch-zeroing repair remains paused and EXT-002 remains closed at weight1.

Stop after exactly15 predictor fits (main10, auxiliary5), any worker failure or audit failure. No automatic retries, new seed search, data regeneration, parameter changes, final-epoch adoption, or official test evaluation. Write a report, metrics, plot, run manifest, chronological ledger update and current state checkpoint. This is not a published paper experiment: current metric → paper metric unavailable → paper gap not applicable → only conditional five-predictor variation → strict reproduction conclusions unchanged.
