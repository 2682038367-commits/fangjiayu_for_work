# Seed13 real-validation high-RUL direction, cached records only

Diagnostic/supporting `scope-20261102-fd001-seed13-real-high-rul-direction`. User authorizes ONE existing-record check: same seed13 generator rerun epoch64 versus70, five synthetic-trained predictors per arm, same real validation windows, split91–125 into91–124 and125, measure signed error and MSE. No training, generation, inference, network changes, new windows or other seeds/checkpoints.

## Provenance and estimands

Use10 `real_validation` predictions in `fd001_final70_two_seeds_20261030/seed13/{best,final70}/predictions`, paired predictor seeds3/13/23/33/43, and its exact `real_validation.npz`:4875 windows/30 original real-training engines, capped125 truth plus engine/end-cycle/window identities. Verify against the completed source run's bound hashes and audit records; do not substitute real-predictor readouts of generated windows from experiment2. No clipping or prediction ensemble.

For each RUL bin b, use engines E_b with at least one window in that bin. Let n_ib be the fixed count in engine i/bin b. For predictor j, define MSE_jib=mean over those SAME engine/bin windows of (prediction−truth)^2, bias_jib=mean(prediction−truth). Report:

- MSE_b=mean over5 predictors and E_b engines of MSE_jib.
- RMSE_b=sqrt(MSE_b), not mean individual RMSE and not RMSE of mean predictions.
- signed_b=mean over5 predictors and E_b engines of bias_jib, NOT the old window-weighted bias field.
- change=MSE70_b−MSE64_b; also expose all5 predictor-specific metrics and direction counts, eligible engine/window counts and per-engine paired results.

125 is the capped-label group; errors are against125, not uncapped life. Negative signed error indicates underprediction of the capped truth. Validation engines were seen by generator/MinMax; this remains downstream exploration, not whole-pipeline independent validation.

## Additive MSE attribution

Bin metrics above re-equalize eligible engines within each bin, so their MSEs/changes cannot simply be summed, nor weighted only by global window fractions. For parent P (91–125 or ALL4875), define parent per-window weight w_iP=1/(|E_P| n_iP), zero outsideP. The contribution of child bin b to parent MSE change is:

`C_b(P) = mean_j sum_{windows in b} w_iP * [(pred70_j−truth)^2−(pred64_j−truth)^2]`.

This preserves each parent's original engine equalization and each engine's actual within-parent bin fractions. For high parent, children91–124/125 must exactly sum to change of parent high MSE. For ALL, children0–30/31–60/61–90/91–124/125 must exactly sum to overall MSE change. Report high-parent and ALL contributions separately. Negative contributions are improvements; a positive bin may exceed100% of net overall degradation when other bins offset it. This is arithmetic attribution, not causal attribution.

## Interpretation and stop

Report whether91–124 MSE increases AND its engine-weighted signed error shifts more negative, and whether all5 paired predictors agree. That combination is consistent with (not proof of) the hypothesis that generated signals read younger at given conditions, making synthetic training labels relatively low and encouraging lower predictions on real windows. A non-negative signed shift weakens this simple directional explanation. Report125 contribution independently: its generated-condition readout improved in experiment2, but real prediction may still degrade. Do not infer coverage versus bias versus learning uniquely from these aggregate checks or treat signed average as every-window uniform shift.

Freeze source/config/validation/prediction hashes before analysis. Independently verify source per-model/per-engine MSE and recompute aggregate means, weights, all contributions and closure. Save requested two-row table,5-predictor results, per-engine metrics, per-window weights and contributions, report and manifest. Stop after one cached check or any audit failure; no automatic sensor/coverage/learning intervention. Budget0 new models/samples/inference, no selection/default changes. Current diagnostic→no matching paper metric→paper gap not applicable→5 fixed predictor seeds conditional on this generator/sample→strict reproduction status unchanged. Numeric-zero repair paused, EXT-002weight1, main/A/B unchanged,C deferred.
