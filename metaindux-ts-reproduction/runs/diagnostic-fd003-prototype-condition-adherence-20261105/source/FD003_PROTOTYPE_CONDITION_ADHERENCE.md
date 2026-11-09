# FD003 saved-generation prototype adherence diagnosis

Diagnostic/supporting, user-authorized low-cost closeout. Scope scope-20261105-fd003-prototype-condition-adherence-v1. No models trained, no new generated windows, no refitting standardizers/centers, no K search and no official test access. All4pre-existing B cells(seed23/13,epoch64/70), plus paired originalA samples as a negative control, are read only. The previous ridge result is not grounds for restarting diffusion.

## Provenance and fixed assignment

Verify source/config/data/samples against parent and real-diagnostic manifests. B samples contain data/label/prototype_id; sampler code sets model.prototype_ids once per batch and stores actual ID vector. Replay only the independent ID-draw RNG from stored train frequencies and target label list (no diffusion/model call) to verify the stored vector. Check all candidates share label list/ID vector and A/B share sampling RNG audit hashes. A samples have -1 IDs: for this diagnosis use B IDs as explicitly PSEUDO labels on the same sample indices; A never received them. No fake generated-engine identity.

Recompute42statistics on saved normalized windows using exact frozen features() AST, transform with train feature_mean/scale, nearest4centers. No further normalization/clipping/fitting. Check trainIDs reproduced. Frozen classifier may extrapolate on synthetic data: report nearest-center distance quantiles and fraction above training99thpercentile alongside realvalidation reference. This is an applicability diagnostic, not a newly tuned rejection rule.

## Control RUL and class imbalance

Report full4×4requested-to-readout counts and row-normalized frequencies by all fixed RUL ranges, including31–40/41–50/51–60 and capped125 separately. Primary31–60. 125 matches only capped label, not uncapped true RUL.

At each exact targetRUL r, let p_r(k) be requested frequency, q_r(k) output frequency and n_r count. Raw accuracy alone is not adherence evidence. Conditional no-association chance=sum_r(n_r/N)sum_k p_r(k)q_r(k); conditional kappa=(accuracy−chance)/(1−chance). Also report always-requested-majority baseline perRUL and equal-RUL macro recall across requested classes present. Balanced chance averages q_r over those requested classes. Per-class recall is saved. Conditional mutual information=sum_r(n_r/N)I(request;output|r) detects response even if classes systematically mis-map; compare against1000within-exactRUL shuffles, preserving both marginals. Accuracy and CMI permutation p-values/descriptive null bounds for overall and31–60 only. No threshold/K/checkpoint selection. Same permutation realization across cells permits paired context; four cells share requested IDs/sampling seeds and are not4independent experimental replicates. Many windows are not independent generator seeds.

A pseudo-label conditional metrics provide a zero-prototype-input control for finite noise/RUL association. Evidence of changed B distribution relativeA alone does NOT establish effective prototype conditioning, since weights/training changed. CMI>shuffle indicates association in savedB batch; expected class correspondence and real shape retention assessed separately.

## Synchrony preservation

Reuse prior shape_features(): quarter sensor means→3quarter increments→eachsensor argmax absolute change→maximum fraction of14sensors whose maximum lies in same transition. Preselected real-positive contrasts P2−P1 andP3−P1. Reuse prior aligned real train/validation commonRUL sets. On eachgeneratedcell use requested IDs (not reassigned IDs) as groups, retain exactRUL strata with>=3samples per requested group, compute mean differences perRUL then equal-average. Recompute realreference on same remainingRUL support if anystrata excluded; no labels inferred from features to define requested group. Save actualRUL distributions and groupcounts; report generated effect, realvalidation effect, signed effect ratio, pairedA pseudo effect and B−A effect.

2000bootstraps resample generated window indices within exactRUL/requestedclass, using same row draws for A/B. Intervals describe fixed saved samples/noise only, not engine or training-seed uncertainty. No assigning realtrain engineIDs to generated rows. Realreferences retain previous engine-level limits; same-direction and effect magnitude reported without a posthoc beneficial threshold. Per-window descriptors may correlate with clustering inputs and do not identify known faults/causal degradation.

## Artifacts and stopping

Configuration configs/diagnostics/fd003_prototype_condition_adherence.json. Script analysis/diagnostics/diagnose_fd003_prototype_condition_adherence.py. Snapshot runs/diagnostic-fd003-prototype-condition-adherence-20261105; outputs results/diagnostics/fd003_prototype_condition_adherence_20261105. Record source/input/output hashes/environment/exact command and synthetic collapsed-class sanity checks (80%rawaccuracy must have0kappa/CMI). Report all4cells and both syncpairs, no new model fitting or downstreamevaluation. Stop once computed/audited; do not reopen generator training on ridge or adherence results alone.

From workspace root: `metaindux-ts-reproduction/.venv/bin/python analysis_path` where analysis_path is `metaindux-ts-reproduction/analysis/diagnostics/diagnose_fd003_prototype_condition_adherence.py`. Use --suffix replay1 for same scientific calculation under fresh output paths, preserving original manifest.
