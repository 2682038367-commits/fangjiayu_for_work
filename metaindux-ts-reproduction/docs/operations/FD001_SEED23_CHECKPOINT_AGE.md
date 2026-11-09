# FD001 seed23原设置重跑与第15/64/70轮权重比较

Run `diagnostic-fd001-seed23-checkpoint-age-20261030`; diagnostic/supporting, expressly authorized by user. Question: could minimum training-loss selection have discarded better later generator weights? Search existing records first; if no reliable late weights, rerun ONLY original seed23 once for70epochs, adding checkpoint persistence, then verify epoch15 reproduction before late evaluation.

## Archive finding and interpretation limit

[Observed] Workspace search of .pth/.pt/.ckpt/.safetensors (excluding environment/Git) found754files at inspection. Only one original FD001w48seed23 generator checkpoint is attributable to full-data run: selected zero-basedepoch14/one-based15. Other seed23 few/zero-shot and w24 runs are different training tasks. The supplied `upstream/weights/DiffUnet_fre_FD001_48.pth` predates this training and has no matching seed23late provenance. No attributable epoch64/70 snapshots found. Save inventory and exclusions.

[Code] Original TrainCondition.train saves best-loss state_dict only, overwriting the same path; no final-state backup. Therefore the original run's later55epochs cannot be reconstructed from its loss CSV. Any new result concerns a same-setting rerun; even a functional epoch15 reproduction cannot prove what the missing historical later weights were.

## Fixed training, added observation only

Use original per-run CLI and parser defaults, FD001 full15931×48×14 and same capped125label order, seed23, UNetdim32/(1,2)/14, hard_random_quantile and seed-derived parameters, DDPM1000linear/pred_noise training,70epochs, batch256/shuffle, Adam.002, warmup_cosine/multiplier2.5, gradclip1. Keep training conditional dropout.2 AND original whole-batch numeric label zeroing.1. No optimizer/LR/duration/threshold/network/EMA/determinism changes. Preserve original torch_seed/cudnnbenchmark setting and local offline W&B logging.

Call original train function without modifying upstream files. Observer captures model construction and Adam.step; after actual updates945/4032/4410 save detached states as one-basedepochs15/64/70. Keep original best-loss saving separately under this run. Record per-parameter optimizer step counters, data permutation indices, original NumPy batch-label random draws, initial/final RNG fingerprints, and verify added snapshot serialization changes no TorchCPU/CUDA/NumPy RNG state. Observers do not add random draws. Complete exactly one70epoch training, not multiple attempts to recover old metrics.

## Epoch15 reproduction gate before later sampling

Historical anchor is the ORIGINAL selected epoch15 sampled with sampling103 and condition dropout disabled in the completed paired experiment. Reuse its frozen generated data, full actual noise trace fingerprints, fiveDS scores, real-predictor output arrays, labels/indices and source/hash provenance. Unlike the earlier untraced cached-control attempt, this anchor has a full legacy-reference noise/state audit. New sampling reuses the SAME retained-condition tracer/core and requires identical full start/end states and63000-draw schedule fingerprint to that anchor. Original/control metrics remain historical evidence, not silently re-evaluated or overwritten.

First evaluate only rerun epoch15: same sampling103/all15931 labels/DDPM1000/CFG3/batch256, conditional branch alwaysretain while consuming original mask draw, explicit null branch unchanged. Run same five discriminator seeds3/13/23/33/43, original80/20fixed split, and same five fixed real-trained predictor readouts, frozen generation-seed23testmatched100/devmatched3000 and full/boundaryRULbins.

Before new results, define practical functional epoch15 reproduction as all of: full data/noise/structure audit passes; meanDS absolute change from original retained-condition epoch15 <=0.03; batch failure category(meanDS>0.3) unchanged; devmatched3000 pooled inverseRMSE relative change <=10%. Also report exact tensor equality, relative tensorL2/maxabsolute differences, all per-seed score differences, original/rerun training losses, and LR/data-order evidence. Thresholds pragmatic diagnostic tolerances, not paper success criteria or statistical equivalence. No strict bitwise-training reproducibility claim when cudnn remains nondeterministic as original.

If gate fails, stop before epoch64/70 sampling/DS fits, report reproduction-limited/inconclusive. Training's saved64/70 weights remain available, but no new attempts or settings changes. If gate passes, evaluate64then70 with exactly the same103 sampling noise/labels/settings, fiveDS seeds and predictor readout. Max one70epoch rerun,3full generated batches/15newDSfits; original epoch15 anchor cached. No extra seeds, training, tuning or real downstream evaluation.

## Interpretation and stop

Primary generation-quality metric is mean5DS, lowbetter; fixed practical clear gain >=0.03reduction and >=4/5paired-evaluator decreases relative to rerun15. Report inverseRMSE/bias/proxy stage tradeoffs separately; not an oracle or downstream-utility claim. Report monotonic or nonmonotonic15→64→70, whether stillabove.3, and changes belowpracticalgate without claiming significance.

Later clear improvement within a validated rerun supports a risk that best-training-loss selection discards better generated distributions. Monotonic late improvement while stillpoor can motivate a separately authorized duration test; do not run it here. Persistently poor or worsening laterweights weakens this selected-epoch explanation and motivates initialstate/order/randomspectralparameter audits, not automatic interventions. Two late points cannot establish asymptotic training trends. Original historical late-model quality remains unobserved.

Bind configs/source/data/weights/anchor/proxy/inputhashes, environment/Git snapshots, actual70epoch/4410update observations, all per-worker logs and progress, reproduction decision BEFORE late evaluation, noise audits, predictions, score identities and independent recalculation. Stop on worker/audit failure or budget completion. Preserve all original checkpoints/results/defaults, strict scope, paused numeric-label-fix route and closedEXT-002(weight1).
