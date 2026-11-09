# FD001采样条件丢弃单因素诊断

Run `diagnostic-fd001-sampling-condition-dropout-20261030`; diagnostic/supporting, explicitly authorized by user. Existing seed13 best epoch63 (one-based64), seed23 best epoch14 (one-based15) remain fixed. No generator training, checkpoint substitution or downstream predictor training. This is distinct from training-time numeric whole-batch RUL zeroing, which remains paused.

## Frozen intervention and pairing

[Code] Only change the sampling conditional CFG branch: always retain the specified RUL embedding. Preserve explicit unconditional CFG branch (drop probability1), scale3, original DDPM1000/linear/pred_noise/clipping/normalization, batch256, all15931 original48×14 windows, label ordering, stored frequency parameters, strict state loading and model.eval(). Upstream sources/defaults stay unchanged; intervention is a worker-local override restored after sampling.

[Code] To preserve the exact legacy RNG schedule, the worker calls the original conditional mask generator with retain probability0.8, consumes the same uniform draws, then replaces its returned mask with all-true. The unconditional mask retain probability0 remains all-false without a random draw. This disables the effective conditional dropout without removing its RNG consumption.

[Observed] Controls are the existing six generated batches and thirty discriminator rows from `diagnostic-fd001-checkpoint-resampling-20261029`, each matched by checkpoint and sampling seed. Immutable data/row hashes are bound. Replaying the original first256-window control prefix must agree with its stored data within max absolute1e-6; record exact equality separately. If replay fails, stop before using that cached control, preserve the failure, and do not silently substitute another control or relax the threshold.

[Observed] During intervention sampling, verify every actual initial/per-step normal tensor against a second CUDA generator that follows the unmodified legacy normal/uniform operation order from the same starting RNG state. Verify actual random condition masks against that generator as well before forcing retain. Require before/after state equality and exact tensor equality for every draw, record counts, cumulative state fingerprints and per-batch progress. Default noise operations use contiguous float32 CUDA tensors. This explicitly checks RNG alignment rather than assuming identical seed implies identical noise. The earlier control did not store a full noise tape; provenance, prefix replay and reconstructed legacy RNG schedule are the link to cached controls, with that limitation stated.

## Screening and conditional confirmation

First sampling103 only: 2 fixed checkpoints × 2 arms =4 groups. Control data and5DS rows each are reused; generate2 intervention batches and run10 new original discriminator evaluations. Per-batch mean of eval3/13/23/33/43, same original discriminator and fixed80/20 split; failure remains strict meanDS>0.3.

Auxiliary condition readout uses the existing five real-trained FD001 predictors on BOTH control/intervention, no clipping/retraining. Save predictions for all windows and report original frozen `test_matched`100 and `dev_matched`3000 index/label sets from `real_predictor_condition_w48/FD001_predictions.npz` for each checkpoint's generation index. Pair identical indices within checkpoint/arm, retain all-window and low-RUL0/1/2–5/6–30/stage bins. Prior real-reference predictions are retained as context. No newly selected test cases and no new official-real-test inference. These are inverse-model proxies, not physical RUL truth or real downstream utility.

Before observing new data, define "clear change" at sampling103 for either checkpoint as either:

1. Absolute paired five-evaluator meanDS change >=0.03 AND at least4/5 paired changes have that sign; OR
2. Absolute pooled real-predictor inverse-RMSE change on frozen dev_matched3000 windows >=1 cycle AND relative magnitude >=10% AND at least4/5 paired predictor MSE changes have that sign.

Both improvements and deteriorations qualify. Thresholds are practical exploratory choices, not paper rules or significance tests. The dev_matched3000 set is the previously frozen stage-balanced generated index set, not an independently held-out generator population. Report test_matched100 and full-window readouts separately; the generator had seen all real training engines.

If either checkpoint satisfies either gate, automatically run BOTH checkpoints for sampling113 and123, using their existing controls and identical evaluation/proxy procedure. Otherwise stop after103. Maximum6 intervention batches,30 new discriminator fits; controls reuse6existing datasets/30rows. Prefix replay adds one256-window verification per executed cell, not an additional evaluated group. No new hyperparameters/seeds/model changes or further training after this budget.

## Outputs and acceptance

Save run/config/input source/data/checkpoint/proxy-model hashes, original control provenance, frozen input NPZ, all intervention samples, prefix replay evidence, per-draw noise/mask checks, five raw DS rows per cell, full proxy predictions, paired score/condition errors, trigger decision BEFORE confirmation and automatic final report. Fail closed on worker/audit mismatch; preserve infrastructure failures. Stop after the above diagnostic. Strict reproduction results, original outputs and training code remain frozen. No new samples or results are presented as completed before their worker records exist.
