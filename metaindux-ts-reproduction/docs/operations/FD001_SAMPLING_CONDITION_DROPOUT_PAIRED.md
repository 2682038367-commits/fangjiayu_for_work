# FD001采样条件丢弃：重新采样两组并逐步核对噪声

Run `diagnostic-fd001-sampling-condition-dropout-paired-20261030`, diagnostic/supporting. User explicitly requests four fixed-checkpoint/condition-dropout groups and corresponding noise alignment, initially103 then113/123if clear change.

[Observed] Previous cached-control approach `diagnostic-fd001-sampling-condition-dropout-20261030` stopped before any full new batch or discriminator evaluation. First256-window old-control replay maxabs0.000109851 exceeded its frozen1e-6 tolerance. Failure is preserved; tolerance is not relaxed, cached control is not substituted into the formal paired experiment. Cause of old-data numerical replay discrepancy is not uniquely established. User was informed before fresh paired execution. Current plan generates BOTH legacy control and retained-condition intervention anew; old scores/data are only historical context.

## Fixed factor and controls

Use the same seed13 best epoch63/one-based64 and seed23 best epoch14/one-based15 weights. Change only effective condition dropout during sampling: control original probability0.2, intervention0. Original conditional uniform random draw remains consumed before forcing intervention mask all-true. Explicit unconditional CFG drop1 remains unchanged. No retraining, different weights, final-epoch substitution or global upstream modification.

Both arms use original TrainCondition.sample/DDPM1000/linear/pred_noise/CFG3/clipping/normalization/48×14/all15931labels/order/batch256/stored spectral parameters/evalmode and original torch_seed initialization. Runtime-only mask override is restored when worker exits. Controls and intervention use the SAME trace instrumentation; control returns the actual original mask, intervention forces retention.

Require every actual initial/step normal tensor and every conditional uniform random mask to exactly match an independent CUDA generator following the legacy normal/uniform call order; before/after states must match. Require63000 normal draws,63000 conditional calls and63000 explicit-null calls per batch; record normal scalar counts, starting/ending state SHA and full schedule fingerprint. Within each fixed-checkpoint/sampling-seed pair, compare all these fingerprints and states between freshly generated arms. Save full outputs and real condition-retention counts. There is no pretense of a saved old-control noise tape: the fresh pair itself is traced on both sides.

## Evaluation and screen

103 first:2checkpoints×2fresharms=4fullbatches,each5originaldiscriminatorfits=20newDSfits. Use original fixed80/20 real/generated split, GRU7/3000updates/batch128, evaluator3/13/23/33/43, same TFsettings. No discriminator configuration tuning or model-selection change. MeanDS>0.3 flags batch failure.

Same five fixed real-trained FD001 predictors read each arm, no clipping/retraining. Save full predictions and original frozen checkpoint-specific test_matched100/dev_matched3000 generated window indices and labels, all15931windows andRUL0/1/2–5/6–30/stage bins. This is a proxy, not physical truth or real downstream utility. Keep original training-source/generator exposure limitations; no new test-case selection or official-test inference.

Freeze the same gate announced before any new scores: for either checkpoint at103, |paired meanDS change|>=0.03 and>=4/5same-direction OR |dev_matched inverse pooledRMSE change|>=1cycle AND >=10% relative AND >=4/5predictorssame-direction. Improvements OR deterioration qualify. Thresholds practical/exploratory, not significance or adoption rules. If triggered, run BOTH checkpoints and BOTH arms for113and123; otherwise stop103. Store gate decision/hash before confirmation. Maximum12fullbatches/60newDSfits, matching user2checkpoints×2arms×3samplingseeds. No extra weights, seeds, hyperparameters, retraining or downstream evaluation.

## Audits and stopping

Save immutable config/source/data/checkpoint/real-predictor/control-context hashes, old failure reference, worker logs and atomic progress. Require complete expected arm/checkpoint/seed grid, strict untouched weights/source inputs, identical labels, finite outputs, per-pair trace identity, original null branch preserved, score identities and proxy errors recomputed from saved predictions. Stop on any mismatch/failure; do not auto-adjust the protocol or relax checks. Completion automatically writes report and ledger/state. After budget stop. Strict reproduction defaults/history unchanged, training whole-batch numeric RULzeroing stays paused, EXT-002 stays closed/weight1.
