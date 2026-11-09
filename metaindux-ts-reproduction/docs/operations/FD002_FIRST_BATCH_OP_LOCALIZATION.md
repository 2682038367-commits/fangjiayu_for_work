# FD002 first-batch earliest operator localization

Diagnostic/supporting, scope `scope-20261103-fd002-first-batch-op-localization`.
User authorizes locating the first differing operator of the saved seed43 first batch. Zero backward, optimizer updates, second batches, generated data or predictor/discriminator training.

Reuse the current original-entry/audited-entry setup from the sealed epoch1 campaign, with identical environment, initialization and first-batch actual random question. Stop after the first loss forward, before backward. Use common module-stack hooks and TorchDispatch ATen operator tracing to record chronological actual arguments/results, dtype/shape/strides/storage offset/alignment and closest upstream source. Passive observation excludes observer-origin operations. Prefix256 operator tensors are content-addressed for magnitude/standalone replay, all remaining operators are hashed.

Preconditions: recreated full initial state exactly matches saved canonical state, all first-batch actual indices/NumPy draw/timestep/noise/mask exactly match canonical tape, pre-loss RNG and noisy input/actual conditions match where recorded. No changes to numerical loss/network/defaults.

Budget: initial two first-batch forward traces (public/worker). Compare chronological operator signatures and inputs before declaring the first unequal output. An operator receiving already unequal inputs is downstream, not the origin. If tracing suppresses the prior divergence, allow one lighter deferred-tensor pair (no synchronous tensor copies inside forward), preserving it as separate instrumentation condition; no more than four full first-batch forwards. Stop inconclusive if difference remains unobserved. Never select favorable seeds/examples.

If first origin is a supported convolution/GEMM, at most four standalone calls (two fresh processes, one native and one deterministic calculation each), replaying the exact saved arguments. This is a local diagnostic of that operator; do not run the full network in strict mode, replace median, change thresholds or adopt fixes. Report supported/unsupported outcomes without silently relaxing checks. No historical kernel-name attribution without measured evidence.

All hooks may affect timing, autotuning, allocator addresses and observed divergence. Conclusions apply to traced current setups; original historical Sep14 exact kernel state remains unavailable. Stop after localization/limited confirmation, preserve failures and snapshots; no training/default changes, no FD003/4 or125 investigation.
