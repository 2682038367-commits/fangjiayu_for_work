# FD002 binary-forward STE protocol

This implementation follows the paper's normalized-energy threshold comparison,
binary forward mask, and learnable threshold. Frequencies below the threshold
are removed, following the paper prose and the public code's retention direction.
The public code instead uses a quantile parameter with a hard comparison that
does not propagate threshold gradients. That original mode remains unchanged.

Unspecified gradient implementation is completed using a sigmoid straight-through
estimator: `hard + (soft - soft.detach())`. Forward values are exactly 0 or 1;
backward derivatives use `sigmoid((energy - theta) / 0.1)`. Sampling uses the same
binary forward mask as training, not sigmoid weights.

Other explicitly disclosed completions: positive energy threshold via softplus,
initial effective theta 1.0 (the median-normalized energy reference scale), no
upper bound, and surrogate temperature 0.1. These are not claimed author settings.
Median normalization and epsilon 1e-6 are retained from public code. The same
random initialization draw is consumed in every mode, preserving paired
initialization of non-threshold weights for a given generation seed.

Paper settings: Adam, initial lr 0.002, 70 epochs, DDPM T=1000, linear beta schedule,
predictive synthetic train/validation 70/30 and real test data.
Public-code settings retained: architecture, warm-up/cosine and multiplier 2.5,
batch size 256, grad clipping 1.0, conditioning behavior, and best-training-loss
checkpoint selection after epoch 5. Sampling remains memory-bounded for the 8GB
GPU, as in all prior local runs.

Seven predetermined generation seeds: 3,13,23,33,43,53,63. The original first five
can be reported as the paper-style five repeats; all seven are the extended
diagnostic set. Each generated dataset uses evaluator seeds 3,13,23,33,43 and
split seed 20260915. Evaluator repetitions are not extra generator repetitions.

Artifacts have suffix `binary_ste_theta1_tau01` and do not replace prior hard,
fixed-quantile, or continuous-soft artifacts. No improvement is guaranteed.
