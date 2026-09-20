# FD002 learning-rate validation

## Question

Does replacing the public code's warm-up plus cosine schedule (configured LR
0.002, actual peak LR 0.005) with a constant LR of 0.002 prevent the FD002
generation failures?

## Controlled design

- Dataset/window/model: FD002 / 48 / MetaIndux-TS (`DiffUnet_fre`)
- Paired generation seeds: 13 (previously good), 23 and 43 (previously failed)
- Baseline: warm-up multiplier 2.5 followed by cosine decay
- Intervention: constant LR 0.002 for all 70 epochs
- All other training and DDPM sampling settings unchanged
- Each generated dataset evaluated on the same fixed split with evaluator seeds
  3, 13, 23, 33, and 43
- Predictive evaluator uses the paper's 70/30 train/validation split
- Discriminator uses the public code's fixed 80/20 train/test split

## Training behavior

| Seed | Warm-up best loss/epoch | Constant best loss/epoch | Observation |
|---:|---:|---:|---|
| 13 | 0.02419 / 68 | 0.02586 / 65 | stable, but constant LR converged worse |
| 23 | 0.02681 / 32 | 0.02644 / 63 | instability delayed, not eliminated |
| 43 | 0.03018 / 66 | 0.02636 / 67 | constant LR substantially improved loss |

Seed 23's baseline loss jumped from 0.02681 to 0.05861 at epoch 32→33. With
constant LR, its large jump moved to epoch 68→69; sampling used the earlier
best checkpoint at epoch 63. Seed 43's baseline jump from 0.03397 to 0.06251
at epoch 10→11 disappeared under constant LR.

The three-seed mean best diffusion loss improved from 0.02706 to 0.02622, but
diffusion loss did not reliably predict downstream quality.

## Fixed-evaluator results

Values are mean ± sample standard deviation over the five evaluator seeds.
Delta is constant minus warm-up; lower is better.

| Seed | Warm-up RMSE | Constant RMSE | Δ RMSE | Warm-up discriminative | Constant discriminative | Δ discriminative |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 23.54036 ± 0.44398 | 26.57000 ± 0.37388 | +3.02964 | 0.10669 ± 0.03220 | 0.40658 ± 0.07281 | +0.29989 |
| 23 | 26.84948 ± 1.11343 | 26.37018 ± 0.63239 | -0.47930 | 0.45628 ± 0.03220 | 0.45034 ± 0.01463 | -0.00595 |
| 43 | 31.36616 ± 0.36403 | 35.04514 ± 0.42915 | +3.67899 | 0.42110 ± 0.01502 | 0.36975 ± 0.04645 | -0.05135 |
| **Mean** | **27.25200** | **29.32844** | **+2.07644** | **0.32803** | **0.40889** | **+0.08087** |

Constant LR turned the previously good seed 13 into a clear fidelity failure.
It did not resolve seed 23. Seed 43's fidelity improved moderately, but its
predictive RMSE became substantially worse.

## Generated-distribution diagnostics

The fraction of values within 0.001 of the clipping boundaries changed as
follows:

| Seed | Warm-up edge fraction | Constant edge fraction |
|---:|---:|---:|
| 13 | 2.86% | 1.88% |
| 23 | 8.23% | 2.79% |
| 43 | 3.83% | 3.51% |

Constant LR reduced obvious clipping artifacts, especially for seed 23, and
improved several marginal/conditional distribution diagnostics for seeds 23
and 43. Nevertheless, the deterministic discriminator still separated their
real and generated sequences reliably. Boundary saturation is therefore a
symptom, not the sole cause of failure.

For seed 13, constant LR reduced edge saturation but increased mean shifts,
correlation error, spectral error, and conditional-distribution error. This
explains why a visually cleaner numeric range did not translate into better
fidelity.

## Conclusion

The hypothesis is rejected: simply replacing warm-up plus cosine decay with a
constant LR of 0.002 does not stabilize FD002 generation. High LR contributes
to the timing and severity of loss spikes, but it is not the root cause of the
seed-dependent failures. Constant LR should **not** replace the paper/code
baseline for subsequent reproduction runs.

The next highest-value hypotheses are the randomly initialized but
non-learnable frequency-mask threshold and the hard-coded classifier-free
guidance scale of 3. These should be tested in separate one-factor experiments;
the current learning-rate results must remain a distinct ablation series.

## Artifacts

- Generation configuration: `configs/lr_validation_fd002_constant.yaml`
- Evaluation configuration: `configs/evaluation_fd002_lr_constant.yaml`
- Paired comparison: `results/lr_validation/fd002_lr_paired_comparison.csv`
- Raw 15-run metrics: `results/lr_validation/fd002_constant_lr_evaluation_raw.csv`
- Evaluation summary: `results/lr_validation/fd002_constant_lr_evaluation_summary.csv`
- Generation log: `logs/fd002_lr_constant_generation.log`
- Evaluation log: `logs/fd002_lr_constant_evaluation.log`
