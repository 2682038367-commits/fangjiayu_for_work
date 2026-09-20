# FD002 seed 43: constrained soft learnable frequency threshold

## Scope

This is an isolated diagnostic for **one generative-model seed only: seed 43**.
The five seeds `3, 13, 23, 33, 43` below are evaluator seeds applied to the
same seed-43 synthetic dataset; they are not additional diffusion-model runs.

This experiment is not author-code-exact. It implements the paper's statement
that the frequency threshold is learnable, using an explicit constrained
parameterization and a differentiable mask that are not specified by the paper:

- effective threshold: `theta = 0.5 * sigmoid(alpha)`, hence `0 < theta < 0.5`;
- initial effective threshold: `0.25`;
- mask: `sigmoid((normalized_energy - theta) / temperature)`;
- temperature: `0.1`.

All other generative settings match the FD002 paper-aligned run: window 48,
14 sensors, DDPM with `T=1000`, linear beta schedule, 70 epochs, Adam,
learning rate `2e-3` with warm-up plus cosine scheduling, batch size 256, and
gradient clipping at 1.0.

## Training and sampling

- Best diffusion loss: `0.0298166849` at epoch 69.
- Training time: `212.10 s`.
- Full DDPM sampling time: `1826.02 s`.
- Learned effective thresholds: `0.25287744`, `0.02860600`.
- Synthetic data: `(41539, 48, 14)`, finite, range `[0, 1]`.
- Synthetic labels: `(41539, 1)`, finite, range `[0, 125]`.
- Exact-zero / exact-one data fractions: `1.64785%` / `2.05899%`.

The original hard-mask seed-43 run had `0.95151%` exact zeros and `2.15378%`
exact ones. Therefore this test does not indicate a simple removal of boundary
saturation.

## Fixed-split five-evaluator result

The split seed is `20260915`. Every split-index array is byte-for-byte equal at
the array level to the strict FD002 evaluation split. Evaluator training is 35
epochs for every evaluator seed.

| Evaluator seed | Predictive RMSE | Discriminative score |
|---:|---:|---:|
| 3  | 29.63899 | 0.31289 |
| 13 | 30.54540 | 0.19746 |
| 23 | 29.64783 | 0.26234 |
| 33 | 29.76072 | 0.28214 |
| 43 | 30.53962 | 0.04809 |
| **Mean ± sample SD** | **30.02651 ± 0.47348** | **0.22058 ± 0.10529** |

Using the previously declared failure rule, mean discriminative score `> 0.3`,
the constrained soft-mask seed-43 run **does not fail**. One evaluator seed is
still above 0.3 and the within-dataset evaluator variation is much larger than
for the original run, so this is evidence of improvement for this single
generation seed, not evidence that the method is generally stable.

## Paired context

| Seed-43 variant | RMSE mean ± SD | Discriminative mean ± SD | Failure (`mean > 0.3`) |
|---|---:|---:|:---:|
| Public-code random hard quantile | 31.36616 ± 0.36403 | 0.42110 ± 0.01502 | Yes |
| Fixed hard threshold 0.25 | 30.45776 ± 0.49110 | 0.26875 ± 0.03128 | No |
| Constrained soft learnable threshold | 30.02651 ± 0.47348 | 0.22058 ± 0.10529 | No |

Relative to the original seed-43 run, the soft variant lowers mean RMSE by
`1.33964` and mean discriminative score by `0.20052`. Relative to the fixed
hard-0.25 diagnostic, it lowers them by `0.43125` and `0.04817`, respectively.
Because only one generation seed was rerun, these differences cannot separate
the effect of threshold learning from seed-specific training variability.

## Artifacts

- Configuration: `configs/soft_learnable_threshold_fd002_seed43.yaml`
- Checkpoint: `checkpoints/FD002_w48_seed43_soft_theta025_tau01.pth`
- Loss history: `logs/FD002_w48_seed43_soft_theta025_tau01_training_loss.csv`
- Synthetic data: `outputs/FD002_w48_seed43_soft_theta025_tau01.npz`
- Evaluation configuration: `configs/evaluation_fd002_soft_threshold_seed43.yaml`
- Raw evaluation: `results/soft_learnable_threshold/fd002_seed43_evaluation_raw.csv`
- Evaluation summary: `results/soft_learnable_threshold/fd002_seed43_evaluation_summary.csv`
- Split indices: `results/soft_learnable_threshold/fd002_seed43_split.npz`
- Manifest: `results/soft_learnable_threshold/fd002_seed43_manifest.json`

