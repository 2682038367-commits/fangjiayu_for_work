# Public-code quantile parameter vs RMSE

Scope: `hard_random_quantile` public-code checkpoints only. Every RMSE is the mean of the same five evaluator seeds from the formal summary CSV for that dataset.

`theta_1_quantile` and `theta_2_quantile` are quantile parameters. They are not measured frequency-drop ratios. This diagnostic is descriptive; it does not identify a causal effect of theta because other model initializations differ by generation seed.

| Dataset | n | rho(theta1, RMSE) | p | rho(theta2, RMSE) | p | rho(mean theta, RMSE) | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| FD002 | 7 | +0.143 | 0.783 | -0.643 | 0.139 | -0.464 | 0.302 |
| FD003 | 5 | -0.600 | 0.350 | +0.100 | 0.950 | -0.600 | 0.350 |
| FD004 | 5 | -0.200 | 0.783 | +0.300 | 0.683 | -0.200 | 0.783 |

FD001 is excluded because its completed table is not the same formal five-evaluator protocol. Fixed, soft, and STE arms are excluded by design rather than treated as cross-dataset correlation observations.

## FD002 fixed theta=0.25 paired ablation

This is kept separate from the correlations above. It uses all seven matched generation seeds and five common evaluator seeds per generator.

| Generation seed | fixed theta RMSE | random-quantile RMSE | fixed - random |
|---:|---:|---:|---:|
| 3 | 27.350267 | 26.990459 | +0.359807 |
| 13 | 24.022609 | 23.540359 | +0.482249 |
| 23 | 24.255546 | 26.849480 | -2.593934 |
| 33 | 25.322802 | 25.212566 | +0.110235 |
| 43 | 30.457763 | 31.366155 | -0.908392 |
| 53 | 24.200268 | 35.304366 | -11.104099 |
| 63 | 26.226433 | 28.799061 | -2.572628 |

Mean paired difference (fixed - random): -2.3181 RMSE; sample SD: 4.0872; paired t(6) = -1.501, p = 0.1841; exact Wilcoxon p = 0.2188.

This ablation estimates the consequence of replacing random quantile parameters with 0.25 in this implementation. It does not establish that theta alone causes an individual seed's RMSE, nor does it alter the frozen public-code reproduction result.
