# FD002 evaluation stability analysis

## Controlled protocol

- Dataset/window: FD002, 48
- Generation seeds: 3, 13, 23, 33, 43
- Evaluator seeds applied to every generated dataset: 3, 13, 23, 33, 43
- Fixed split seed: 20260915
- Predictive evaluator split: 70% synthetic train / 30% synthetic validation
  (paper protocol), 29,077 / 12,462 samples
- Discriminator split: fixed 80% train / 20% test per class (public-code
  default), 33,231 / 8,308 samples
- Split-index SHA-256:
  `8c390ebfebf4cc3b18ff4b9e0cfa93c7b883e8d6a213364fc45d8bed80e82c76`
- TensorFlow oneDNN disabled; TensorFlow intra/inter-op threads fixed to one;
  NumPy, TensorFlow, PyTorch, GRU, and dense initializers explicitly seeded.

The same split-index arrays are reused for all 25 generation/evaluator seed
combinations. Evaluator seeds affect initialization and mini-batch order, but
not sample membership in train, validation, or test sets.

An independent fresh-process repeat of `(generation seed 3, evaluator seed 3)`
matched RMSE, MAE, RUL score, discriminative score, fake accuracy, and real
accuracy exactly. This confirms that the final evaluator implementation is
deterministic for a fixed seed.

## Per-generation results

Values are mean ± sample standard deviation over the five evaluator seeds.

| Generation seed | Predictive RMSE | Discriminative score | Interpretation |
|---:|---:|---:|---|
| 3 | 26.99046 ± 0.18422 | 0.13065 ± 0.01184 | moderate fidelity |
| 13 | 23.54036 ± 0.44398 | 0.10669 ± 0.03220 | good fidelity |
| 23 | 26.84948 ± 1.11343 | 0.45628 ± 0.03220 | generation failure |
| 33 | 25.21257 ± 0.35757 | 0.09878 ± 0.02085 | good fidelity |
| 43 | 31.36616 ± 0.36403 | 0.42110 ± 0.01502 | generation failure |

The balanced mean over the five generation-seed means is RMSE 26.79180 and
discriminative score 0.24270. The corresponding paper values for FD002-48 are
24.565 and 0.109.

## Variance decomposition

`Pooled within SD` is the square root of the mean evaluator variance within
each generated dataset. `Between SD` is the sample standard deviation of the
five generation-seed means. The corrected generation component subtracts the
finite-five-evaluator contribution from the between-generation variance.

| Metric | Pooled within-evaluator SD | Between-generation SD of means | Corrected generation SD | Estimated variance from generation |
|---|---:|---:|---:|---:|
| RMSE | 0.58841 | 2.91691 | 2.90501 | 96.1% |
| Discriminative | 0.02398 | 0.17973 | 0.17941 | 98.2% |

Generation-seed variation is therefore the dominant uncertainty: its estimated
standard-deviation component is 4.94 times the evaluator component for RMSE and
7.48 times for the discriminative score.

## Extreme-seed diagnosis

Seeds 23 and 43 are not isolated evaluator failures. All five evaluators give
them very high discriminative scores. The three remaining seeds average RMSE
25.24780 and discriminative score 0.11204, close to the paper's discriminative
score of 0.109. This exclusion is diagnostic only and is **not** the reported
five-seed result.

The observed generation-failure rate is 2/5 (40%), but five generation seeds
are too few to estimate that probability precisely. Best diffusion loss is
positively associated with worse downstream results in this small sample, but
the sample is insufficient to use training loss as a reliable rejection rule.

## Decision

Keep the model and paper-aligned training configuration fixed for the next
stage and increase the number of generation seeds first. Add at least five new
generation seeds (ten total), evaluate each with this same five-seed evaluator
panel, and report the failure rate plus mean, standard deviation, median, and
interquartile range. Do not discard failed seeds from headline results.

If the failure rate remains material after ten or more generation seeds (for
example, discriminative score above 0.3 in at least 20% of runs), then investigate
training-stability changes separately rather than tuning against these five
runs. Candidate diagnostics are checkpoint-selection sensitivity, loss curves,
gradient norms, and generated-data saturation; any model change must start a
new experiment series and must not be mixed with the present paper-aligned
results.

## Artifacts

- Raw 25-run results: `fd002_evaluation_stability_paper70_raw.csv`
- Variance summary: `fd002_evaluation_stability_paper70_summary.csv`
- Fixed split indices: `fd002_evaluation_fixed_split_paper70.npz`
- Configuration and environment manifest:
  `fd002_evaluation_stability_paper70_manifest.json`
- Full console log: `logs/fd002_evaluation_stability_paper70.log`
- Evaluator checkpoints: `checkpoints/evaluators/fd002_paper70/`
- Independent determinism check: `results/checks/`

The earlier 80/20 predictive-split run is retained as a public-code-aligned
diagnostic in files without the `paper70` suffix. It is not used for the final
paper comparison above.
