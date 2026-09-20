# MetaIndux-TS FD002 paper-aligned reproduction

## Protocol

- Dataset: C-MAPSS FD002
- Sensors: 14
- Window size: 48
- Model: `DiffUnet_fre` (MetaIndux-TS)
- Optimizer: Adam
- Learning rate: 0.002
- Batch size: 256 (from the public code; not reported in the paper)
- Epochs: 70
- Diffusion steps: 1000
- Beta schedule: linear
- Sampling: DDPM
- Seeds: 3, 13, 23, 33, 43
- Hardware: NVIDIA GeForce RTX 4060 Ti, 8188 MiB, driver 595.84
- Software: Python 3.10.12, PyTorch 2.1.2+cu121, CUDA runtime 12.1

## Results

| Seed | Best epoch | Diffusion loss | RMSE | MAE | RUL score | Discriminative | Train (s) | Sample (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 64 | 0.02420 | 27.06103 | 22.25618 | 15.23299 | 0.17158 | 200.65 | 1814.83 |
| 13 | 68 | 0.02419 | 23.44926 | 17.95671 | 34.97257 | 0.08197 | 201.11 | 1796.04 |
| 23 | 32 | 0.02681 | 25.55040 | 21.04426 | 30.05365 | 0.47791 | 200.15 | 1897.85 |
| 33 | 67 | 0.02423 | 24.52933 | 19.09611 | 26.50148 | 0.10279 | 200.80 | 1856.14 |
| 43 | 66 | 0.03018 | 31.20257 | 24.95005 | 158.55157 | 0.39378 | 204.03 | 1873.50 |
| **Mean** | **59.40** | **0.02592** | **26.35852** | **21.06066** | **53.06245** | **0.24561** | **201.35** | **1847.67** |
| **Sample std** | **15.39** | **0.00263** | **3.01749** | **2.74093** | **59.41623** | **0.17929** | **1.54** | **41.84** |

Additional Context-FID and cross-correlation results are retained in
`fd002_paper_metrics.csv` and the per-seed JSON files under `results/runs/`.

## Paper comparison

Table I of the paper reports, specifically for **FD002 with window length 48**,
a predictive RMSE of 24.565 and a discriminative score of 0.109. Lower is
better for both metrics. The five-seed reproduction obtained:

- RMSE 26.35852 ± 3.01749: 1.79352 (7.30%) higher than the paper value.
- Discriminative score 0.24561 ± 0.17929: 0.13661 higher than the paper value.

The mean therefore does not reproduce the paper's FD002-48 scores. Variation
is substantial: seeds 13 and 33 reach discriminative scores of 0.08197 and
0.10279, while seeds 23 and 43 are much worse. The authors state that each
experiment was repeated five times and averaged, but do not publish their five
random seeds. This report retains every run rather than selecting the best one.

The paper's FD002 predictive RMSE of 22.860 belongs to window length 96, not
window length 48, and is not the comparison target for this run.

## Timing

- Total diffusion training time across five seeds: 1006.74 s (16 min 46.74 s).
- Total DDPM sampling time across five seeds: 9238.36 s (2 h 33 min 58.36 s).
- Mean end-to-end wall time: 2129.55 ± 43.18 s per seed.

## Artifact validation

All five runs have a complete metadata JSON, a 70-row training-loss CSV, a
best checkpoint, a generated archive, and a metrics JSON. Every generated
archive contains finite `float32` data with shape `(41539, 48, 14)` and labels
with shape `(41539, 1)`. Generated values lie in `[0, 1]`; RUL labels lie in
`[0, 125]`.

The complete console log is `logs/fd002_paper.log`. Machine-readable aggregate
results are in `results/fd002_paper_metrics.csv`; per-seed metrics and runtime
metadata are in `results/runs/`.

The upstream `wandb_record` function originally swapped only the display labels
for MAE and RUL score. Seeds 3 and 13 had already imported that function before
the label-only fix was applied, so their raw offline W&B summaries show those
two names reversed. Their per-seed metrics JSON files, this CSV, RMSE, and
discriminative scores use the correct return-value mapping. Seeds 23, 33, and
43 use the corrected W&B labels as well.

The original one-evaluator-per-generation scores in this report mix evaluator
and generation randomness. The controlled 5 × 5 follow-up analysis is in
`fd002_evaluation_stability_report.md`; use that report for uncertainty and
extreme-seed conclusions.
