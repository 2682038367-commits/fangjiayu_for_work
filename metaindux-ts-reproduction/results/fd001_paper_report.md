# MetaIndux-TS FD001 paper-aligned reproduction

## Protocol

- Dataset: C-MAPSS FD001
- Sensors: 14
- Window size: 48
- Model: `DiffUnet_fre`
- Optimizer: Adam
- Learning rate: 0.002
- Batch size: 256 (inferred from the public code; not reported in the paper)
- Epochs: 70
- Diffusion steps: 1000
- Beta schedule: linear
- Sampling: DDPM
- Seeds: 3, 13, 23, 33, 43
- Hardware: NVIDIA GeForce RTX 4060 Ti 8 GB

## Results

| Seed | RMSE | MAE | RUL score | Discriminative | Context-FID | Cross-correlation |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 14.45696 | 3.26346 | 10.75189 | 0.10292 | 0.19913 | 0.40140 |
| 13 | 15.24286 | 3.26003 | 11.07606 | 0.10276 | 0.68906 | 0.55619 |
| 23 | 13.88194 | 3.00965 | 10.27471 | 0.12818 | 0.42024 | 0.78818 |
| 33 | 14.34857 | 2.92024 | 10.81472 | 0.15422 | 0.70933 | 0.69458 |
| 43 | 13.83272 | 2.83630 | 10.21964 | 0.15328 | 0.28583 | 0.60457 |
| **Mean** | **14.35261** | **3.05794** | **10.62740** | **0.12827** | **0.46072** | **0.60898** |
| **Sample std** | **0.56905** | **0.19589** | **0.36829** | **0.02546** | **0.23163** | **0.14606** |

## Paper comparison

Table I of the paper reports FD001 with window length 48 RMSE 13.949 and
discriminative score 0.102. This run obtained mean RMSE 14.35261 (0.40361
higher, or 2.89%) and mean discriminative score 0.12827 (0.02627 higher).
Lower is better for both metrics. The previously cited RMSE 15.191 belongs to
FD001 with window length 96, not window length 48.

Both five-seed mean metrics are worse than the corresponding paper values.
Seeds 3 and 13 individually reproduce the reported discriminative score
closely (0.10292 and 0.10276), but the five-seed mean exposes substantial
stochastic variation. The exact five seeds used by the authors were not
disclosed.

## Artifact validation

All five checkpoints exist. Each generated archive contains finite `float32`
data with shape `(15931, 48, 14)` and labels with shape `(15931, 1)`. Generated
values lie in `[0, 1]`; RUL labels lie in `[0, 125]`.

The complete console log is `logs/fd001_paper.log`, and the machine-readable
metrics are in `results/fd001_paper_metrics.csv`.
