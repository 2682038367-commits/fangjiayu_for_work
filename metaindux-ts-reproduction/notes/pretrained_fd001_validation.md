# FD001 pretrained checkpoint validation

- Date: 2026-09-14
- Model: `DiffUnet_fre`
- Checkpoint: `upstream/weights/DiffUnet_fre_FD001_48.pth`
- Dataset: FD001
- Window size: 48
- Sensors: 14
- Diffusion steps: 1000
- Beta schedule: linear
- Sampling: DDPM
- Seed: 3
- Sampling batch size: 256
- End-to-end runtime reported by W&B: 731 seconds

## Generated artifact

- File: `upstream/weights/syn_data/syn_FD001_DiffUnet_fre_48ddpm.npz`
- Size: 42,886,756 bytes
- Keys: `data`, `label`
- `data`: shape `(15931, 48, 14)`, `float32`, finite, range `[0, 1]`
- `label`: shape `(15931, 1)`, `float32`, finite, range `[0, 125]`

## Evaluation

- RMSE: 14.52561
- MAE: 3.40983
- asymmetric RUL score: 10.72452
- discriminative score: 0.31487
- Context-FID: 1.19842
- cross-correlation loss: 0.72888

This is one evaluation run using the supplied checkpoint. It verifies the full
pipeline, but it is not the paper's five-run mean. Downstream evaluators are
stochastic, so the five-seed experiment remains necessary for a strict result
comparison.
