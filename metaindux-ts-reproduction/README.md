# MetaIndux-TS reproduction

> Status: the window-48 reproduction and its local generated-vs-real baseline diagnostic are frozen.
> Do not mix historical diagnostic outputs with the finalized public-code
> comparison. The authoritative numerical record is
> `results/frozen_numbers.json`; the supporting reports are in
> `docs/conclusions/final/`.

## Project map

- `upstream/`: pinned author source snapshot with documented minimal
  reproduction patches.
- `scripts/`: core data, generation, and evaluation entry points.
- `configs/`: executable configurations; kept flat because provenance records
  reference their exact paths.
- `analysis/`: project-owned audit, diagnostic, and historical helper code.
- `results/`: finalized tables and small machine-readable results. The one
  incompatible legacy FD002 split is isolated under
  `results/_deprecated_incompatible_split/`.
- `docs/`: conclusions, figures, and handoff provenance.

This directory contains a paper-aligned reproduction environment for
MetaIndux-TS. The source snapshot is in `upstream/`, with minimal patches so
command-line seeds, batch size, optimizer, beta schedule, and output paths are
honored.

## Environment

```bash
source .venv/bin/activate
```

The exact installed environment is recorded in `requirements-lock.txt`.
PyTorch 2.1.2 with CUDA 12.1 matches the version commented in the authors'
requirements file.

The experiment runner redirects Matplotlib and PyKeOps caches into the local
`.cache/` directory so it also works when the home directory is read-only.

Recreate the environment from scratch with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduction.txt
```

Validate data loading, the supplied checkpoint, and one GPU forward pass with:

```bash
.venv/bin/python scripts/smoke_test.py
```

## Data

The canonical C-MAPSS directory is linked at `data/raw/CMAPSSData`. FD002,
FD003, and FD004 are linked into `upstream/data/`; the source snapshot already
includes FD001.

## Paper-aligned runs

Inspect all commands without starting training:

```bash
.venv/bin/python scripts/run_experiments.py --dry-run
```

Run five seeds:

```bash
.venv/bin/python scripts/run_experiments.py
```

The default configuration is `configs/paper_fd001.yaml`. The full four-dataset,
three-window matrix is in `configs/paper_full.yaml`.

The completed FD001 results are summarized in
`results/fd001_paper_report.md`, with machine-readable values in
`results/fd001_paper_metrics.csv`.

The completed FD002 window-48 results are summarized in
`results/fd002_paper_report.md`, with machine-readable values in
`results/fd002_paper_metrics.csv`. Its exact run configuration is
`configs/paper_fd002.yaml`.

The controlled FD002 evaluation study (one fixed split, five evaluator seeds
for each of five generated datasets, and within/between variance decomposition)
is in `results/fd002_evaluation_stability_report.md`. The paper-aligned 70/30
evaluation configuration is `configs/evaluation_fd002_stability_paper.yaml`.

The paired FD002 learning-rate ablation (warm-up plus cosine versus constant
0.002 for seeds 13, 23, and 43) is documented in
`results/lr_validation/fd002_lr_validation_report.md`. Constant LR was not an
improvement and is not adopted as the reproduction default.

## FD002 paper-intended learnable threshold

The optional `soft_learnable_energy` mode implements the paper-described
backpropagation of the frequency threshold with the explicitly documented
differentiable relaxation
`theta = 0.5 * sigmoid(alpha)` and
`mask = sigmoid((normalized_energy - theta) / 0.1)`. This is a paper-intended
mechanism reconstruction, not an author-code-exact setting; the public hard
mask remains the default.

Seed 43 is already complete. Inspect, then run the other six generation seeds:

```bash
.venv/bin/python scripts/run_experiments.py \
  --config configs/soft_learnable_threshold_fd002_remaining_seeds.yaml \
  --dry-run

.venv/bin/python scripts/run_experiments.py \
  --config configs/soft_learnable_threshold_fd002_remaining_seeds.yaml \
  --skip-completed
```

After all six synthetic datasets exist, evaluate each one on the identical
fixed split with the same five evaluator seeds:

```bash
TF_ENABLE_ONEDNN_OPTS=0 TF_DETERMINISTIC_OPS=1 \
.venv/bin/python scripts/evaluate_fd002_stability.py \
  --config configs/evaluation_fd002_soft_threshold_remaining_seeds.yaml
```

Evaluation has resume enabled and writes one row after every evaluator run.
The generation runner records each best checkpoint's raw parameter and
normalized-energy threshold in its metrics JSON. These values are not a
frequency-drop percentage.

New sampling runs also record `frequency_mask_statistics` for every spectral
module. The cross-mode field is `mask_drop_ratio`: the zero fraction for a
binary mask and `mean(1-mask)` (mean attenuation) for a soft mask. Analyse it
with:

```bash
.venv/bin/python analysis/diagnostics/theta_rmse_analysis.py
```

Legacy checkpoints do not contain the intermediate DDPM masks, so their exact
historical drop ratios require sampling to be repeated; they are never inferred
from the threshold parameter alone.

## Binary-forward STE run

The frozen protocol and setting provenance are in
`notes/binary_ste_fd002_protocol.md`. Run all seven generators followed by their
35 fixed-split evaluator runs from the host terminal:

```bash
bash scripts/run_fd002_ste_all.sh
```

For background execution with a system sleep inhibitor:

```bash
nohup systemd-inhibit --what=sleep --mode=block --why="FD002 reproduction" \
  bash scripts/run_fd002_ste_all.sh \
  >> logs/fd002_binary_ste_console.log 2>&1 < /dev/null &
tail -f logs/fd002_binary_ste_console.log
```

Screen blanking and locking are fine; suspend, hibernation, shutdown and reboot
are not. Keep AC power connected. If systemd-inhibit reports an authorization
error, disable automatic suspend in desktop power settings before using nohup
without the inhibitor. Completed generators/evaluators are skipped on rerun;
an interrupted unfinished generator is restarted from scratch.

The STE launcher also enables `--reuse-generated`: when checkpoint, full loss
history, synthetic data and recorded configuration validate successfully, a run
that failed during built-in evaluation resumes with `--state eval`. It does not
repeat training or sampling. Recovered synthetic labels are checked against the
current dataset, and the checkpoint is strictly loaded. Invalid complete-looking
artifacts or mismatched configuration abort rather than being silently reused.
Partial training/sampling still restarts that unfinished generator.

Built-in evaluators now explicitly use the generation seed as their evaluator
seed. This fixes TensorFlow deterministic-mode initialization and makes the
built-in metric reproducible after recovery; it is separate from the formal
fixed-split five-evaluator protocol. Earlier single-run metrics are not strictly
paired with this seeded built-in evaluation. Future runs persist generation
loss, epoch, thresholds and timings in `*_metrics.generation.json` before
post-hoc evaluation. For legacy failed runs, loss/epoch can be reconstructed;
unrecorded timings remain unavailable rather than being invented. Recovery
keeps the earlier runner metadata under `recovery_source`.

## GPU device access

The host GPU is an NVIDIA GeForce RTX 4060 Ti (8 GB), and its driver is
working. The managed execution sandbox used to prepare this directory does not
expose `/dev/nvidia*`. Run from a normal host terminal, or start a container or
session with NVIDIA GPU device passthrough enabled.
