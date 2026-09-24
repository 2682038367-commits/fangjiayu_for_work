# Source snapshot

- Repository: https://github.com/Dolphin-wang/MetaIndux-TS
- Branch archive: `main`
- Downloaded: 2026-09-14
- Archive SHA-256: `4e675046131a61cbd747901c801dafee5f34b603f5bcce638a87e2f502b1f959`
- Archive size: 5,945 KiB

The host does not have the `git` executable, so the repository was downloaded
from GitHub as a branch archive. The archive checksum pins the exact retrieved
source contents.

Local paper-alignment patches are limited to:

- honoring command-line batch size;
- honoring the requested optimizer;
- passing the beta schedule into the diffusion implementation;
- exposing and honoring a random seed;
- preserving caller-provided checkpoint and synthetic-data paths.
- sampling synthetic data in bounded batches to avoid GPU out-of-memory errors.
- recording per-epoch loss, best checkpoint metadata, timings, metrics, and runtime metadata.
- correcting the public code's swapped W&B labels for MAE and RUL score.
- allowing predictive and discriminative evaluators to reuse fixed split
  indices and explicit evaluator seeds.
- explicitly seeding TensorFlow discriminator layer initializers and disabling
  nondeterministic evaluation paths for cross-process reproducibility.
- exposing a constant-learning-rate option for the FD002 learning-rate
  stability ablation while preserving warm-up plus cosine as the default.
- exposing an optional fixed frequency-mask threshold for a paired stability
  ablation; the public random-threshold behavior remains the default.
- exposing an optional constrained, differentiable soft frequency mask for the
  paper-described learnable-threshold diagnostic; the public hard-mask behavior
  remains the default.
- exposing an optional binary-forward, sigmoid-STE-backward energy threshold
  implementation; its unspecified completions are documented in
  `notes/binary_ste_fd002_protocol.md`. Existing modes remain unchanged.
- explicitly passing evaluator seeds in the built-in evaluation entry point,
  persisting generation-stage records before evaluation, and providing validated
  evaluation-only recovery of already generated artifacts.
- recording per-module frequency masks during DDPM sampling without changing
  their forward values. Cross-mode analysis uses observed `mask_drop_ratio`;
  quantile fractions and normalized-energy cutoffs remain explicitly distinct.
