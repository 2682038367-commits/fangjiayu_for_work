# Reproduction analysis

This directory contains project-owned analysis code. It is deliberately
separate from `upstream/`, which is the author source snapshot.

- `audit/`: read-only integrity checks and the generator for the frozen result
  record. Run `bash analysis/audit/audit_checklist.sh .` from the project root.
- `diagnostics/`: exploratory analyses of thresholds, checkpoints and metric
  variation. They do not alter the frozen public-code reproduction result.
- `historical/`: runners retained to document how completed FD002 experiments
  were created. They are not part of the current frozen protocol.

The machine-readable result authority is `results/frozen_numbers.json`; use
that file rather than copying values from prose reports.

For cross-mode frequency-mask analysis, use
`diagnostics/theta_rmse_analysis.py`. It compares the observed
`mask_drop_ratio`, not raw/effective threshold parameters: public-code θ is a
quantile fraction, whereas learnable-mode θ is a normalized-energy cutoff.
Exact ratios from completed legacy runs cannot be reconstructed from their
checkpoints; repeating sampling records them without changing mask behavior.
