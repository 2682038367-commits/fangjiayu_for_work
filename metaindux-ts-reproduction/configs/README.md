# Experiment configurations

Configuration files remain flat because the executable runners and frozen
provenance records reference these paths directly.

- `paper_*.yaml`: paper-aligned generation runs.
- `evaluation_*public_code*.yaml` and
  `evaluation_fd002_frequency_threshold_seven_seeds.yaml`: frozen formal
  evaluation protocols.
- `*threshold*`, `binary_ste*`, and `*lr*`: completed diagnostic variants;
  they must not be substituted for the author public-code main result.

The main finalized FD002 comparison uses
`evaluation_fd002_frequency_threshold_seven_seeds.yaml` and
`evaluation_fd002_public_arm_seven_seeds.yaml`.
