# Core executable scripts

These are the maintained execution entry points for the reproduction:

- `smoke_test.py`: environment and data-path check.
- `run_experiments.py`: generation training and sampling from a YAML config.
- `evaluate_fd002_stability.py`: fixed-split synthetic-data evaluation.
- `evaluate_real_data_baseline.py`: real-data control under the saved split.
- `run_fd002_ste_all.sh` and `run_fd003_fd004_public_w48.sh`: recorded batch
  launchers for completed experiments.

Result auditing and exploratory analyses live in `../analysis/`, not here.
