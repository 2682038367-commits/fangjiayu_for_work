# Zero-label repair: downstream real-test comparison

FD001/FD002 full-data w48, generation seed3, five paired evaluator seeds. Diagnostic, not replacement paper results.

| Dataset | RUL bin | Control RMSE | Fix RMSE | Fix − control |
|---|---|---:|---:|---:|
| FD001 | 0-30 | 11.176 | 6.731 | -4.445 |
| FD001 | 125 | 10.841 | 14.307 | +3.465 |
| FD001 | 31-60 | 13.809 | 11.788 | -2.021 |
| FD001 | 61-90 | 25.230 | 21.901 | -3.329 |
| FD001 | 91-124 | 15.136 | 15.633 | +0.497 |
| FD001 | all | 15.710 | 14.524 | -1.186 |
| FD002 | 0-30 | 23.383 | 26.597 | +3.215 |
| FD002 | 125 | 30.786 | 37.208 | +6.422 |
| FD002 | 31-60 | 21.637 | 18.159 | -3.479 |
| FD002 | 61-90 | 19.773 | 19.301 | -0.471 |
| FD002 | 91-124 | 25.892 | 29.094 | +3.202 |
| FD002 | all | 24.896 | 27.654 | +2.758 |

Means above average evaluator RMSE. Pooled segment RMSE also saved in comparison.csv.
CIs quantify evaluator-seed variability only, conditional on one fixed generated dataset per arm; not generator-seed variability.
No output clipping, no real-test selection, no generator retraining.
