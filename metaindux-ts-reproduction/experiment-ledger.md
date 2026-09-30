# Experiment ledger

## Chronological index

| Time | Run ID | Role | Changed factor | Key result | Conclusion | Snapshot |
|---|---|---|---|---|---|---|
| 2026-09-28T16:57:15+08:00 | core-a-fewshot-fd001-w48-20260928 | core:A | full data to paper few-shot split | RMSE 27.336 vs 5.219 | not reproduced | `runs/core-a-fewshot-fd001-w48-20260928/run-manifest.yaml` |
| 2026-09-28T17:04:13+08:00 | core-a-fewshot-full-w48-20260928 | core:A | expand to FD001–FD003 | all three completed datasets miss paper | stop Candidate A remainder | `runs/core-a-fewshot-full-w48-20260928/run-manifest.yaml` |
| 2026-09-29T16:38:25+08:00 | main-w24-fd003-fd004-20260929 | main | w48 to w24 | average RMSE 23.646 vs 22.318 | partial numerical reproduction | `runs/main-w24-fd003-fd004-20260929/run-manifest.yaml` |
| 2026-09-30 | evidence-closeout-20260930 | supporting | no experiment; rebuild canonical evidence | unified CSV/JSON/report | experiments frozen | `results/final_evidence_bundle.json` |

## Corrections

- 2026-09-30: supersedes stale progress text that described few-shot as running. FD001–FD003 completed; FD004 was deferred under the stopping rule.
- 2026-09-30: adds the completed four-dataset w24 main experiment and distinguishes FD002 w48 seven-seed headline (28.295) from the common-five paired subset (26.792).

## `core-a-fewshot-fd001-w48-20260928`

- Status: completed; core Candidate A minimum viable run.
- Controlled factors: public preprocessing/model, w48, T=1000 linear DDPM, Adam lr2e-3, 70 epochs, five generation/evaluator seeds.
- RMSE: 27.336 → 5.219 → +22.117 → beyond local variation → changes the paper conclusion.
- Conclusion: FD001 few-shot numerical result not reproduced.

## `core-a-fewshot-full-w48-20260928`

- Status: stopped after completed FD001–FD003; FD004 has no accepted result.
- RMSE FD002: 54.044 → 24.758 → +29.286 → beyond variation → conclusion impact yes.
- RMSE FD003: 24.509 → 9.244 → +15.265 → beyond variation → conclusion impact yes.
- Stopping rule: three independent datasets agree; further FD004 compute has low discriminating value. Candidate A remainder deferred.

## `main-w24-fd003-fd004-20260929`

- Status: completed; main Table I window-24 matrix.
- Single changed factor from w48 baseline: sequence/window length 48 to 24. Model, optimizer, epochs, seeds and evaluation protocol frozen.
- Four-dataset mean RMSE: 23.646 → 22.318 → +1.328 (+5.95%) → heterogeneous across datasets → partial numerical reproduction.
- Four-dataset mean DS: 0.208 → 0.154 → +0.055 → FD003/FD004 remain high → partial fidelity reproduction.
- Window trend: FD001 DS and FD004 RMSE significantly favor w24 contrary to paper direction; overall trend only partially reproduced.

## `evidence-closeout-20260930`

- Status: completed; supporting audit, no model run.
- Inputs: formal summary/raw CSVs, complete evaluation manifests, Table I/VI provenance JSONs.
- Outputs: `results/final_main_results.csv`, `results/final_window_paired.csv`, `results/final_fewshot_results.csv`, `results/final_evidence_bundle.json`, and the canonical final report.
- Decision: freeze experiment line. w96 and formal ablations remain optional future work, not missing executed results.
