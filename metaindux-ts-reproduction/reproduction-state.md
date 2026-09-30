# Persistent reproduction state

## Identity and goal

- Paper: MetaIndux-TS: Frequency-Aware AIGC Foundation Model for Industrial Time Series, TNNLS 2025.
- Workspace: `/home/cw_boe/projects/rul/metaindux-ts-reproduction`.
- Type: official-author-code execution with paper-protocol completion; not an independent reimplementation.
- Current state: experiments frozen; canonical evidence package and final report generated on 2026-09-30.
- Route: public-code mainline for headline results; paper settings take precedence only where explicitly documented.

## Main/core priority

| ID | [Paper] claim | Priority | Experiment | Final status |
|---|---|---|---|---|
| C1 | Frequency-aware generation fidelity and utility, Table I | primary | Main w24/w48 | partial numerical/trend reproduction |
| C2 | Zero/few-shot generation, Tables V–VI | co-primary | Candidate A Table VI | FD001–FD003 not reproduced; FD004 deferred by stopping rule |
| C3 | Dual frequency learners and contrastive synthesis explain gains, Tables III–IV | secondary | Candidate C ablation | deferred / mechanism inconclusive |

- Main: Table I public-code w24 and w48, four datasets, complete.
- Candidate A relationship to main: shared-base but outcome-independent.
- Post-main core gate: resolved as **defer remaining Candidate A execution** after three independent numerical misses.
- Candidate B zero-shot and Candidate C ablation remain documented alternatives.

## Current stage

- Phase 4 complete: formal evaluation, evidence audit, and final report.
- Canonical report: `docs/conclusions/final/REPRODUCTION_FINAL_REPORT.md`.
- Canonical machine-readable evidence: `results/final_evidence_bundle.json`.
- No experiment service is running. No additional tuning is authorized or recommended.

## Configuration fidelity

| Item | Selected requirement | Actual | Status |
|---|---|---|---|
| Sensors/preprocessing | public code | 14 sensors, train-fit MinMax, RUL cap 125 | exact [Code] |
| Diffusion | T=1000, linear, DDPM | same | exact [Paper] |
| Training | Adam, lr 2e-3, 70 epochs | same | exact [Paper] |
| Repeats | five; IDs unpublished | 3/13/23/33/43 | count exact; IDs local |
| Frequency mask | paper threshold learning vs code behavior | hard_random_quantile headline | explicit paper-code difference |
| Predictor | 70/30 synthetic, real test | fixed split 20260915; 5 evaluator seeds | paper-aligned |

## Run ledger

| Run ID | Snapshot | Key result | Status |
|---|---|---|---|
| main-w48-public-code-final | `runs/main-w48-public-code-final/run-manifest.yaml` | w48 Table I values not fully reproduced | complete |
| main-w24-fd003-fd004-20260929 | `runs/main-w24-fd003-fd004-20260929/run-manifest.yaml` | four-dataset average RMSE 23.646 vs 22.318 | complete; partial reproduction |
| core-a-fewshot-full-w48-20260928 | `runs/core-a-fewshot-full-w48-20260928/run-manifest.yaml` + `final-status-20260930.yaml` | FD001 27.336, FD002 54.044, FD003 24.509; all miss paper | stopped after sufficient cross-dataset evidence |

## Hypothesis and stopping status

| [Hypothesis] | Evidence | Decision |
|---|---|---|
| Public-code mainline can match Table I | w24 partly matches; w48 systematic gaps remain | partial support; stop tuning |
| Strong few-shot benefit is reproduced | three independent large RMSE misses | falsified in tested scope; defer FD004 |
| w24 is intrinsically superior | length also changes sample/update counts; mixed paired directions | inconclusive; no causal claim |

## Session checkpoint

- [Observed] w24 and w48 main experiments are complete and auditable through formal manifests/raw CSVs.
- [Observed] few-shot FD001–FD003 are complete; FD004 is explicitly deferred, not reported as executed.
- [Observed] final numbers are generated from CSV by `analysis/audit/gen_complete_reproduction_report.py`.
- Overall status: **partial reproduction**; numerical, trend, and mechanism dimensions are reported separately.
- Recommended next state: freeze and deliver. Optional future branches are w96 for full scaling or formal ablations for mechanism.
