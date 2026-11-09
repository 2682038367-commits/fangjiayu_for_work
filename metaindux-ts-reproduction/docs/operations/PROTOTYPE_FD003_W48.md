# FD003 w48 prototype guidance: frozen extension protocol

Run: ext-prototype-fd003-w48-20261103. Scope: extension:EXT-PROTO-002/supporting, contract scope-20261103-prototype-fd003-v1. Explicit user authorization: next round changes dataset only, retains4-cluster feature/injection design, engine train/validation split, two paired generator seeds, overall primary and predeclared31–60 secondary. No extra approval required; no automatic followup expansion.

## Question and source

[Hypothesis] Does the same minimal train-window clustering prototype condition improve downstream prediction utility on FD003? This is a dataset transfer of the prior FD001 intervention, not K-ProtoDiff reproduction; cluster IDs are not known fault labels.

[Observed] Prior test analysis `docs/conclusions/final/rul_segment_error_diagnostic_w48.md` found FD00331–60 pooled generated-train predictor RMSE29.466 versus real-train20.514 and50.7% of excessMSE; this motivates dataset/stage choice only. No original test files, labels or predictions enter this run; no prototype/epoch choices based on them. Since dataset/stage choice is test-inspired, this is exploratory, not a new untouched-test claim. Prior strict conclusions unchanged.

## Single changed factor and controls

[Code] Relative to `ext-prototype-fd001-w48-20261103`, only dataset and its pre-existing engine split change. Reuse FD003 split from `runs/ext-fd003-rul-weight-screen-20261027/config.json`: RandomState20261027 first30 validation, remaining70 train. Old generator/scaler from that screen saw validation; none reused. Both A(original) and B(original+prototype) are trained from scratch. All subsequent generator/MinMax/prototype fits see only the70 train engines. Validation has5134 windows/30engines, train14886windows/70engines; stage31–60 has900validation windows/all30engines.

Prototype source `upstream/prototype_guidance.py` remains byte-identical to the completed FD001 snapshot. Preflight asserts baseline config equality except runID/engine IDs and AST equality of seed/diffusion/check/train_generator/sample_generator/downstream, and parent upstream source hashes. It is not merely a design description: code parity is asserted before every run.

- Same14sensors,48window,125cap, continuous raw-cycle RUL. Train-only MinMax; features per sensor mean/std(ddof0)/last12mean−first12mean,42dimensions standardized on train windows; KMeans4/n_init10/seed20261103.
- B adds learned4×32 embedding and Linear→SiLU→Linear to existing classes_emb output, followed by original shared CFG mask/MLP. Empty condition replaces entire joint embedding with original null_classes_emb; no prototype in null branch. Original training .2 condition dropout and .1 numericRULbatchzero remain in both groups. No loss weighting, anchors, fault-type labels, richer features, new embeddings or K search.
- All5 fixed RUL bins0–30/31–60/61–90/91–124/125; frequencies only from training windows. No empty realtrain bins; explicit global-frequency fallback retained and tested. Prototype draws have independent NumPy stream7103; fixed for each window through all1000steps, no reference windows at sampling.
- Generator seeds23 then13; train A/B from same shared initialization, same sample order and diffusion RNG;70epochs/batch256/Adam.002/clip1/warmup8 multiplier2.5+cosine. Original frequency module and pred_noise MSE retained. Fixed cuDNN benchmark=False/deterministic=True; global strict=False. Each arm4130updates. Epoch64 and70 saved in advance, no lowest-training-loss candidate.
- Each checkpoint generates14886samples with exact same RUL list, batch256/DDPM1000/linear/CFG3/sample seed103; all candidate runs reset noise stream. Same sample-per-training-window count rule as FD001: count/update numbers differ across datasets as derived from dataset size, never differ across A/B. Both conditional sampling branches have drop0 and explicit null drop1.
- Five LSTM predictor seeds3/13/23/33/43, original architecture14→28singleLSTM/flatten64ReLU→1, same initialization/order, synthetic70/30split(seed20261030,10420train/4466dev), batch256/Adam.002/original15,25,35decay,35epochs1435updates each. Same syntheticdev checkpoint selection after zero-basedepoch5, same evaluation cadence; patience disabled symmetrically exactly as FD001. Real validation never used for predictor gradient/earlystop/dev selection.
- Budget4generator fits/8sample sets/40predictor fits. Compare all four seed×epoch cells, and every predictor seed pair. Do not run DS or condition-readout fits as substitutes for primary utility.

## Metrics, selection, decision (fixed before results)

Primary: overall internal-real-validation engine-balanced pooledRMSE = sqrt(mean_predictors(mean_engines(mean_window_squared_error))). Predeclared secondary:31–60 with same engine weighting within that bin. Also save all RUL bins, engine counts, per-engine predictions/MSE and window-weighted diagnostics.

Checkpoint selection: independently for A/B, choose minimum overall pooled realvalidation RMSE among64/70, tie earlier epoch; no31–60 or other statistic used in selection. Report both candidates and selected overall AND selected31–60, even if A/B choose different epochs. These are exploratory selection-set scores, not independent test gains.

Expansion gate: only consider recommending further validation when BOTH generator seeds have lower selected overall RMSE AND lower selected31–60 RMSE in B. Also show each fixed-round direction and5paired-predictor improvement counts, so candidate sensitivity is visible. Gate is descriptive, not significance. Stage-only improvement with overall worsening is explicitly a tradeoff; overall-only or one-seed gains are insufficient. A passed gate does not authorize additional runs, changed design, more K/seeds/datasets. No automatic shuffle ablation.

## Execution and artifacts

From `/home/cw_boe/projects/rul`:

```bash
# Already prepared; preserve snapshot, do not re-prepare
.venv/bin/python metaindux-ts-reproduction/scripts/run_prototype_guidance_fd003.py check
# Must run with GPU exposed (current sandbox hides /dev/nvidia*)
.venv/bin/python -u metaindux-ts-reproduction/scripts/run_prototype_guidance_fd003.py run > metaindux-ts-reproduction/logs/prototype_fd003_full.log 2>&1
# Rebuild finished-cell overall/secondary tables only
.venv/bin/python metaindux-ts-reproduction/scripts/run_prototype_guidance_fd003.py summarize
```

Config: configs/extensions/prototype_fd003_w48.json. Inputs/source/config/environment/launch: runs/ext-prototype-fd003-w48-20261103/. Models/data/predictions: results/extensions/ext-prototype-fd003-w48-20261103/seed{23,13}/{A,B}/. Top-level summary.csv/paired.csv and segment_summary.csv/paired_segments.csv/target31_60_summary.csv/selected_comparison.json. Upon completion, automatic CPU audit recomputes all40predictors/allRULbin metrics, verifies shared hashes/orders/updates, and writes completion_audit.json, expansion_gate.json and final report docs/conclusions/final/prototype_fd003_w48_20261103_report.md; manifest/ledger/state updated.

Stop on source/input mismatch, shared init/order/RNG or budget mismatch, nonfinite samples, audit failure, or GPU failure; preserve partial artifacts. Do not silently resume partial training or expand budget. Original FD001 source/config/results preserved.
