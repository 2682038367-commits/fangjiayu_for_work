# Frozen FD003 prototype diagnosis and two-ridge probe

Diagnostic/supporting, scope-20261103-fd003-prototype-probe-v1. User explicitly authorized two CPU Ridge fits and read-only reuse of FD003 parent clustering. No generator training, generation, K change, reclustering or original-test access.

Before outcome analysis freeze configs/diagnostics/fd003_prototype_probe.json. Parent is completed ext-prototype-fd003-w48-20261103, original engine split70/30. Verify parent source/input hashes. TrainingNPZ includes engine_id/end_cycle; validationNPZ lacks end_cycle and explicit window index. Reconstruct both from original train_FD003.txt using frozen MinMax, compare arrays/labels/order, save engine+start/end cycle+raw row indices and parent array indices. Do not invent generated-engine IDs.

Actual clustering inputs: each14sensor mean/std(ddof0)/(last12mean−first12mean),42features. Frozen StandardScaler and nearest4centers assign real train/validation IDs. RUL is only used in parent frequency tables for generation, not KMeans IDs; NEVER call draw_prototypes for the probe. Verify training labels against stored prototype_id. No RUL/engine/stage/frequency columns in either probe design matrix.

Count all bins plus31-40/41-50/51-60, including zeros, window proportions and distinct engines (overlap allowed across prototypes).125 is capped, kept separate from91-124. Compute RUL-prototype association descriptively; not independent-window significance.

Shape features from quarter means Q1..Q4 and14sensor increments D1=Q2−Q1,D2=Q3−Q2,D3=Q4−Q3:
1. middle_energy_fraction=sum_s D2² / sum_{s,j} Dj².
2. peak_middle=1 if argmax_j sum_s Dj² is middle transition, otherwise0.
3. sensor_peak_synchrony=max_j fraction_s[argmax_k |Dks|=j].
They are not the direct clustering inputs; correlations with those inputs remain possible, so differences do not demonstrate fault types or new causal degradation information. Their timing/relative-energy construction avoids claiming mean-level differences as independent evidence.

Primary shape comparison restricted31–60. For each pair and domain, same integer RUL, different engines; only RUL strata with>=3engines per prototype enter. At each RUL match min(countA,countB) windows, prioritizing least-used engines with engineID ties; no use of curve features in selection, no replacement within pair. Missing common support marked insufficient. No ±5 fallback (allowed by user but not needed to force results). Report actual matched RUL distributions and all engine counts. Require>=8engines each side,>=30pairs and>=5commonRUL to discuss replication. Within each RUL compute B−A feature differences then average RUL strata equally. Shared engine exponential bootstrap weights reused across windows/RUL,2000replicates/seed20261103, descriptive95percentileCI. Repeat same method on held-out engines. Same sign and bothCIs excluding0 constitutes descriptive repeated-pattern evidence only, not multiplicity-adjusted confirmation. All3features×allpairs reported. Sparse subranges remain insufficient even if the broad31–60pair passes.

Plots fixed before outcomes: raw and per-window standardized quarter curves at exactRUL45, sensorss2/s11/s20, all available engines by prototype; median and25-75percentile curves plus engine counts. No chosen eye-catching windows; raw level plots are context only, not independent evidence. Plot matched noncluster descriptors separately.

Probe exactly2fits, sklearnRidge(alpha1,solverSVD,fit_intercept=True), all14886trainingwindows/allRUL incl125. A uses42features standardized by frozen train clustering statistics; B concatenates same42+4unscaledonehot. Alpha1 fixed once (default scale, no search); same normalization, target and samples; raw RULcycle prediction/no clipping. Predict all5134held-out windows; only afterward bin by actual labels. Persist bothmodels/coefficients/predictions and design provenance.

Metric: per-engine windowMSE then equal-average engines then sqrt. Report overall and31–60 plus allbins, individual engine changes, number improved, top1/top3positive-improvement shares and leave-top3-out sensitivity. Resample engines, not windows, for descriptive paired RMSE intervals. Improvement is expressiveness of nonlinear partition features for one regularized linear model, not new sensor information. The prior diffusion negative result remains unchanged regardless of probe result. No tuning/extra models/followup runs after seeing results.

Run from workspace root:
`metaindux-ts-reproduction/.venv/bin/python metaindux-ts-reproduction/analysis/diagnostics/diagnose_fd003_prototypes.py`
Outputs: runs/diagnostic-fd003-prototype-probe-20261103 snapshot and results/diagnostics/fd003_prototype_probe_20261103/ tables/figures. Report docs/conclusions/final/fd003_prototype_probe_20261103_report.md. Entry refuses to overwrite existing manifest; reproduce under a new run suffix with --suffix NAME, same config/data/source hashes and exactlytwofits.

Execution environment: use existing repository .venv (NumPy1.23.5/sklearn1.4.1.post1/matplotlib3.8.3). Workspace-root .venv lacked matplotlib and failed at import before any analysis/model fit; preserved preflight log, no install or scientific changes. All diagnostics and two probes use the same selected environment; replay records it.
