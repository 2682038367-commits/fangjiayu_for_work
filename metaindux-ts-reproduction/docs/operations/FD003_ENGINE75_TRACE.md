# FD003 engine75 trace: frozen diagnostic protocol

Run `diagnostic-fd003-engine75-trace-20261027`, diagnostic/supporting. User
explicitly requested test75→similarrealtrain→same-lifegenerated→bothpredictors,
including both predictors on the SAME real neighbor windows. No training.

## Selection before predictions

- Official real test engine75 last48cycles144–191, terminalRUL45.
- Primary real batch: the5 distinct-engine joint neighbors from the completed
  real-neighbor audit: trainwindow368/12592/7213/8520/9855, realengines2/60/33/38/45,
  RUL49/40/43/40/46. No replacements based on their predictions.
- Sensitivity realdev batch:5distinct-engine nearest joint windows in original
  realdev, RUL40–50; select without looking at outputs. Realdev randomwindow
  split still overlaps training engines and adjacent windows, NOT engine holdout.
- For test75 and each of5primaryreal neighbors, compare actualpredictortrain
  real and generated candidate pools at each query's RUL±5 within31–60. Exclude
  the real query's own training engine from the realreference. Apply the SAME
  index exclusions to generated arrays ONLY to match count/RUL composition,
  not to assign physical engine identities to generated samples.
- Repeat exactRUL as a fixed sensitivity, not a parameter sweep.
- Representations: existing real-fit level28, dynamics42, equal-familyjoint,
  rawcoupling91. Additional centeredshape: subtract per-window sensor means,
  divide each sensor by its within-window residual RMS over realtrainwindows,
  flatten48×14 and take RMS distance. This retains amplitude/phase variation
  while removing sensor mean levels; noise alignment can still dominate and
  shape similarity is not proof of physical/failure equivalence.
- For each query/representation/gen seed, save nearest5 real/generated windows,
  distances and number of generated candidates inside the real5th-neighbor
  radius. Equal candidate counts and exact label composition in each pair.
  No universal distance threshold or forced mutuallyexclusive diagnosis.
- Sensor-level offsets: on centeredshape-selected generated neighbors, compare
  means/terminal levels against the query and shape-selected realreference.
  Save per-generation signed residuals in real descriptorSD units. Label them
  LOCAL matched-neighbor offsets, not population-level systematic bias.
- Freeze/hash all selections BEFORE reading per-window predictor outputs or
  running the real predictors on selected generated samples.

## Prediction comparisons

- Use existing headline5generationseeds×5evaluatorseeds and realbaseline5.
  Reuse hashed per-sample caches for realqueries and corresponding-generated
  predictions. New CPU inference ONLY for realbaseline5 on selected generated
  windows, plus replay check on realtest75. No optimizer/checkpoint selection.
- Both predictor families evaluate the identical test/realneighbor labels.
  Report mean prediction AND sqrt(mean per-model squared errors), not ensemble
  error; summarize per-window and pooled batch. Also report same-count5vs5 for
  each generationseed, never treat repeated realpredictions as25independentfits.
- On generated samples, use that generated dataset's own5synthetic-trained
  predictors versus realbaseline5. Primary generated candidates are train
  samples seen by the corresponding synthetic predictor. Select additional
  nearest5 generateddev joint samples for every primary query to check this
  optimism; both families predict these same samples and their actual labels.
- Realtrain neighbors were direct training inputs for realbaseline; cannot
  interpret their gap as out-of-sample superiority. Realdev sensitivity is only
  partial protection because of overlapping windows/engines.

## Interpretation and stop

Consider jointly (a) local density loss even after centering,(b) centeredshape
exists but level offsets persist,(c) close generated support plus prediction
failure. Multiple explanations can coexist; similar mean/dynamics does not prove
RUL-relevant equivalence. Use both-family real-neighbor outputs to see whether
error is local to75 or extends over this real region. Do not claim prototypes
help without an intervention or use selected test neighbors to build training.

Stop after this one case's5generationseed trace and report. No new generator,
predictor, prototype, anchor selection or lossweight experiments. Strictmain/A/B
unchanged,Cdeferred,zero-label route paused. No paper diagnostic targets.
