# FD003 high-error real-neighbor diagnostic (2026-10-27)

Scope diagnostic/supporting; user asks whether high-error FD003 test windows
have similar real training windows. No training, modifications to data, or
test-informed anchor selection. One descriptive audit, stop after the report.

- Examine all21 real test last windows with cappedRUL31–60. Focus table uses
  top5 errors of the existing25synthetic-trained predictors; also retain engine75
  from previous local-support discussions. No new error-based model selection.
- Main candidate pool: original predictor TRAIN indices only (not dev), real
  windows with RUL±5 and within31–60. Also give all-real-training-window pool
  sensitivity because generator training included both predictortrain/dev.
- Reuse existing level28, dynamics42 and rawcoupling91 descriptors. Standardize
  level/dynamics on all original real training windows; raw correlations unscaled.
  Distances are RMS coordinate differences. Joint representation balances level
  and dynamics: sqrt((d_level²+d_dynamics²)/2). Do not tune weights on results.
- Calibrate each test window's distance using the same-exact-RUL windows of100
  real training engines, each queried against the predictortrain pool with its
  OWN engine excluded. RUL±5 and31–60 restriction remains. Report empirical
  percentile and95th percentile reference; above95 is a descriptive tail flag,
  not a significance test or proof of no comparable physical degradation.
- Compute normal5window-NN and5DISTINCT-engine-NN (one closest window per engine)
  to expose overlapping-window concentration. Test/train numericengineIDs are
  different namespaces and must not be excluded merely for matching numbers.
- Save nearest realengine/window indices, RUL and raw cycle boundaries. At joint
  neighbors report level/dynamics/coupling distances so a level-only match is not
  mistaken for simultaneous matching of signal variation.
- Additional query without RUL restriction against actualtrain: give nearest
  joint-neighbor RUL, descriptive only; cannot establish an identifiable RUL
  mapping from one nearest neighbor.
- Associate saved real-trained and synthetic-trained per-engine errors as context,
  not as a causal experiment. No new predictor inference required.
- The original randomwindow train/dev overlap within engines; leave-engine-out
  calibration is only a descriptor reference. Distances do not certify equal
  failure modes, causal mechanisms or raw-waveform equivalence. Officialtest is
  used for diagnosis only; never use these neighbors to select training anchors.
- No paper target or numerical-reproduction claim. Strictmain/A/B unchanged,
  C deferred and zero-label route remains paused.
