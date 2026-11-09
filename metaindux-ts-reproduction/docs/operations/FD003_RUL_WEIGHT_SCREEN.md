# EXT-002: fixed-generator FD003 RUL-weight screening

Run `ext-fd003-rul-weight-screen-20261027`; exploratory extension/supporting.
User explicitly authorizes31–60sample weighting,5generationseeds, paired training
budget/initialization, engine-grouped INTERNALvalidation and honest test-inspired
scope. User further confirmed fixed-generator DOWNSTREAM-only validation despite
the generator having seen every real training engine. No officialtest use here.

## Frozen configuration

- Existing original FD003 w48 synthetic datasets, generationseeds3/13/23/33/43,
  unchanged generator/data/preprocessing. Original predictortrain14014/dev6006
  indices fixed. Generated windows have no reliable physical engine identity;
  do NOT infer engine isolation from their array indices.
- Split real original100training engines by RandomState(20261027).permutation:
  first30 are internal realvalidation, remaining70 are not used for weight choice.
  Use all48step windows of those30engines, cappedRUL125. Predictors fit only
  synthetic inputs. Original generator and MinMax scaler saw ALL100 engines;
  this is NOT generator-independent or whole-pipeline leakage-free validation.
- Architecture: original LSTM14→28hidden,48steps, flatten→64ReLU→1output,
  1layer. Adam lr.002, betas(.9,.999),eps1e-8,weight_decay0; original multistep
  decayx.1 at zero-basedepochs15/25/35. Batch256,35epochs,1925updates/model.
- Primary checkpoints are FINALepoch35 weights, not variable-step bestdev;
  both arms trained from scratch under this matched protocol. Historical
  bestdev/earlystop headline metrics are not the causal control for this run.
- One predictorinitialization seed3, SAME for all15fits; fixed35epoch permutations
  shared by every arm and generationseed, saved before running. Check initial
  parameterSHA256 and actual consumed sample-orderSHA256 across models.
  This is a screen, not initialization-seed robustness validation.
- Candidate weights1(control),2,4; no additional weights/ratios after results.
  Total15fits=5datasets×3weights. No realanchors or other changes.
- Loss: L=mean_b[w(y_b)*(pred_b-y_b)^2]/mean_train[w(y)]. Weight w applies ONLY
  to31≤RUL≤60, elsewhere1. Fixed full-train normalization holds average weight1
  and avoids changing nominal loss scale merely by multiplying all losses.
  No resampling or samplecount/RULhistogram changes.
- CUDA deterministic/cuDNN deterministic flags and CUBLAS config frozen.
  Evaluate unweighted syntheticdev once/epoch for diagnostics, never use real
  validation for gradient updates, earlystop or epochselection. Exactly35epochs
  for every candidate. Evaluate final models on synthetictrain/dev and realval.

## Validation and selection rule, before execution

- Main realvalidation score: compute MSE within each engine then equal-average
  over engines, then sqrt. Overlapping windows are not independent replicates.
  Also retain native window-weighted results. Five RULbins0–30,31–60,61–90,
  91–124,125 plusoverall; withinbin equal-average available engines and state
  their counts. These are life-spanning internal windows, not test-endpoint RMSE.
- Per candidate first pool MSE over5generationseeds, then sqrt. Also give all5
  paired per-generation results/deltas, not only best generatorseed.
- Exploratory eligible candidate: (1)overall engine-balanced RMSE lower than
  weight1, (2)31–60RMSE lower, (3)targetstage improves in at least3/5generation
  seeds, (4)each OTHERbin meanRMSE worsens by no more than5%. This5% is a frozen
  screening tolerance, not a statistical significance threshold.
- If eligible, choose smallest overall realvalidationMSE; exactties prefer lower
  weight. If noneeligible, retainweight1. Report all tradeoffs even if a candidate
  fails. Recommendation remains conditional/exploratory, never a globaldefault
  adoption or claim of fully independent validation.
- No officialtest files/labels/predictions loaded during this run or selection.
  Stage31–60 was inspired by prior officialtest results; document this in every
  report and any future officialtest comparison. No more weight search here.

## Audit and stopping

Hash15run configs,source,5synthetic datasets,split,realtrainfile,validationcache,
frozenorders before training. Keep15final weights, predictions, histories,
per-engine/bin errors and selectionaudit. Validate finiteoutputs/labels/counts,
all1925updates and490490samplevisits, identicalinit/order in allfits, normalized
meanweights1, pairedset equality. Stop15fit budget;no extra initializer/weight
scans, generatortraining, prototype or new testresult. Strictmain/A/B unchanged,
Cdeferred,zero-label repairpaused. EXT-001anchorwork remains frozen.
