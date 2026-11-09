# FD001固定检查点重采样诊断

- Run ID: `diagnostic-fd001-checkpoint-resampling-20261029`
- Scope: diagnostic/supporting; user explicitly requested 2 checkpoints × 3 new sampling seeds × 5 discriminator evaluations. Not a strict paper reproduction experiment or a new training intervention.
- Purpose: separate persistent checkpoint-associated differences, within-checkpoint sampling differences, and within-batch discriminator-evaluation differences. No generator training or downstream predictor training.

## Frozen selection and controls

[Observed] Select historical FD001 full-data w48 generator seed13 (lowest mean DS, 0.212708) and seed23 (highest mean DS, 0.422749), by the existing five-evaluator means. This selection uses historical discriminator outcomes and is exploratory; two selected checkpoints cannot characterize all training runs. Metadata records separate 70-epoch training runs. File hashes, state-dictionary tensor hashes, histories, best epochs and different tensor counts must be recorded before sampling.

[Code] Each checkpoint remains fixed. Use original `TrainCondition.sample`, original `UNet1D_fre`, hard_random_quantile frequency mode and stored frequency parameters. Preserve dim32, dim_mults(1,2), cond_drop_prob0.2, 14 channels, window48; DDPM1000 steps, linear beta schedule, pred_noise objective, cond_scale3, original clipping and final normalization, batch256 (last batch59). No EMA, accelerated sampler, rescaling or threshold override.

[Observed] Sampling seeds103/113/123 are fixed in advance and identical for both checkpoints. Reset NumPy and PyTorch CPU/CUDA RNGs using the existing `utils.torch_seed` before constructing the sampling model; keep original cudnn benchmark setting. Record pre-construction RNG hashes and generated tensor hashes. These seeds differ from the historical training seed; original sampling continued after training and its RNG state is not reconstructed here. Only progress display may be suppressed; retain sampling math.

[Observed] Each batch contains all15,931 windows, shape(15931,48,14), with exactly the original FD001 training-label vector/order (RUL capped125). Real discriminator reference and original fixed80/20 real/generated index split are shared across all six batches. Frozen source inputs are saved and hashed. No label-condition composition, count or split changes.

[Code] Every batch receives evaluator seeds3/13/23/33/43 using the existing `discriminative_score_metrics`: TensorFlow compat-v1 GRU(hidden7), Adam defaults, 3000updates, batch128, same fixed split. Preserve TF_DETERMINISTIC_OPS=1, TF_ENABLE_ONEDNN_OPTS=0 and one intra/inter-op thread as original evaluator. Existing TF2.12 cannot register a GPU on this host; CPU backend is expected, matching historical evaluation environment. Record actual framework/backend versions. Fresh discriminator training is evaluation, not generator retraining.

## Outputs and interpretation

[Observed] Save six generated NPZs, sample/RNG/timing/checkpoint records, 30 raw DS/fake-accuracy/real-accuracy rows, per-batch mean/SD/min/max, checkpoint summaries and paired checkpoint differences at matched sampling/evaluator seeds. Use mean of five DS values for batch classification with user threshold strictly >0.3. Keep individual evaluator values and plot them to distinguish mean changes from evaluator spread. No rounding before comparison.

[Hypothesis] If the historical poorer checkpoint is >0.3 on all3 new batches, historical better is <=0.3 on all3, and all3 matched sampling-seed mean contrasts retain the historical ordering, prioritize checkpoint/training differences. This does not establish whether training initialization, training dynamics, stored frequency thresholds or best-loss checkpoint selection is causal. No generator training occurs in this diagnostic.

[Hypothesis] Within-checkpoint sampling-mean ranges, paired seed contrasts and within-batch evaluator SD/ranges indicate the scale of sampling and evaluator sensitivity. Report raw ranges and SDs; do not invent a post-hoc significance threshold, treat correlated shared evaluator seeds as independent data, or claim a formal variance-component decomposition. The three patterns may coexist. A threshold crossing alone need not be a large effect. Three sampling seeds are a small exploratory sample.

## Audit and stopping rule

[Observed] Require distinct trained checkpoints, exact condition/count/split identity, finite generated data, unchanged checkpoint/source hashes, complete6×5 seed grid with no duplicates, DS reconstructed from equal-class fake/real accuracies and independently recomputed aggregates. Save software/config/source/weight/data hashes, command, Git state/diff, timestamps and per-worker logs before interpreting results. Original output/checkpoint files are never overwritten.

Budget: exactly six full sampling runs and30 discriminator evaluations. Stop after this diagnostic, or on a failed worker with failure recorded. No automatic generator training, extra seeds, different sample sizes, algorithm changes, evaluation tuning or downstream testing. Existing strict-reproduction conclusions, zero-label pause and EXT-002 stop decision remain unchanged. A supervisor writes progress and automatically summarizes completed results; any incomplete run is reported as incomplete.
