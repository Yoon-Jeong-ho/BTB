# Run Summary

## 1. Problem

- Task: Large-scale multiclass tabular classification on Covertype.
- Track / Stage: 01_ml / 04 Large Scale Tabular
- Why this run exists: Compare accuracy-cost tradeoffs before moving larger tabular workloads to a dedicated server workflow.

## 2. Hypothesis

- What changed from the previous run: Added explicit large-scale baselines plus a GPU MLP on CUDA device 0.
- Expected effect: HistGradientBoosting should provide the strongest CPU baseline, while the GPU MLP gives a fast nonlinear reference and sample-efficiency curve.

## 3. Dataset

- Dataset name: covertype
- Split version: train=406708, valid=87152, test=87152
- Preprocessing: bool -> int, imputation, scaling for linear/MLP paths.
- Data caveats: Class imbalance and one-hot-like soil / wilderness flags create class-specific recall gaps.

## 4. Training Setup

- Model: SGD linear / shallow random forest / hist GBDT / GPU MLP
- Seed: 42
- Batch size: 2048 for GPU MLP
- Epochs / steps: 10 epochs for main GPU MLP, 6 epochs for sampling study
- Hardware: cuda NVIDIA RTX A6000

## 5. Best Metrics

| Model | Macro F1 | Accuracy | Recall | Fit sec |
| --- | --- | --- | --- | --- |
| hist_gbdt | 0.798 | 0.837 | 0.776 | 6.6 |
| gpu_mlp | 0.737 | 0.838 | 0.705 | 29.0 |
| shallow_tree | 0.639 | 0.780 | 0.581 | 3.3 |
| sgd_linear | 0.458 | 0.709 | 0.445 | 2.0 |

## 6. Result Figures

- `figures/results/class_distribution.svg`, `metric_vs_training_time.svg`, `metric_vs_memory.svg`, `score_distribution.svg`
- Why these matter: They show dataset scale, cost-quality tradeoffs, and confidence spread.

## 7. Analysis Figures

- `figures/analysis/slice_metric_by_class.svg`, `throughput_bottleneck_summary.svg`, `sampling_strategy_performance.svg`
- Key failure patterns: Minority forest-cover types remain hardest, and throughput strongly favors the GPU MLP as sample size increases.

## 8. Sample Predictions

- Test predictions: `predictions/test_predictions_sample.csv`
- Failure examples: class-wise recall slices highlight where mistakes concentrate.

## 9. Decision

- 대표 artifact로 남길까? yes
- Promote weights to `artifacts/promoted/` or HF Hub? no
- Next run: compare against XGBoost/LightGBM GPU variants once those dependencies are explicitly approved.
