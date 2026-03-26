# Run Summary

## 1. Problem

- Task: Time-aware count regression for bike rental demand.
- Track / Stage: 01_ml / 03 Model Selection And Interpretation
- Why this run exists: Practice validation strategy, leakage prevention, and parameter selection with a held-out future window.

## 2. Hypothesis

- What changed from the previous run: Introduced TimeSeriesSplit and tuned HistGradientBoosting over a Poisson baseline.
- Expected effect: Time-aware tuning should improve RMSE on the final 20% future window and expose where seasonal/weather slices remain difficult.

## 3. Dataset

- Dataset name: bike-sharing-hourly
- Split version: first 80% train/CV, last 20% test
- Preprocessing: dropped leakage columns `casual`, `registered`, `cnt`, `instant`; added cyclic weekday/hour features.
- Data caveats: Extreme demand spikes during commuting peaks and bad weather cause heavy-tailed residuals.

## 4. Training Setup

- Model: Poisson baseline / tuned HistGradientBoosting / GPU MLP comparator
- Seed: 42
- Batch size: 1024 for GPU MLP
- Epochs / steps: 16 epochs for GPU MLP
- Hardware: cuda NVIDIA RTX A6000

## 5. Best Metrics

| Model | RMSE | MAE | R2 |
| --- | --- | --- | --- |
| tuned_hist_gbdt | 60.052 | 38.159 | 0.926 |
| poisson_baseline | 163.181 | 120.111 | 0.452 |
| gpu_mlp | 164.316 | 109.021 | 0.445 |

## 6. Result Figures

- `figures/results/cv_fold_score_boxplot.svg`, `validation_curve.svg`, `top_feature_importance.svg`
- Why these matter: They justify the chosen validation strategy and tuned parameter range.

## 7. Analysis Figures

- `figures/analysis/subgroup_metric_comparison.svg`, `confidence_bin_plot.svg`, `common_failure_slice_summary.svg`
- Key failure patterns: High predicted-demand bins and adverse weather / non-working-day combinations remain hardest.

## 8. Sample Predictions

- Failure examples: `predictions/worst_predictions.csv`
- Best examples: low-error hours embedded in the test split.

## 9. Decision

- Promote to `reports/`? yes
- Promote weights to `artifacts/promoted/` or HF Hub? no
- Next run: explicit holiday features and fold-aware hyperparameter search expansion.
