# Run Summary

## 1. Problem

- Task: Tabular regression for California housing prices.
- Track / Stage: 01_ml / 02 Tabular Regression
- Why this run exists: Practice baseline comparison, residual analysis, and feature interpretation with a lightweight GPU reference.

## 2. Hypothesis

- What changed from the previous run: Initial regression suite for the ML track.
- Expected effect: Tree-based models should improve RMSE over linear baselines while the GPU MLP provides a non-linear comparator.

## 3. Dataset

- Dataset name: california-housing
- Split version: train=14448, valid=3096, test=3096
- Preprocessing: median imputation and scaling for linear / MLP models.
- Data caveats: Target is capped and the coastal geography introduces structured residual patterns.

## 4. Training Setup

- Model: dummy / linear regression / ridge / random forest / hist GBDT / GPU MLP
- Seed: 42
- Batch size: 768 for GPU MLP
- Learning rate: 2e-3 for GPU MLP
- Epochs / steps: 18 epochs for GPU MLP
- Hardware: cuda NVIDIA RTX A6000

## 5. Best Metrics

| Model | RMSE | MAE | R2 | Fit sec |
| --- | --- | --- | --- | --- |
| hist_gbdt | 0.472 | 0.318 | 0.830 | 1.5 |
| random_forest | 0.515 | 0.336 | 0.798 | 0.7 |
| gpu_mlp | 0.585 | 0.403 | 0.739 | 1.8 |
| ridge | 0.733 | 0.535 | 0.590 | 0.0 |
| linear_regression | 0.733 | 0.535 | 0.590 | 0.0 |
| dummy_mean | 1.145 | 0.903 | -0.000 | 0.0 |

## 6. Result Figures

- `figures/results/target_histogram.svg`, `learning_curve.svg`, `parity_plot.svg`, `residual_histogram.svg`, `residual_vs_target.svg`
- Why these matter: They expose target shape, sample-efficiency, fit quality, and residual structure.

## 7. Analysis Figures

- `figures/analysis/feature_importance.svg`, `worst_prediction_cases.svg`, `regional_error_slice.svg`, `error_slice_by_income.svg`
- Key failure patterns: Largest misses cluster in high-value tracts and geographic pockets near the coast.

## 8. Sample Predictions

- Worst examples: `predictions/worst_predictions.csv`
- Failure examples: High absolute-error rows in the table figure.

## 9. Decision

- Promote to `reports/`? yes
- Promote weights to `artifacts/promoted/` or HF Hub? no
- Next run: Add log-target experiments or capped-target aware losses.
