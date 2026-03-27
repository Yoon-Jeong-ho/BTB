# Run Summary

## 1. Problem

- Task: Binary income classification on adult census records.
- Track / Stage: 01_ml / 01 Tabular Classification
- Why this run exists: Establish leakage-safe preprocessing and compare weak/strong baselines plus a GPU MLP on device `cuda`.

## 2. Hypothesis

- What changed from the previous run: Initial execution for the ML track.
- Expected effect: Tree ensembles should outperform dummy/logistic baselines on AUPRC; GPU MLP should provide a competitive nonlinear reference while explicitly using GPU 0.

## 3. Dataset

- Dataset name: adult-census-income
- Split version: train=22792, valid=4884, test=4885
- Preprocessing: `?` -> missing, median numeric imputation, frequent categorical imputation, one-hot encoding.
- Data caveats: Adult labels are imbalanced and include sensitive attributes that can create slice disparities.

## 4. Training Setup

- Model: dummy / logistic regression / random forest / GPU MLP
- Tokenizer / Processor: N/A
- Seed: 42
- Batch size: 768 for GPU MLP
- Learning rate: 1e-3 for GPU MLP
- Epochs / steps: 14 epochs for GPU MLP
- Hardware: cuda NVIDIA RTX A6000

## 5. Best Metrics

| Model | AUPRC | AUROC | F1 | Fit sec |
| --- | --- | --- | --- | --- |
| random_forest | 0.783 | 0.911 | 0.697 | 1.5 |
| logistic_regression | 0.766 | 0.904 | 0.672 | 3.9 |
| gpu_mlp | 0.757 | 0.902 | 0.685 | 3.9 |
| dummy_prior | 0.241 | 0.500 | 0.000 | 0.0 |

## 6. Result Figures

- `figures/results/class_distribution.svg`, `age_histogram.svg`, `roc_curve.svg`, `pr_curve.svg`, `confusion_matrix.svg`, `calibration_curve.svg`
- Why these matter: They show imbalance, feature spread, ranking quality, threshold behavior, and calibration quality.

## 7. Analysis Figures

- `figures/analysis/permutation_importance.svg`, `error_slice_by_sex.svg`, `confidence_vs_correctness.svg`, `failure_examples.svg`
- Key failure patterns: High-confidence errors concentrate in minority positive samples and slice disparity is visible between sex groups.

## 8. Sample Predictions

- Best examples: `predictions/top_scored_predictions.csv`
- Failure examples: `predictions/high_confidence_errors.csv`

## 9. Decision

- 대표 artifact로 남길까? yes
- Promote weights to `artifacts/promoted/` or HF Hub? no
- Next run: threshold tuning or cost-sensitive calibration for the positive income class.
