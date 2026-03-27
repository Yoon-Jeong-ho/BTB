from __future__ import annotations

import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

from _runtime import (
    SEED,
    TRACK,
    ModelResult,
    bar_chart,
    ensure_dir,
    json_dump,
    line_chart,
    markdown_table,
    process,
    regression_metrics,
    scatter_plot,
    table_figure,
    timed_fit_predict,
    to_dense_float32,
    train_torch_regressor,
    yaml_dump,
)

ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = STAGE_ROOT / 'artifacts'
STAGE_TITLE = '02 Tabular Regression'
PRIMARY_METRIC = 'rmse'
DATASET_SLUG = 'california-housing'
MODEL_SLUG = 'model-suite'


def _new_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{DATASET_SLUG}_{MODEL_SLUG}_s{SEED}"


def _artifact_paths(run_id: str) -> dict[str, Path]:
    artifact_dir = ensure_dir(ARTIFACTS_ROOT / run_id)
    return {
        'artifact_dir': artifact_dir,
        'figures_results': ensure_dir(artifact_dir / 'figures' / 'results'),
        'figures_analysis': ensure_dir(artifact_dir / 'figures' / 'analysis'),
        'predictions': ensure_dir(artifact_dir / 'predictions'),
        'logs': ensure_dir(artifact_dir / 'logs'),
        'checkpoints': ensure_dir(artifact_dir / 'checkpoints'),
    }


def load_dataset_frame() -> pd.DataFrame:
    return fetch_california_housing(as_frame=True).frame


def _write_default_summary(paths: dict[str, Path], results: dict[str, ModelResult], best_name: str, best: ModelResult) -> None:
    summary = f"""# 02. 표형 회귀 실행 요약

- 과제: California Housing 회귀
- 최고 모델: `{best_name}`
- 핵심 지표: RMSE={best.metrics['rmse']:.4f}, MAE={best.metrics['mae']:.4f}, R2={best.metrics['r2']:.4f}

## 모델 비교

{markdown_table(['모델', 'RMSE', 'MAE', 'R2', 'Fit sec'], [[name, f"{res.metrics['rmse']:.4f}", f"{res.metrics['mae']:.4f}", f"{res.metrics['r2']:.4f}", f"{res.fit_time_sec:.2f}"] for name, res in sorted(results.items(), key=lambda kv: kv[1].metrics['rmse'])])}

## 파일 둘러보기

- 이론 노트: [../../THEORY.md](../../THEORY.md)
- stage 가이드: [../../README.md](../../README.md)
- 결과 figure: `figures/results/`
- 분석 figure: `figures/analysis/`
- 실패 사례: `predictions/worst_predictions.csv`
"""
    (paths['artifact_dir'] / 'summary.md').write_text(summary, encoding='utf-8')
    readme_path = paths['artifact_dir'] / 'README.md'
    if not readme_path.exists():
        readme_path.write_text(summary, encoding='utf-8')


def run_stage(device: str) -> dict[str, Any]:
    run_id = _new_run_id()
    paths = _artifact_paths(run_id)
    gpu_name = None
    try:
        import torch
        if device.startswith('cuda') and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = None

    df = load_dataset_frame()
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal'].to_numpy()
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.17647, random_state=SEED)

    num_cols = X.columns.tolist()
    linear_preprocessor = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    tree_preprocessor = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    models = {
        'dummy_mean': Pipeline([('prep', clone(tree_preprocessor)), ('model', DummyRegressor(strategy='mean'))]),
        'linear_regression': Pipeline([('prep', clone(linear_preprocessor)), ('model', LinearRegression())]),
        'ridge': Pipeline([('prep', clone(linear_preprocessor)), ('model', Ridge(alpha=2.0))]),
        'random_forest': Pipeline([('prep', clone(tree_preprocessor)), ('model', RandomForestRegressor(n_estimators=240, min_samples_leaf=2, n_jobs=-1, random_state=SEED))]),
        'hist_gbdt': Pipeline([('prep', clone(tree_preprocessor)), ('model', HistGradientBoostingRegressor(max_depth=6, max_iter=220, learning_rate=0.06, random_state=SEED))]),
    }

    results: dict[str, ModelResult] = {}
    for name, model in models.items():
        model, y_pred, _, fit_time, predict_time, peak_rss = timed_fit_predict(model, X_train, y_train, X_test)
        results[name] = ModelResult(name=name, metrics=regression_metrics(y_test, y_pred), fit_time_sec=fit_time, predict_time_sec=predict_time, peak_rss_mb=peak_rss, y_pred=np.asarray(y_pred))

    mlp_prep = clone(linear_preprocessor)
    X_train_mlp = to_dense_float32(mlp_prep.fit_transform(X_train))
    X_valid_mlp = to_dense_float32(mlp_prep.transform(X_valid))
    X_test_mlp = to_dense_float32(mlp_prep.transform(X_test))
    rss_before = process.memory_info().rss
    t0 = time.perf_counter()
    y_pred_mlp, extras = train_torch_regressor(X_train_mlp, y_train, X_valid_mlp, y_valid, X_test_mlp, device=device, epochs=18, batch_size=768)
    fit_time = time.perf_counter() - t0
    peak_rss = max(rss_before, process.memory_info().rss) / (1024 ** 2)
    results['gpu_mlp'] = ModelResult(name='gpu_mlp', metrics=regression_metrics(y_test, y_pred_mlp), fit_time_sec=fit_time, predict_time_sec=0.0, peak_rss_mb=peak_rss, y_pred=y_pred_mlp, extras=extras)

    best_name = min(results, key=lambda name: results[name].metrics[PRIMARY_METRIC])
    best = results[best_name]
    if best_name == 'gpu_mlp':
        analysis_name = min((n for n in results if n != 'gpu_mlp'), key=lambda name: results[name].metrics[PRIMARY_METRIC])
        analysis_pipeline = models[analysis_name].fit(X_train, y_train)
    else:
        analysis_name = best_name
        analysis_pipeline = models[best_name].fit(X_train, y_train)

    yaml_dump(paths['artifact_dir'] / 'config.yaml', {
        'track': TRACK,
        'stage': STAGE_TITLE,
        'dataset': DATASET_SLUG,
        'seed': SEED,
        'split': {'train': int(len(X_train)), 'valid': int(len(X_valid)), 'test': int(len(X_test))},
        'hardware': {'device': device, 'gpu_name': gpu_name},
        'models': list(results.keys()),
    })
    json_dump(paths['artifact_dir'] / 'metrics.json', {
        'primary_metric': PRIMARY_METRIC,
        'best_model': best_name,
        'models': {name: {**res.metrics, 'fit_time_sec': res.fit_time_sec, 'predict_time_sec': res.predict_time_sec, 'peak_rss_mb': res.peak_rss_mb, **(res.extras or {})} for name, res in results.items()},
    })

    pred_df = X_test.copy()
    pred_df['target'] = y_test
    pred_df['pred'] = best.y_pred
    pred_df['abs_error'] = np.abs(pred_df['target'] - pred_df['pred'])
    pred_df.sort_values('abs_error', ascending=False).head(30).to_csv(paths['predictions'] / 'worst_predictions.csv', index=False)

    target_counts, target_bins = np.histogram(y, bins=12)
    bar_chart(paths['figures_results'] / 'target_histogram.svg', [f"{target_bins[i]:.1f}-{target_bins[i+1]:.1f}" for i in range(len(target_counts))], target_counts.astype(float).tolist(), 'Target histogram', 'California housing target distribution.', 'target bucket', 'count', value_fmt='{:.0f}')
    fractions = [0.1, 0.3, 0.6, 1.0]
    rmse_values = []
    for frac in fractions:
        n = max(200, int(len(X_train) * frac))
        model = Pipeline([('prep', clone(tree_preprocessor)), ('model', HistGradientBoostingRegressor(max_depth=6, max_iter=180, learning_rate=0.06, random_state=SEED))])
        model.fit(X_train.iloc[:n], y_train[:n])
        pred = model.predict(X_valid)
        rmse_values.append(math.sqrt(mean_squared_error(y_valid, pred)))
    line_chart(paths['figures_results'] / 'learning_curve.svg', [{'label': 'hist_gbdt', 'x': fractions, 'y': rmse_values, 'color': '#2563eb'}], 'Learning curve', 'Validation RMSE across training fractions.', 'training fraction', 'RMSE')
    scatter_plot(paths['figures_results'] / 'parity_plot.svg', y_test, best.y_pred, 'Parity plot', f'Best model: {best_name}', 'true target', 'predicted target', diagonal=True)
    residuals = best.y_pred - y_test
    res_counts, res_bins = np.histogram(residuals, bins=14)
    bar_chart(paths['figures_results'] / 'residual_histogram.svg', [f"{res_bins[i]:.1f}" for i in range(len(res_counts))], res_counts.astype(float).tolist(), 'Residual histogram', 'Prediction residual distribution on the test split.', 'residual bucket', 'count', value_fmt='{:.0f}')
    scatter_plot(paths['figures_results'] / 'residual_vs_target.svg', y_test, residuals, 'Residual vs target', 'Residual structure across target values.', 'true target', 'residual')

    perm = permutation_importance(analysis_pipeline, X_test, y_test, n_repeats=5, random_state=SEED, scoring='neg_root_mean_squared_error')
    top_idx = np.argsort(np.abs(perm.importances_mean))[-10:][::-1]
    bar_chart(paths['figures_analysis'] / 'feature_importance.svg', [num_cols[i] for i in top_idx], np.abs(perm.importances_mean[top_idx]).tolist(), 'Feature importance', f'Permutation importance from {analysis_name}.', 'feature', 'importance drop')
    medinc_bin = pd.qcut(X_test['MedInc'], 5, duplicates='drop')
    slice_rmse = pred_df.assign(medinc_bucket=medinc_bin.astype(str)).groupby('medinc_bucket')[['target', 'pred']].apply(lambda g: math.sqrt(mean_squared_error(g['target'], g['pred']))).sort_values(ascending=False)
    bar_chart(paths['figures_analysis'] / 'error_slice_by_income.svg', slice_rmse.index.tolist(), slice_rmse.values.tolist(), 'Error slice by income bucket', 'RMSE varies across income ranges.', 'MedInc quantile', 'RMSE')
    worst_rows = pred_df.sort_values('abs_error', ascending=False).head(10)[['MedInc', 'AveRooms', 'Latitude', 'Longitude', 'target', 'pred', 'abs_error']].round(3).values.tolist()
    table_figure(paths['figures_analysis'] / 'worst_prediction_cases.svg', 'Worst prediction cases', 'Largest absolute errors on the test split.', ['MedInc', 'AveRooms', 'Lat', 'Lon', 'target', 'pred', 'abs err'], worst_rows)
    lat_bin = pd.cut(X_test['Latitude'], bins=6)
    lat_mae = pred_df.assign(lat_bucket=lat_bin.astype(str)).groupby('lat_bucket')['abs_error'].mean().sort_values(ascending=False)
    bar_chart(paths['figures_analysis'] / 'regional_error_slice.svg', lat_mae.index.tolist(), lat_mae.values.tolist(), 'Regional error slice', 'MAE differs by latitude bucket.', 'latitude bucket', 'MAE')

    _write_default_summary(paths, results, best_name, best)
    return {
        'stage': STAGE_TITLE,
        'run_id': run_id,
        'best_model': best_name,
        'best_metrics': best.metrics,
        'artifact_dir': str(paths['artifact_dir'].relative_to(ROOT)),
    }
