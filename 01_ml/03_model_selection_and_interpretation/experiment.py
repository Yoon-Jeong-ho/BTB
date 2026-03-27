from __future__ import annotations

import math
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _runtime import (
    DATA_ROOT,
    SEED,
    TRACK,
    ModelResult,
    bar_chart,
    boxplot_chart,
    ensure_dir,
    json_dump,
    line_chart,
    markdown_table,
    process,
    regression_metrics,
    table_figure,
    to_dense_float32,
    train_torch_regressor,
    yaml_dump,
)

ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = STAGE_ROOT / 'artifacts'
STAGE_TITLE = '03 Model Selection And Interpretation'
PRIMARY_METRIC = 'rmse'
DATASET_SLUG = 'bike-sharing-hourly'
MODEL_SLUG = 'tuned-hgbdt'


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
    cache_dir = ensure_dir(DATA_ROOT / 'external' / 'bike_sharing')
    zip_path = cache_dir / 'bike_sharing_dataset.zip'
    if not zip_path.exists():
        response = requests.get('https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip', timeout=60)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
    with zipfile.ZipFile(zip_path) as zf:
        df = pd.read_csv(zf.open('hour.csv'))
    df['dteday'] = pd.to_datetime(df['dteday'])
    return df.sort_values(['dteday', 'hr']).reset_index(drop=True)


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    df['day_of_year'] = df['dteday'].dt.dayofyear
    df['month'] = df['dteday'].dt.month
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7.0)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7.0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hr'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hr'] / 24.0)
    y = df['cnt'].to_numpy()
    X = df.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])
    return X, y


def _write_default_summary(paths: dict[str, Path], results: dict[str, ModelResult], best_name: str, best: ModelResult) -> None:
    summary = f"""# 03. 모델 선택과 해석 실행 요약

- 과제: Bike Sharing 시간축 count 회귀
- 최고 모델: `{best_name}`
- 핵심 지표: RMSE={best.metrics['rmse']:.4f}, MAE={best.metrics['mae']:.4f}, R2={best.metrics['r2']:.4f}

## 모델 비교

{markdown_table(['모델', 'RMSE', 'MAE', 'R2'], [[name, f"{res.metrics['rmse']:.4f}", f"{res.metrics['mae']:.4f}", f"{res.metrics['r2']:.4f}"] for name, res in sorted(results.items(), key=lambda kv: kv[1].metrics['rmse'])])}

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
    X, y = make_features(df)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx].reset_index(drop=True), X.iloc[split_idx:].reset_index(drop=True)
    y_train, y_test = y[:split_idx], y[split_idx:]
    preprocessor = Pipeline([('imputer', SimpleImputer(strategy='median'))])

    baseline_model = Pipeline([('prep', clone(preprocessor)), ('model', PoissonRegressor(alpha=0.001, max_iter=500))])
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_result = ModelResult(name='poisson_baseline', metrics=regression_metrics(y_test, baseline_pred), fit_time_sec=0.0, predict_time_sec=0.0, peak_rss_mb=process.memory_info().rss / (1024 ** 2), y_pred=baseline_pred)

    tscv = TimeSeriesSplit(n_splits=5)
    candidate_params = [
        {'learning_rate': 0.08, 'max_leaf_nodes': 31, 'min_samples_leaf': 20, 'max_iter': 160},
        {'learning_rate': 0.05, 'max_leaf_nodes': 63, 'min_samples_leaf': 20, 'max_iter': 220},
        {'learning_rate': 0.03, 'max_leaf_nodes': 127, 'min_samples_leaf': 30, 'max_iter': 280},
    ]
    fold_scores: dict[str, list[float]] = {}
    model_records = []
    X_train_np = preprocessor.fit_transform(X_train)
    for idx, params in enumerate(candidate_params, start=1):
        key = f'candidate_{idx}'
        scores = []
        for tr_idx, val_idx in tscv.split(X_train_np):
            model = HistGradientBoostingRegressor(loss='poisson', random_state=SEED, **params)
            model.fit(X_train_np[tr_idx], y_train[tr_idx])
            pred = model.predict(X_train_np[val_idx])
            scores.append(math.sqrt(mean_squared_error(y_train[val_idx], pred)))
        fold_scores[key] = scores
        model_records.append({'name': key, 'params': params, 'mean_rmse': float(np.mean(scores)), 'std_rmse': float(np.std(scores))})
    best_record = min(model_records, key=lambda rec: rec['mean_rmse'])
    best_params = best_record['params']
    tuned_model = Pipeline([('prep', clone(preprocessor)), ('model', HistGradientBoostingRegressor(loss='poisson', random_state=SEED, **best_params))])
    tuned_model.fit(X_train, y_train)
    tuned_pred = tuned_model.predict(X_test)
    tuned_result = ModelResult(name='tuned_hist_gbdt', metrics=regression_metrics(y_test, tuned_pred), fit_time_sec=0.0, predict_time_sec=0.0, peak_rss_mb=process.memory_info().rss / (1024 ** 2), y_pred=tuned_pred, extras={'cv': best_record})

    scaler = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    n_valid = int(len(X_train) * 0.2)
    X_fit, X_valid = X_train.iloc[:-n_valid], X_train.iloc[-n_valid:]
    y_fit, y_valid = y_train[:-n_valid], y_train[-n_valid:]
    X_fit_np = to_dense_float32(scaler.fit_transform(X_fit))
    X_valid_np = to_dense_float32(scaler.transform(X_valid))
    X_test_np = to_dense_float32(scaler.transform(X_test))
    y_pred_mlp, extras = train_torch_regressor(X_fit_np, y_fit, X_valid_np, y_valid, X_test_np, device=device, epochs=16, batch_size=1024)
    gpu_result = ModelResult(name='gpu_mlp', metrics=regression_metrics(y_test, y_pred_mlp), fit_time_sec=0.0, predict_time_sec=0.0, peak_rss_mb=process.memory_info().rss / (1024 ** 2), y_pred=y_pred_mlp, extras=extras)

    results = {baseline_result.name: baseline_result, tuned_result.name: tuned_result, gpu_result.name: gpu_result}
    best_name = min(results, key=lambda name: results[name].metrics[PRIMARY_METRIC])
    best = results[best_name]

    yaml_dump(paths['artifact_dir'] / 'config.yaml', {
        'track': TRACK,
        'stage': STAGE_TITLE,
        'dataset': DATASET_SLUG,
        'seed': SEED,
        'split': {'train': int(len(X_train)), 'test': int(len(X_test)), 'cv': 5},
        'hardware': {'device': device, 'gpu_name': gpu_name},
        'best_params': best_params,
    })
    json_dump(paths['artifact_dir'] / 'metrics.json', {
        'primary_metric': PRIMARY_METRIC,
        'best_model': best_name,
        'models': {name: {**res.metrics, **(res.extras or {})} for name, res in results.items()},
        'cv_records': model_records,
    })

    pred_df = X_test.copy()
    pred_df['target'] = y_test
    pred_df['pred'] = best.y_pred
    pred_df['abs_error'] = np.abs(pred_df['target'] - pred_df['pred'])
    pred_df.sort_values('abs_error', ascending=False).head(30).to_csv(paths['predictions'] / 'worst_predictions.csv', index=False)

    boxplot_chart(paths['figures_results'] / 'cv_fold_score_boxplot.svg', fold_scores, 'CV fold RMSE boxplot', 'TimeSeriesSplit RMSE across tuned candidates.', 'RMSE')
    leaf_values = [15, 31, 63, 127]
    val_curve_leaf = []
    X_train_np2 = preprocessor.fit_transform(X_train)
    for leaf in leaf_values:
        scores = []
        for tr_idx, val_idx in tscv.split(X_train_np2):
            model = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.05, max_leaf_nodes=leaf, min_samples_leaf=20, max_iter=220, random_state=SEED)
            model.fit(X_train_np2[tr_idx], y_train[tr_idx])
            pred = model.predict(X_train_np2[val_idx])
            scores.append(math.sqrt(mean_squared_error(y_train[val_idx], pred)))
        val_curve_leaf.append(float(np.mean(scores)))
    line_chart(paths['figures_results'] / 'validation_curve.svg', [{'label': 'mean CV RMSE', 'x': leaf_values, 'y': val_curve_leaf, 'color': '#2563eb'}], 'Validation curve', 'Effect of max_leaf_nodes under time-aware CV.', 'max_leaf_nodes', 'RMSE')
    perm = permutation_importance(tuned_model, X_test, y_test, n_repeats=4, random_state=SEED, scoring='neg_root_mean_squared_error')
    top_idx = np.argsort(np.abs(perm.importances_mean))[-12:][::-1]
    feature_names = X.columns.tolist()
    bar_chart(paths['figures_results'] / 'top_feature_importance.svg', [feature_names[i] for i in top_idx], np.abs(perm.importances_mean[top_idx]).tolist(), 'Top feature importance', 'Permutation importance on the held-out test window.', 'feature', 'importance drop')

    slice_rmse = pred_df.groupby('season')[['target', 'pred']].apply(lambda g: math.sqrt(mean_squared_error(g['target'], g['pred']))).sort_values(ascending=False)
    bar_chart(paths['figures_analysis'] / 'subgroup_metric_comparison.svg', [f'season_{i}' for i in slice_rmse.index.tolist()], slice_rmse.values.tolist(), 'Subgroup metric comparison', 'RMSE by season on the test split.', 'season', 'RMSE')
    pred_bins = pd.cut(pred_df['pred'], bins=8)
    bin_mae = pred_df.groupby(pred_bins, observed=False)['abs_error'].mean().fillna(0.0)
    line_chart(paths['figures_analysis'] / 'confidence_bin_plot.svg', [{'label': 'mean abs error', 'x': list(range(1, len(bin_mae) + 1)), 'y': bin_mae.values, 'color': '#dc2626'}], 'Prediction-bin error plot', 'Higher predicted demand bins are harder to fit.', 'prediction bin index', 'MAE')
    failure_slices = pred_df.assign(weather=X_test['weathersit'].values).groupby(['workingday', 'weather'])['abs_error'].mean().sort_values(ascending=False).head(10)
    rows = [[idx[0], idx[1], f'{val:.2f}'] for idx, val in failure_slices.items()]
    table_figure(paths['figures_analysis'] / 'common_failure_slice_summary.svg', 'Common failure slice summary', 'Highest-error combinations of working day and weather.', ['workingday', 'weather', 'MAE'], rows)

    _write_default_summary(paths, results, best_name, best)
    return {
        'stage': STAGE_TITLE,
        'run_id': run_id,
        'best_model': best_name,
        'best_metrics': best.metrics,
        'artifact_dir': str(paths['artifact_dir'].relative_to(ROOT)),
    }
