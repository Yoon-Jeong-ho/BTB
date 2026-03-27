from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _runtime import (
    SEED,
    TRACK,
    XGBClassifier,
    ModelResult,
    bar_chart,
    ensure_dir,
    json_dump,
    line_chart,
    markdown_table,
    multiclass_metrics,
    process,
    table_figure,
    timed_fit_predict,
    to_dense_float32,
    train_torch_classifier,
    yaml_dump,
    sanitize_bool_columns,
)

ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = STAGE_ROOT / 'artifacts'
STAGE_TITLE = '04 Large Scale Tabular'
PRIMARY_METRIC = 'macro_f1'
DATASET_SLUG = 'covertype'
MODEL_SLUG = 'large-scale-suite'


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
    ds = load_dataset('mstz/covertype', 'covertype', split='train')
    return sanitize_bool_columns(ds.to_pandas())


def _write_default_summary(paths: dict[str, Path], results: dict[str, ModelResult], best_name: str, best: ModelResult) -> None:
    summary = f"""# 04. 대규모 표형 데이터 실행 요약

- 과제: Covertype 대규모 다중분류
- 최고 모델: `{best_name}`
- 핵심 지표: Macro-F1={best.metrics['macro_f1']:.4f}, Accuracy={best.metrics['accuracy']:.4f}, Macro-Recall={best.metrics['macro_recall']:.4f}

## 모델 비교

{markdown_table(['모델', 'Macro-F1', 'Accuracy', 'Macro-Recall', 'Fit sec'], [[name, f"{res.metrics['macro_f1']:.4f}", f"{res.metrics['accuracy']:.4f}", f"{res.metrics['macro_recall']:.4f}", f"{res.fit_time_sec:.2f}"] for name, res in sorted(results.items(), key=lambda kv: kv[1].metrics['macro_f1'], reverse=True)])}

## 파일 둘러보기

- 이론 노트: [../../THEORY.md](../../THEORY.md)
- stage 가이드: [../../README.md](../../README.md)
- 결과 figure: `figures/results/`
- 분석 figure: `figures/analysis/`
- 예측 샘플: `predictions/test_predictions_sample.csv`
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
    X = df.drop(columns=['cover_type'])
    y = df['cover_type'].astype(int).to_numpy()
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.17647, random_state=SEED, stratify=y_train_valid)

    tree_prep = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])
    linear_prep = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('scale', StandardScaler())])
    model_specs = {
        'sgd_linear': Pipeline([('prep', clone(linear_prep)), ('model', SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=30, early_stopping=True, n_jobs=-1, random_state=SEED))]),
        'shallow_tree': Pipeline([('prep', clone(tree_prep)), ('model', RandomForestClassifier(n_estimators=80, max_depth=12, n_jobs=-1, random_state=SEED))]),
        'hist_gbdt': Pipeline([('prep', clone(tree_prep)), ('model', HistGradientBoostingClassifier(max_depth=10, max_iter=80, learning_rate=0.08, random_state=SEED))]),
    }
    if XGBClassifier is not None:
        model_specs['xgboost_gpu'] = Pipeline([
            ('prep', clone(tree_prep)),
            ('model', XGBClassifier(objective='multi:softprob', num_class=len(np.unique(y)), n_estimators=220, max_depth=10, learning_rate=0.10, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method='hist', device='cuda' if device.startswith('cuda') else 'cpu', eval_metric='mlogloss', random_state=SEED)),
        ])

    results: dict[str, ModelResult] = {}
    for name, model in model_specs.items():
        model, y_pred, y_proba, fit_time, predict_time, peak_rss = timed_fit_predict(model, X_train, y_train, X_test)
        results[name] = ModelResult(name=name, metrics=multiclass_metrics(y_test, y_pred, y_proba), fit_time_sec=fit_time, predict_time_sec=predict_time, peak_rss_mb=peak_rss, y_pred=np.asarray(y_pred), y_score=y_proba)

    scaler = clone(linear_prep)
    X_train_mlp = to_dense_float32(scaler.fit_transform(X_train))
    X_valid_mlp = to_dense_float32(scaler.transform(X_valid))
    X_test_mlp = to_dense_float32(scaler.transform(X_test))
    t0 = time.perf_counter()
    y_pred_mlp, y_prob_mlp, extras = train_torch_classifier(X_train_mlp, y_train, X_valid_mlp, y_valid, X_test_mlp, n_classes=len(np.unique(y)), device=device, epochs=10, batch_size=2048)
    fit_time = time.perf_counter() - t0
    results['gpu_mlp'] = ModelResult(name='gpu_mlp', metrics=multiclass_metrics(y_test, y_pred_mlp, y_prob_mlp), fit_time_sec=fit_time, predict_time_sec=0.0, peak_rss_mb=process.memory_info().rss / (1024 ** 2), y_pred=y_pred_mlp, y_score=y_prob_mlp, extras=extras)

    best_name = max(results, key=lambda name: results[name].metrics[PRIMARY_METRIC])
    best = results[best_name]

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
    pred_df['label'] = y_test
    pred_df['pred'] = best.y_pred
    pred_df['correct'] = (pred_df['label'] == pred_df['pred']).astype(int)
    pred_df.to_csv(paths['predictions'] / 'test_predictions_sample.csv', index=False)

    class_dist = pd.Series(y).value_counts().sort_index()
    bar_chart(paths['figures_results'] / 'class_distribution.svg', [str(i) for i in class_dist.index.tolist()], class_dist.values.astype(float).tolist(), 'Covertype class distribution', 'Seven-class target balance across the full dataset.', 'class', 'count', value_fmt='{:.0f}')
    metric_vs_time = [res.metrics['macro_f1'] for res in results.values()]
    train_times = [res.fit_time_sec for res in results.values()]
    line_chart(paths['figures_results'] / 'metric_vs_training_time.svg', [{'label': 'macro_f1', 'x': train_times, 'y': metric_vs_time, 'color': '#2563eb'}], 'Metric vs training time', 'Higher is better; shows the quality-cost tradeoff.', 'fit time (sec)', 'macro_f1')
    mems = [res.peak_rss_mb for res in results.values()]
    line_chart(paths['figures_results'] / 'metric_vs_memory.svg', [{'label': 'macro_f1', 'x': mems, 'y': metric_vs_time, 'color': '#dc2626'}], 'Metric vs memory', 'Resident memory versus macro-F1.', 'peak rss (MB)', 'macro_f1')
    conf = np.max(best.y_score, axis=1) if best.y_score is not None and np.asarray(best.y_score).ndim == 2 else pred_df['correct'].to_numpy()
    conf_counts, conf_bins = np.histogram(conf, bins=10, range=(0, 1))
    bar_chart(paths['figures_results'] / 'score_distribution.svg', [f"{conf_bins[i]:.1f}-{conf_bins[i+1]:.1f}" for i in range(len(conf_counts))], conf_counts.astype(float).tolist(), 'Score distribution', f'Confidence profile for best model: {best_name}.', 'score bucket', 'count', value_fmt='{:.0f}')

    recall_by_class = []
    labels = []
    for klass in sorted(np.unique(y_test)):
        mask = y_test == klass
        recall_by_class.append(float((best.y_pred[mask] == y_test[mask]).mean()))
        labels.append(str(klass))
    bar_chart(paths['figures_analysis'] / 'slice_metric_by_class.svg', labels, recall_by_class, 'Slice metric by class', f'Per-class recall for {best_name}.', 'class', 'recall')
    throughput_rows = [[name, f"{len(X_train) / max(res.fit_time_sec, 1e-6):.0f}", f"{res.fit_time_sec:.1f}", f"{res.predict_time_sec:.2f}", f"{res.peak_rss_mb:.0f}"] for name, res in results.items()]
    table_figure(paths['figures_analysis'] / 'throughput_bottleneck_summary.svg', 'Throughput bottleneck summary', 'Examples/sec and memory for each large-scale model.', ['model', 'train ex/s', 'fit sec', 'pred sec', 'rss MB'], throughput_rows)
    sample_fracs = [0.1, 0.3, 1.0]
    sample_scores = []
    for frac in sample_fracs:
        n = int(len(X_train_mlp) * frac)
        pred_frac, _, _ = train_torch_classifier(X_train_mlp[:n], y_train[:n], X_valid_mlp, y_valid, X_test_mlp, n_classes=len(np.unique(y)), device=device, epochs=6, batch_size=2048)
        sample_scores.append(float(f1_score(y_test, pred_frac, average='macro')))
    line_chart(paths['figures_analysis'] / 'sampling_strategy_performance.svg', [{'label': 'gpu_mlp macro_f1', 'x': sample_fracs, 'y': sample_scores, 'color': '#059669'}], 'Sampling strategy vs performance', 'Macro-F1 as more training data is introduced.', 'train fraction', 'macro_f1')

    _write_default_summary(paths, results, best_name, best)
    return {
        'stage': STAGE_TITLE,
        'run_id': run_id,
        'best_model': best_name,
        'best_metrics': best.metrics,
        'artifact_dir': str(paths['artifact_dir'].relative_to(ROOT)),
    }
