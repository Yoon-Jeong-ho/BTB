from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline

from _runtime import *  # noqa: F401,F403
from dataset import make_split


def run_stage(device: str) -> dict[str, Any]:
    ctx = build_stage_context('01_tabular_classification', '01 Tabular Classification', 'adult-census-income', 'auprc', 'model-suite', device)
    split = make_split()

    sklearn_models = {
        'dummy_prior': DummyClassifier(strategy='prior'),
        'logistic_regression': Pipeline([
            ('preprocessor', clone(split.preprocessor)),
            ('model', LogisticRegression(max_iter=1200, class_weight='balanced', n_jobs=-1)),
        ]),
        'random_forest': Pipeline([
            ('preprocessor', clone(split.preprocessor)),
            ('model', RandomForestClassifier(
                n_estimators=260,
                min_samples_leaf=2,
                class_weight='balanced_subsample',
                n_jobs=-1,
                random_state=SEED,
            )),
        ]),
    }

    results: dict[str, ModelResult] = {}
    for name, model in sklearn_models.items():
        model, y_pred, y_score, fit_time, predict_time, peak_rss = timed_fit_predict(model, split.X_train, split.y_train, split.X_test)
        if y_score is None:
            y_score = y_pred.astype(float)
        results[name] = ModelResult(
            name=name,
            metrics=binary_metrics(split.y_test, y_pred, np.asarray(y_score)),
            fit_time_sec=fit_time,
            predict_time_sec=predict_time,
            peak_rss_mb=peak_rss,
            y_pred=np.asarray(y_pred),
            y_score=np.asarray(y_score),
        )

    mlp_transformer = clone(split.mlp_preprocessor)
    X_train_mlp = to_dense_float32(mlp_transformer.fit_transform(split.X_train))
    X_valid_mlp = to_dense_float32(mlp_transformer.transform(split.X_valid))
    X_test_mlp = to_dense_float32(mlp_transformer.transform(split.X_test))
    rss_before = process.memory_info().rss
    t0 = time.perf_counter()
    y_pred_mlp, y_prob_mlp, extras = train_torch_classifier(
        X_train_mlp,
        split.y_train,
        X_valid_mlp,
        split.y_valid,
        X_test_mlp,
        n_classes=2,
        device=device,
        epochs=14,
        batch_size=768,
    )
    fit_time = time.perf_counter() - t0
    peak_rss = max(rss_before, process.memory_info().rss) / (1024 ** 2)
    results['gpu_mlp'] = ModelResult(
        name='gpu_mlp',
        metrics=binary_metrics(split.y_test, y_pred_mlp, y_prob_mlp[:, 1]),
        fit_time_sec=fit_time,
        predict_time_sec=0.0,
        peak_rss_mb=peak_rss,
        y_pred=y_pred_mlp,
        y_score=y_prob_mlp[:, 1],
        extras=extras,
    )

    best_name = max(results, key=lambda model_name: results[model_name].metrics[ctx.primary_metric])
    best = results[best_name]
    if best_name == 'gpu_mlp':
        analysis_name = max((name for name in results if name != 'gpu_mlp'), key=lambda model_name: results[model_name].metrics[ctx.primary_metric])
        analysis_pipeline = sklearn_models[analysis_name].fit(split.X_train, split.y_train)
    else:
        analysis_name = best_name
        analysis_pipeline = sklearn_models[best_name].fit(split.X_train, split.y_train)

    yaml_dump(ctx.run_paths.run_dir / 'config.yaml', {
        'track': TRACK,
        'stage': ctx.stage_name,
        'dataset': ctx.dataset_name,
        'seed': SEED,
        'split': {'train': int(len(split.X_train)), 'valid': int(len(split.X_valid)), 'test': int(len(split.X_test))},
        'hardware': {'device': device, 'gpu_name': ctx.gpu_name, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')},
        'models': list(results.keys()),
    })
    json_dump(ctx.run_paths.run_dir / 'metrics.json', {
        'primary_metric': ctx.primary_metric,
        'best_model': best_name,
        'models': {
            name: {
                **result.metrics,
                'fit_time_sec': result.fit_time_sec,
                'predict_time_sec': result.predict_time_sec,
                'peak_rss_mb': result.peak_rss_mb,
                **(result.extras or {}),
            }
            for name, result in results.items()
        },
    })

    pred_df = split.X_test.copy()
    pred_df['label'] = split.y_test
    pred_df['pred'] = best.y_pred
    pred_df['score'] = best.y_score
    pred_df['error'] = (pred_df['label'] != pred_df['pred']).astype(int)
    pred_df.sort_values('score', ascending=False).head(20).to_csv(ctx.run_paths.predictions_dir / 'top_scored_predictions.csv', index=False)
    pred_df.sort_values(['error', 'score'], ascending=[False, False]).head(30).to_csv(ctx.run_paths.predictions_dir / 'high_confidence_errors.csv', index=False)

    class_dist = pd.Series((split.full_frame['income'] == '>50K').astype(int)).value_counts().sort_index()
    bar_chart(
        ctx.run_paths.figures_results / 'class_distribution.svg',
        ['<=50K', '>50K'],
        [float(class_dist.get(0, 0)), float(class_dist.get(1, 0))],
        'Adult income class distribution',
        'Primary dataset class balance before split.',
        'class',
        'count',
        colors=['#60a5fa', '#2563eb'],
        value_fmt='{:.0f}',
    )
    fpr, tpr, _ = roc_curve(split.y_test, best.y_score)
    line_chart(
        ctx.run_paths.figures_results / 'roc_curve.svg',
        [{'label': best_name, 'x': fpr, 'y': tpr, 'color': '#2563eb'}, {'label': 'random', 'x': [0, 1], 'y': [0, 1], 'color': '#9ca3af'}],
        'ROC curve',
        f'Best model: {best_name} (AUROC={best.metrics["auroc"]:.3f})',
        'false positive rate',
        'true positive rate',
        y_range=(0.0, 1.0),
    )
    precision, recall, _ = precision_recall_curve(split.y_test, best.y_score)
    line_chart(
        ctx.run_paths.figures_results / 'pr_curve.svg',
        [{'label': best_name, 'x': recall, 'y': precision, 'color': '#dc2626'}],
        'Precision-Recall curve',
        f'Best model: {best_name} (AUPRC={best.metrics["auprc"]:.3f})',
        'recall',
        'precision',
        y_range=(0.0, 1.02),
    )
    heatmap(
        ctx.run_paths.figures_results / 'confusion_matrix.svg',
        confusion_matrix(split.y_test, best.y_pred),
        ['true <=50K', 'true >50K'],
        ['pred <=50K', 'pred >50K'],
        'Confusion matrix',
        f'Best model: {best_name}',
    )
    prob_true, prob_pred = calibration_curve(split.y_test, best.y_score, n_bins=10, strategy='quantile')
    line_chart(
        ctx.run_paths.figures_results / 'calibration_curve.svg',
        [{'label': best_name, 'x': prob_pred, 'y': prob_true, 'color': '#059669'}, {'label': 'ideal', 'x': [0, 1], 'y': [0, 1], 'color': '#9ca3af'}],
        'Calibration curve',
        'Probability calibration on the test split.',
        'mean predicted probability',
        'fraction of positives',
        y_range=(0.0, 1.02),
    )

    perm = permutation_importance(analysis_pipeline, split.X_test, split.y_test, n_repeats=5, random_state=SEED, scoring='average_precision')
    feat_names = analysis_pipeline.named_steps['preprocessor'].get_feature_names_out()
    top_idx = np.argsort(perm.importances_mean)[-12:][::-1]
    bar_chart(
        ctx.run_paths.figures_analysis / 'permutation_importance.svg',
        [feat_names[i].split('__')[-1][:18] for i in top_idx],
        perm.importances_mean[top_idx].tolist(),
        'Permutation importance',
        f'Computed with {analysis_name} on test split.',
        'feature',
        'importance drop',
    )
    slice_error = pred_df.assign(sex=split.X_test['sex'].values).groupby('sex')['error'].mean().sort_values(ascending=False)
    bar_chart(
        ctx.run_paths.figures_analysis / 'error_slice_by_sex.svg',
        slice_error.index.tolist(),
        slice_error.values.tolist(),
        'Error slice by sex',
        'Mean error rate across a simple demographic slice.',
        'slice',
        'error rate',
    )
    conf_bins = pd.cut(pred_df['score'], bins=np.linspace(0, 1, 11), include_lowest=True)
    conf_acc = pred_df.groupby(conf_bins, observed=False).apply(lambda group: 1 - group['error'].mean()).fillna(0.0)
    conf_mid = [0.05 + 0.1 * index for index in range(len(conf_acc))]
    line_chart(
        ctx.run_paths.figures_analysis / 'confidence_vs_correctness.svg',
        [{'label': best_name, 'x': conf_mid, 'y': conf_acc.values, 'color': '#7c3aed'}],
        'Confidence vs correctness',
        'Higher confidence bins should correspond to higher observed accuracy.',
        'predicted probability bin',
        'accuracy',
        y_range=(0.0, 1.02),
    )
    failure_examples = pred_df[pred_df['error'] == 1].sort_values('score', ascending=False).head(8)
    table_rows = failure_examples[['age', 'education', 'occupation', 'hours.per.week', 'label', 'pred', 'score']].round({'score': 3}).values.tolist()
    table_figure(
        ctx.run_paths.figures_analysis / 'failure_examples.svg',
        'High-confidence failure examples',
        'Representative mistakes from the best model.',
        ['age', 'education', 'occupation', 'hours', 'label', 'pred', 'score'],
        table_rows,
    )

    readme = f"""# 01. 표형 분류 실행 요약

- 과제: Adult Census Income 이진 분류
- 최고 모델: `{best_name}`
- 핵심 지표: AUPRC={best.metrics['auprc']:.4f}, AUROC={best.metrics['auroc']:.4f}, F1={best.metrics['f1']:.4f}, Accuracy={best.metrics['accuracy']:.4f}

## 모델 비교

{markdown_table(['모델', 'AUPRC', 'AUROC', 'F1', 'Accuracy'], [[name, f"{result.metrics['auprc']:.4f}", f"{result.metrics['auroc']:.4f}", f"{result.metrics['f1']:.4f}", f"{result.metrics['accuracy']:.4f}"] for name, result in sorted(results.items(), key=lambda item: item[1].metrics['auprc'], reverse=True)])}

## 파일 둘러보기

- 이론 노트: [../../THEORY.md](../../THEORY.md)
- stage 가이드: [../../README.md](../../README.md)
- 결과 figure: `figures/results/`
- 분석 figure: `figures/analysis/`
- 고확신 오답: `predictions/high_confidence_errors.csv`
"""
    summary = f"""# 01. 표형 분류 한눈 요약

- 최고 모델: `{best_name}`
- 핵심 지표: AUPRC={best.metrics['auprc']:.4f}, AUROC={best.metrics['auroc']:.4f}, F1={best.metrics['f1']:.4f}, Accuracy={best.metrics['accuracy']:.4f}
- 이론 링크: [../../THEORY.md](../../THEORY.md)
- 자세한 설명: [README.md](README.md)
"""
    (ctx.run_paths.run_dir / 'README.md').write_text(readme, encoding='utf-8')
    (ctx.run_paths.run_dir / 'summary.md').write_text(summary, encoding='utf-8')

    return {
        'stage': ctx.stage_name,
        'run_id': ctx.run_paths.run_id,
        'best_model': best_name,
        'best_metrics': best.metrics,
        'artifact_dir': str(ctx.run_paths.run_dir.relative_to(ROOT)),
    }
