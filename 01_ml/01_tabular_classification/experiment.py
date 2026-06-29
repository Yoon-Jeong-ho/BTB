from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from _runtime import *  # noqa: F401,F403
from dataset import make_split
from models import (
    build_sklearn_models,
    choose_best_model,
    fit_analysis_pipeline,
    fit_sklearn_model_suite,
    train_mlp_candidate,
)


def run_stage(device: str) -> dict[str, Any]:
    ctx = build_stage_context('01_tabular_classification', '01 Tabular Classification', 'adult-census-income', 'auprc', 'model-suite', device)
    split = make_split()

    sklearn_models = build_sklearn_models(split.preprocessor)
    results = fit_sklearn_model_suite(sklearn_models, split)
    results['gpu_mlp'] = train_mlp_candidate(split, device)

    best_name, best = choose_best_model(results, ctx.primary_metric)
    analysis_name, analysis_pipeline = fit_analysis_pipeline(results, sklearn_models, split, ctx.primary_metric)

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
