from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / 'RESULTS.md'

STAGE_META = {
    '01 Tabular Classification': {
        'title': '01. 표형 분류',
        'theory': '01_tabular_classification/THEORY.md',
        'artifact': '01_tabular_classification/artifacts',
        'hero': 'figures/results/pr_curve.svg',
        'keys': ['auprc', 'auroc'],
    },
    '02 Tabular Regression': {
        'title': '02. 표형 회귀',
        'theory': '02_tabular_regression/THEORY.md',
        'artifact': '02_tabular_regression/artifacts',
        'hero': 'figures/results/parity_plot.svg',
        'keys': ['rmse', 'mae'],
    },
    '03 Model Selection And Interpretation': {
        'title': '03. 모델 선택과 해석',
        'theory': '03_model_selection_and_interpretation/THEORY.md',
        'artifact': '03_model_selection_and_interpretation/artifacts',
        'hero': 'figures/results/cv_fold_score_boxplot.svg',
        'keys': ['rmse', 'mae'],
    },
    '04 Large Scale Tabular': {
        'title': '04. 대규모 표형 데이터',
        'theory': '04_large_scale_tabular/THEORY.md',
        'artifact': '04_large_scale_tabular/artifacts',
        'hero': 'figures/results/metric_vs_training_time.svg',
        'keys': ['macro_f1', 'accuracy'],
    },
}


def _metric_line(best_metrics: dict[str, Any], keys: list[str]) -> str:
    parts = []
    for key in keys:
        value = best_metrics.get(key)
        if isinstance(value, (int, float)):
            parts.append(f'`{key}`={value:.4f}')
    return ', '.join(parts)


def write_results_index(stage_results: list[dict[str, Any]]) -> None:
    rows = []
    previews = []
    for result in stage_results:
        meta = STAGE_META[result['stage']]
        artifact_dir = Path(result['artifact_dir']).relative_to('01_ml')
        rows.append(
            f"| {meta['title']} | `{result['best_model']}` | {_metric_line(result['best_metrics'], meta['keys'])} | [이론]({meta['theory']}) / [artifact]({artifact_dir.as_posix()}/README.md) |"
        )
        hero = artifact_dir / meta['hero']
        previews.append(f"## {meta['title']}\n\n![]({hero.as_posix()})\n")

    body = [
        '# 01 ML 결과 인덱스',
        '',
        '이 문서는 ML 트랙의 최신 artifact를 한눈에 따라가기 위한 입구다.',
        '',
        '## 실행 환경',
        '',
        '- 전용 conda 환경: [env/README.md](env/README.md)',
        '- 공통 이론 문서: [THEORY.md](THEORY.md)',
        '- 전체 실행 명령: `CUDA_VISIBLE_DEVICES=0 conda run -n btb-01-ml python 01_ml/run_all.py --gpu 0`',
        '',
        '## Stage 요약표',
        '',
        '| Stage | 최고 모델 | 핵심 지표 | 링크 |',
        '| --- | --- | --- | --- |',
        *rows,
        '',
        '## 미리보기',
        '',
        *previews,
    ]
    RESULTS_PATH.write_text('\n'.join(body), encoding='utf-8')
