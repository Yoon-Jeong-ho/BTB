from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'
THEORY_BACKLINK = '[THEORY.md](./THEORY.md)'
SCRATCH_REQUIRED_KEYS = (
    'overall_accuracy',
    'answer_type_accuracy',
    'figure_path',
    'rows',
)
FRAMEWORK_REQUIRED_KEYS = (
    'overall_accuracy',
    'question_accuracy',
    'answer_type_accuracy',
    'rows',
)
ROW_REQUIRED_KEYS = (
    'image_label',
    'question',
    'answer_type',
    'gold_answer',
    'predicted_answer',
    'is_correct',
)
ANSWER_TYPE_BUCKETS = ('yes/no', 'color', 'count')


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def _ensure_metrics_exist() -> None:
    missing = [path for path in (SCRATCH, FRAMEWORK) if not path.exists()]
    if not missing:
        return

    missing_list = ', '.join(str(path.relative_to(UNIT_ROOT)) for path in missing)
    raise SystemExit(
        '필수 metrics 파일이 없습니다: '
        f'{missing_list}. 먼저 scratch_lab.py와 framework_lab.py를 실행하세요.'
    )


def _ensure_required_keys(metrics: dict[str, object], *, name: str, required_keys: tuple[str, ...]) -> None:
    missing_keys = [key for key in required_keys if key not in metrics]
    if not missing_keys:
        return

    raise SystemExit(
        'metrics schema validation failed: '
        f'{name} metrics missing keys: {", ".join(missing_keys)}'
    )


def _ensure_row_schema(rows: object, *, name: str) -> None:
    if not isinstance(rows, list):
        raise SystemExit(
            'metrics schema validation failed: '
            f'{name} rows must be a list, got {type(rows).__name__}'
        )

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(
                'metrics schema validation failed: '
                f'{name} rows[{index}] must be an object, got {type(row).__name__}'
            )

        missing_keys = [key for key in ROW_REQUIRED_KEYS if key not in row]
        if missing_keys:
            raise SystemExit(
                'metrics schema validation failed: '
                f'{name} rows[{index}] missing keys: {", ".join(missing_keys)}'
            )


def _ensure_answer_type_accuracy_schema(metrics: dict[str, object], *, name: str) -> None:
    answer_type_accuracy = metrics.get('answer_type_accuracy')
    if not isinstance(answer_type_accuracy, dict):
        raise SystemExit(
            'metrics schema validation failed: '
            f'{name} answer_type_accuracy must be an object, got {type(answer_type_accuracy).__name__}'
        )

    missing_buckets = [bucket for bucket in ANSWER_TYPE_BUCKETS if bucket not in answer_type_accuracy]
    if missing_buckets:
        raise SystemExit(
            'metrics schema validation failed: '
            f'{name} answer_type_accuracy missing buckets: {", ".join(missing_buckets)}'
        )


def _ensure_figure_exists(metrics: dict[str, object], *, name: str) -> None:
    figure_path = metrics.get('figure_path')
    if not isinstance(figure_path, str) or not figure_path.strip():
        raise SystemExit(
            'metrics schema validation failed: '
            f'{name} figure_path must be a non-empty string'
        )

    figure = UNIT_ROOT / figure_path
    if not figure.exists():
        raise SystemExit(
            'metrics schema validation failed: '
            f'{name} figure_path does not exist: {figure_path}'
        )


def _ensure_stable_analysis_ready() -> None:
    if not ANALYSIS_PATH.exists():
        raise SystemExit('stable analysis.md가 없습니다. 먼저 추적된 분석 문서를 복구하세요.')
    stable_analysis = ANALYSIS_PATH.read_text(encoding='utf-8')
    if THEORY_BACKLINK not in stable_analysis:
        raise SystemExit('stable analysis.md에 THEORY 링크가 없습니다. 분석 기준 문서를 먼저 고치세요.')


def run() -> None:
    _ensure_metrics_exist()
    _ensure_stable_analysis_ready()

    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)
    _ensure_required_keys(scratch, name='scratch', required_keys=SCRATCH_REQUIRED_KEYS)
    _ensure_required_keys(framework, name='framework', required_keys=FRAMEWORK_REQUIRED_KEYS)
    _ensure_row_schema(scratch['rows'], name='scratch')
    _ensure_row_schema(framework['rows'], name='framework')
    _ensure_answer_type_accuracy_schema(scratch, name='scratch')
    _ensure_answer_type_accuracy_schema(framework, name='framework')
    _ensure_figure_exists(scratch, name='scratch')

    scratch_overall = float(scratch['overall_accuracy'])
    framework_overall = float(framework['overall_accuracy'])
    framework_question_accuracy = float(framework['question_accuracy'])
    scratch_answer_types = dict(scratch['answer_type_accuracy'])
    framework_answer_types = dict(framework['answer_type_accuracy'])

    scratch_failure = next((row for row in scratch['rows'] if not row['is_correct']), None)
    if scratch_failure is None:
        failure_summary = '이번 scratch toy 예제에서는 answer-type failure가 관측되지 않았다.'
    else:
        error_reason = scratch_failure.get('error_reason') or 'unknown'
        failure_summary = (
            f"scratch failure 예시: `{scratch_failure['image_label']}` 에서 질문 "
            f"`{scratch_failure['question']}` 에 대해 `{scratch_failure['predicted_answer']}` 를 예측해 "
            f"정답 `{scratch_failure['gold_answer']}` 와 달랐다. 분류된 오류 원인은 `{error_reason}` 이다."
        )

    observed_report = f'''# 03 Visual Question Answering 실행 관측

## 관측 결과
- scratch overall accuracy: `{scratch_overall}`
- scratch yes/no accuracy: `{scratch_answer_types.get('yes/no')}`
- scratch color accuracy: `{scratch_answer_types.get('color')}`
- scratch count accuracy: `{scratch_answer_types.get('count')}`
- scratch figure: `{scratch['figure_path']}`
- framework overall accuracy: `{framework_overall}`
- framework question accuracy: `{framework_question_accuracy}`
- framework yes/no accuracy: `{framework_answer_types.get('yes/no')}`
- framework color accuracy: `{framework_answer_types.get('color')}`
- framework count accuracy: `{framework_answer_types.get('count')}`

## 한국어 해석
- scratch 규칙기는 overall accuracy `{scratch_overall}` 로 얼핏 좋아 보이지만, count answer type accuracy가 `{scratch_answer_types.get('count')}` 까지 떨어져 있다. 즉 “정답률은 괜찮다”와 “질문 유형별로 견고하다”는 전혀 다른 말이다.
- {failure_summary}
- framework tiny classifier는 question accuracy `{framework_question_accuracy}` 와 overall accuracy `{framework_overall}` 를 동시에 1.0으로 올렸고, count accuracy도 `{framework_answer_types.get('count')}` 로 회복했다. 이 toy 예제에서는 이미지 특징과 질문 토큰을 함께 본 fusion이 count 질문까지 안정화한 셈이다.
- 따라서 VQA에서는 overall accuracy 하나만 남기지 말고, **answer type breakdown + qualitative row + 오류 원인 태깅**을 함께 남겨야 한다.

## 다음 실험 메모
- 더 큰 VQA subset으로 가면 answer type을 yes/no, number, other처럼 넓게 묶어 먼저 보고, 그다음 count/error case를 세분화한다.
- 이미지 grounding이 필요한 질문은 attention map이나 region evidence까지 붙이면 qualitative debugging이 더 쉬워진다.
- shortcut bias를 점검하려면 이미지 feature를 약하게 만들거나, question-only baseline을 따로 돌려 보는 것이 좋다.

## 이론 다시 연결하기
- 핵심 개념 복습: [THEORY.md](../../THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
