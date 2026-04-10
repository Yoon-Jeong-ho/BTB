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
    'exact_match_rate',
    'corpus_unigram_precision',
    'hallucinated_content_tokens_total',
    'figure_path',
    'rows',
)
FRAMEWORK_REQUIRED_KEYS = (
    'exact_match_rate',
    'token_accuracy',
    'corpus_unigram_precision',
    'hallucinated_content_tokens_total',
    'generated_rows',
)
SCRATCH_ROW_REQUIRED_KEYS = (
    'image_label',
    'reference_caption',
    'generated_caption',
    'is_exact_match',
)
FRAMEWORK_ROW_REQUIRED_KEYS = (
    'image_label',
    'reference_caption',
    'generated_caption',
    'is_exact_match',
)


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


def _ensure_row_schema(
    rows: object,
    *,
    name: str,
    required_keys: tuple[str, ...],
) -> None:
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

        missing_keys = [key for key in required_keys if key not in row]
        if missing_keys:
            raise SystemExit(
                'metrics schema validation failed: '
                f'{name} rows[{index}] missing keys: {", ".join(missing_keys)}'
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
    _ensure_row_schema(scratch['rows'], name='scratch', required_keys=SCRATCH_ROW_REQUIRED_KEYS)
    _ensure_row_schema(
        framework['generated_rows'],
        name='framework',
        required_keys=FRAMEWORK_ROW_REQUIRED_KEYS,
    )

    scratch_exact = float(scratch['exact_match_rate'])
    framework_exact = float(framework['exact_match_rate'])
    scratch_bleu1 = float(scratch['corpus_unigram_precision'])
    framework_bleu1 = float(framework['corpus_unigram_precision'])
    scratch_hallucination = int(scratch['hallucinated_content_tokens_total'])
    framework_hallucination = int(framework['hallucinated_content_tokens_total'])
    framework_token_accuracy = float(framework['token_accuracy'])

    scratch_rows = list(scratch['rows'])
    first_failure = next((row for row in scratch_rows if not row['is_exact_match']), None)
    if first_failure is None:
        failure_summary = '이번 scratch toy 예제에서는 exact match failure가 없었다.'
    else:
        failure_summary = (
            f"scratch failure 예시: `{first_failure['image_label']}` 에서 "
            f"`{first_failure['generated_caption']}` 가 생성되어, reference "
            f"`{first_failure['reference_caption']}` 와 달랐다."
        )

    observed_report = f'''# 02 Image Captioning 실행 관측

## 관측 결과
- scratch exact match rate: `{scratch_exact}`
- scratch corpus unigram precision: `{scratch_bleu1}`
- scratch hallucinated content tokens total: `{scratch_hallucination}`
- scratch figure: `{scratch['figure_path']}`
- framework exact match rate: `{framework_exact}`
- framework token accuracy: `{framework_token_accuracy}`
- framework corpus unigram precision: `{framework_bleu1}`
- framework hallucinated content tokens total: `{framework_hallucination}`

## 한국어 해석
- scratch captioner는 규칙 기반이라 빠르게 동작하지만, 해변 장면에서 `dog` prior가 너무 강해져 hallucination token이 `{scratch_hallucination}`개 발생했다. 자동 지표로는 unigram precision `{scratch_bleu1}` 이 꽤 높아도, qualitative caption을 읽으면 오류가 바로 보인다.
- {failure_summary}
- framework decoder는 tiny teacher-forced 학습 뒤 greedy decode exact match가 `{framework_exact}` 로 올라갔다. token accuracy `{framework_token_accuracy}` 와 hallucination total `{framework_hallucination}` 를 함께 보면, 단순 loss 감소가 아니라 실제 생성 문장 품질도 좋아졌다고 해석할 수 있다.
- 즉 image captioning에서는 retrieval처럼 단일 ranking만 보는 것이 아니라, **자동 지표 + hallucination 사례 + 사람이 읽는 자연스러움**을 함께 봐야 한다.

## 다음 실험 메모
- 더 큰 COCO subset으로 가면 BLEU-1 외에 CIDEr/SPICE 같은 caption 특화 지표를 함께 본다.
- 비슷한 장면이 많은 데이터에서는 hallucination 사례를 먼저 qualitative panel로 모아 두는 편이 빠르다.
- decoder loss가 낮아도 greedy decoding이 흔들릴 수 있으므로, 실제 generated caption을 항상 JSON/markdown artifact로 남긴다.

## 이론 다시 연결하기
- 핵심 개념 복습: [THEORY.md](../../THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
