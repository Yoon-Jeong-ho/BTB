from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'
THEORY_BACKLINK = '[THEORY.md](./THEORY.md)'


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


def _ensure_stable_analysis_ready() -> str:
    if not ANALYSIS_PATH.exists():
        raise SystemExit('stable analysis.md가 없습니다. 먼저 추적된 분석 문서를 복구하세요.')

    stable_analysis = ANALYSIS_PATH.read_text(encoding='utf-8')
    if THEORY_BACKLINK not in stable_analysis:
        raise SystemExit('stable analysis.md에 THEORY 링크가 없습니다. 분석 기준 문서를 먼저 고치세요.')
    return stable_analysis


def _safe_pair_label(labels: list[Any], index: int) -> str:
    if 0 <= index < len(labels):
        return str(labels[index])
    return f'{index}번 쌍'


def run() -> None:
    _ensure_metrics_exist()
    _ensure_stable_analysis_ready()

    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)

    pair_labels = scratch.get('pair_labels', [])
    if not isinstance(pair_labels, list):
        pair_labels = []

    hardest_negative = float(scratch.get('hardest_negative_similarity', 0.0))
    mean_positive = float(scratch.get('mean_positive_similarity', 0.0))
    mean_negative = float(scratch.get('mean_negative_similarity', 0.0))
    top1_accuracy = float(framework.get('top1_alignment_accuracy', 0.0))
    predictions = framework.get('top1_predictions', [])
    hard_negative_gap = round(mean_positive - hardest_negative, 6)

    mismatch_notes: list[str] = []
    if isinstance(predictions, list):
        for idx, predicted in enumerate(predictions):
            if not isinstance(predicted, int):
                continue
            if predicted != idx:
                mismatch_notes.append(
                    f'- `{_safe_pair_label(pair_labels, idx)}` 는 `{_safe_pair_label(pair_labels, predicted)}` 쪽으로 더 끌렸다.'
                )

    mismatch_block = (
        '\n'.join(mismatch_notes)
        if mismatch_notes
        else '- 이번 실행에서는 세 쌍 모두 top-1에서 자기 짝을 가장 가깝게 찾았다.'
    )

    observed_report = f'''# 01 Contrastive Alignment 실행 관측

## 관측 결과
- pair count: `{scratch.get("pair_count", 0)}`
- similarity matrix shape: `{scratch.get("similarity_matrix_shape", [])}`
- logits shape: `{framework.get("logits_shape", [])}`
- temperature: `{scratch.get("temperature", 0.0)}`
- top-1 alignment accuracy: `{top1_accuracy}`
- mean positive similarity: `{mean_positive}`
- mean negative similarity: `{mean_negative}`
- hardest negative similarity: `{hardest_negative}`
- positive-hard-negative gap: `{hard_negative_gap}`
- image→text loss: `{framework.get("loss_i2t", 0.0)}`
- text→image loss: `{framework.get("loss_t2i", 0.0)}`

## 한국어 해석
- 정답 쌍 평균 유사도 `{mean_positive}` 가 음의 쌍 평균 `{mean_negative}` 보다 충분히 높아, 이 tiny 배치에서는 대각선이 분명하게 살아 있다.
- hardest negative가 `{hardest_negative}` 로 남아 있다는 것은 완벽한 분리가 아니라, retrieval 관점에서는 여전히 헷갈릴 수 있는 비슷한 설명이 존재한다는 뜻이다.
- positive-hard-negative gap이 `{hard_negative_gap}` 이므로, 지금은 정답 쌍이 이긴다. 하지만 실제 데이터셋에서 이 간격이 줄면 Recall@K가 먼저 흔들리기 쉽다.
- framework 실행에서 `logits_shape = {framework.get("logits_shape", [])}` 와 `device = {framework.get("device", "cpu")}` 를 확인했으므로, CPU에서도 contrastive/logit-similarity 계산 흐름을 안전하게 재현했다.
- image→text loss와 text→image loss를 함께 보는 이유는 검색 방향 둘 다 안정적으로 정렬되어야 하기 때문이다.

## 정렬 상태 메모
{mismatch_block}

## 이론 다시 연결하기
- 안정적인 해석 프레임은 `analysis.md`에 유지한다.
- 핵심 개념 복습: [THEORY.md](./THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
