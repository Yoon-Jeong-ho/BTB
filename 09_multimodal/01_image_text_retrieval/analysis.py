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
    'image_to_text_recall_at_1',
    'text_to_image_recall_at_1',
    'text_to_image_recall_at_2',
    'hardest_negative_pair',
    'hardest_negative_similarity',
)
FRAMEWORK_REQUIRED_KEYS = (
    'image_to_text_recall_at_1',
    'text_to_image_recall_at_1',
    'symmetric_loss',
    'logits_shape',
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

    scratch_i2t = float(scratch['image_to_text_recall_at_1'])
    scratch_t2i = float(scratch['text_to_image_recall_at_1'])
    framework_i2t = float(framework['image_to_text_recall_at_1'])
    framework_t2i = float(framework['text_to_image_recall_at_1'])
    framework_loss = float(framework['symmetric_loss'])
    hard_negative = str(scratch['hardest_negative_pair'])
    hard_negative_score = float(scratch['hardest_negative_similarity'])
    recall_gain = round(framework_t2i - scratch_t2i, 6)

    observed_report = f'''# 01 Image-Text Retrieval 실행 관측

## 관측 결과
- scratch image→text Recall@1: `{scratch_i2t}`
- scratch text→image Recall@1: `{scratch_t2i}`
- scratch text→image Recall@2: `{scratch['text_to_image_recall_at_2']}`
- scratch hardest negative pair: `{hard_negative}`
- scratch hardest negative similarity: `{hard_negative_score}`
- framework image→text Recall@1: `{framework_i2t}`
- framework text→image Recall@1: `{framework_t2i}`
- framework symmetric loss: `{framework_loss}`
- framework logits shape: `{framework['logits_shape']}`

## 한국어 해석
- scratch에서는 image→text Recall@1이 `{scratch_i2t}` 로 유지됐지만, text→image Recall@1은 `{scratch_t2i}` 에 머물렀다. 즉 같은 similarity matrix라도 query 방향을 바꾸면 다른 failure가 드러난다.
- 특히 hard negative `{hard_negative}` 가 `{hard_negative_score}` 만큼 높게 남아, 텍스트 query가 잘못된 이미지를 top-1로 고를 여지가 있었다.
- 하지만 scratch text→image Recall@2는 `{scratch['text_to_image_recall_at_2']}` 였다. 이는 정답 후보가 완전히 사라진 것이 아니라, ranking calibration이 top-1에서 흔들렸다는 뜻이다.
- PyTorch dual encoder를 학습한 뒤 framework text→image Recall@1은 `{framework_t2i}` 로 올라갔다. scratch 대비 `{recall_gain}` 만큼 개선되어 양방향 retrieval이 더 대칭적으로 맞춰졌다.
- framework symmetric loss가 `{framework_loss}` 까지 내려간 것은 shared embedding space가 실제 ranking 문제를 더 안정적으로 풀기 시작했다는 신호다.

## 다음 실험 메모
- 이 unit의 안정적인 해석 프레임은 `analysis.md`에 유지한다.
- 실제 COCO/CLIP 실험으로 확장할 때도 먼저 Recall@1/5/10을 image→text 와 text→image 둘 다 기록한다.
- hard negative 사례를 qualitative panel로 같이 남기면, retrieval failure를 숫자 이상으로 설명하기 쉬워진다.

## 이론 다시 연결하기
- 핵심 개념 복습: [THEORY.md](../../THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
