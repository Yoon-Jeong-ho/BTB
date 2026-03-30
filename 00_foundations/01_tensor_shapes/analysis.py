from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'


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


def run() -> None:
    _ensure_metrics_exist()
    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)

    content = f'''# 01 Tensor Shapes 분석

## 관측 결과
- scratch matmul shape: `{scratch["matmul_shape"]}`
- scratch broadcast result shape: `{scratch["broadcast_result_shape"]}`
- scratch mismatch error: `{scratch["mismatch_error"]}`
- framework logits shape: `{framework["logits_shape"]}`
- framework probs row sums: `{framework["row_probability_sums"]}`

## 해석
- `2 x 3`과 `3 x 4`의 행렬 곱 결과가 `2 x 4`가 되는 것을 직접 확인했다.
- `(2, 3)` 텐서에 `(2, 1)` 텐서를 더하면 broadcasting으로 `(2, 3)` 결과를 만들 수 있었다.
- 반대로 `(4,)` 벡터를 더하려고 하면 마지막 축 길이가 맞지 않아 shape mismatch가 바로 발생했다.
- PyTorch `Linear(8, 3)`는 `(4, 8)` batch를 `(4, 3)` logits로 바꾸며, softmax 이후 각 행의 확률 합은 1에 가깝다.

## 실패 사례
- mismatch 에러 메시지는 `{scratch["mismatch_error"]}` 였다.
- 이 에러는 “원소 수가 달라서”가 아니라 broadcasting 규칙상 축 정렬이 맞지 않아서 생긴다.
'''
    ANALYSIS_PATH.write_text(content, encoding='utf-8')
    print(content)


if __name__ == '__main__':
    run()
