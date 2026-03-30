from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 02 Activation and Loss 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 activation/loss를 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- activation은 중간 표현을 비선형으로 바꿔 모델이 더 복잡한 패턴을 표현하게 한다.
- loss는 예측과 정답 사이의 차이를 scalar로 압축해, optimizer가 따라갈 방향을 만든다.
- BCE와 cross entropy는 각각 다른 target 형식과 입력 가정을 가진다. 실행 결과를 볼 때 “확률을 넣었는지, logits를 넣었는지”를 먼저 확인한다.
- scratch figure `artifacts/scratch-manual/activation_curves.svg`는 곡선의 모양 차이를 눈으로 보여주고, observed report는 이번 실행의 수치 차이를 문장으로 풀어준다.

## 확인 질문
- ReLU / sigmoid / tanh 중 어떤 activation이 입력을 가장 강하게 잘랐는가?
- BCE와 cross entropy는 각각 어떤 정답 표현을 기대하는가?
- 이번 실행에서 관측한 구체적 loss 값과 확률 합은 `artifacts/analysis-manual/latest_report.md`에서 어떻게 해석되었는가?

## 관련 이론
- [THEORY.md](./THEORY.md): activation, logits, probability, loss 연결을 다시 확인한다.
'''


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

    relu_zero_fraction = float(scratch['relu_zero_fraction'])
    if relu_zero_fraction >= 0.5:
        relu_comment = '입력 절반 이상이 0으로 잘려 ReLU가 sparse activation 감각을 분명하게 보여줬다.'
    else:
        relu_comment = '이번 샘플에서는 ReLU가 일부만 0으로 잘려, 입력 분포가 양수 쪽으로 더 치우쳐 있었다.'

    observed_report = f'''# 02 Activation and Loss 실행 관측

## 관측 결과
- scratch relu_zero_fraction: `{scratch["relu_zero_fraction"]}`
- scratch softmax_probability_sum: `{scratch["softmax_probability_sum"]}`
- scratch binary_cross_entropy: `{scratch["binary_cross_entropy"]}`
- scratch cross_entropy: `{scratch["cross_entropy"]}`
- scratch figure: `{scratch["figure_path"]}`
- framework row_probability_sums: `{framework["row_probability_sums"]}`
- framework cross_entropy_loss: `{framework["cross_entropy_loss"]}`
- framework binary_cross_entropy_loss: `{framework["binary_cross_entropy_loss"]}`

## 한국어 해석
- {relu_comment}
- scratch softmax 합이 `{scratch["softmax_probability_sum"]}`이고 framework row sums가 `{framework["row_probability_sums"]}`로 기록되어, probability 분포는 합이 1이 되도록 정규화된다는 점을 다시 확인했다.
- scratch cross entropy `{scratch["cross_entropy"]}`와 framework cross entropy `{framework["cross_entropy_loss"]}`는 모두 정답 class의 확률이 높을수록 작아지는 방향으로 읽는다.
- binary cross entropy는 single logit + binary target을 비교하고, multi-class cross entropy는 class logits + class index target을 비교한다. 그래서 두 loss는 숫자뿐 아니라 입력 형식 자체가 다르다.
- 자세한 개념 설명은 이 단위의 [THEORY.md](../../THEORY.md)와 함께 읽고, 실제 곡선은 `artifacts/scratch-manual/activation_curves.svg`를 열어보는 것이 가장 빠르다.
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
