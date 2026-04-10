from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 01 Perceptron and MLP 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 perceptron과 tiny MLP를 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- perceptron의 decision rule은 `w·x + b`의 부호 하나로 클래스를 가른다. 따라서 decision boundary는 한 줄의 직선(또는 고차원에서는 hyperplane)이다.
- 어떤 toy 데이터가 linear separable하면 single neuron 하나로도 완벽하게 맞출 수 있다. 이 경우 모델이 약해서가 아니라, 문제 자체가 직선 하나로 충분한 것이다.
- XOR처럼 대각선 패턴을 요구하는 데이터는 single neuron의 표현력 한계에 걸린다. 이때 accuracy가 안 나오는 이유를 optimizer 탓으로만 보면 안 된다.
- hidden layer와 nonlinearity가 들어간 tiny MLP는 입력을 한 번 더 재표현해서, 직선 하나로는 안 되던 문제를 풀 수 있다.

## 확인 질문
- single neuron이 잘 되는 경우와 안 되는 경우를 decision boundary 관점에서 어떻게 구분할 수 있는가?
- XOR 실패는 학습률 문제라기보다 표현력 문제라는 말을 어떤 관측으로 설명할 수 있는가?
- 이번 실행의 실제 숫자는 왜 `analysis.md`가 아니라 `artifacts/analysis-manual/latest_report.md`에 남겨야 하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): perceptron decision rule, linear separability, hidden layer의 역할을 다시 확인한다.
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

    xor_best_accuracy = float(scratch['xor_best_accuracy'])
    single_xor_accuracy = float(framework['single_neuron_xor_accuracy'])
    mlp_xor_accuracy = float(framework['tiny_mlp_xor_accuracy'])
    if scratch['linear_is_separable']:
        linear_comment = 'scratch에서 선형 분리 toy 데이터는 직선 하나로 정확히 나뉘었다. 즉 perceptron decision rule 자체는 충분히 강한 baseline이 될 수 있다.'
    else:
        linear_comment = '선형 toy 데이터조차 직선 하나로 안 나뉘었다면 데이터 정의나 decision rule 구현을 다시 점검해야 한다.'

    if xor_best_accuracy < 1.0:
        xor_comment = (
            f'scratch grid search에서도 single neuron의 XOR 최고 accuracy는 `{scratch["xor_best_accuracy"]}`에 머물렀다. '
            '이 실패는 optimizer 미세 조정보다 표현력 한계를 먼저 의심해야 함을 보여 준다.'
        )
    else:
        xor_comment = '이번 scratch 관측은 XOR를 직선 하나로도 분리한다고 나왔으므로 데이터나 평가 규칙을 다시 확인해야 한다.'

    if mlp_xor_accuracy > single_xor_accuracy:
        mlp_comment = (
            f'framework에서는 tiny MLP가 XOR accuracy를 `{framework["tiny_mlp_xor_accuracy"]}`까지 끌어올려 '
            f'single neuron 대비 `{framework["xor_accuracy_gain"]}`만큼 개선했다.'
        )
    else:
        mlp_comment = 'framework 관측에서 tiny MLP가 single neuron보다 낫지 않으므로 seed, epoch, activation 설정을 다시 점검해야 한다.'

    observed_report = f'''# 01 Perceptron and MLP 실행 관측

## 관측 결과
- scratch decision rule: `{scratch["decision_rule"]}`
- scratch linear_dataset_accuracy: `{scratch["linear_dataset_accuracy"]}`
- scratch xor_best_accuracy: `{scratch["xor_best_accuracy"]}`
- scratch figure: `{scratch["figure_path"]}`
- framework backend: `{framework["backend"]}`
- framework single_neuron_linear_accuracy: `{framework["single_neuron_linear_accuracy"]}`
- framework single_neuron_xor_accuracy: `{framework["single_neuron_xor_accuracy"]}`
- framework tiny_mlp_xor_accuracy: `{framework["tiny_mlp_xor_accuracy"]}`
- framework xor_accuracy_gain: `{framework["xor_accuracy_gain"]}`

## 한국어 해석
- {linear_comment}
- {xor_comment}
- {mlp_comment}
- framework parameter count를 비교하면 single neuron `{framework["single_neuron_parameter_count"]}`개, tiny MLP `{framework["tiny_mlp_parameter_count"]}`개로 커지지만, 거대한 모델이 아니라도 hidden layer 하나가 표현력을 바꾼다는 점을 보여 준다.
- 즉 "안 풀린다"는 관측이 나오면 먼저 linear separability와 model capacity를 확인한 뒤 optimizer를 만지는 습관이 중요하다.
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
