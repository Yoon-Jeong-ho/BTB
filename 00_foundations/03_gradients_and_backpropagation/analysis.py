from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 03 Gradients and Backpropagation 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 gradient와 backpropagation을 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- gradient는 loss를 줄이는 방향 정보를 담고 있으므로, 부호와 크기를 함께 읽어야 한다.
- backpropagation은 마지막 오차 신호를 앞단 local gradient와 곱해 각 파라미터로 전달한다.
- scratch의 finite-difference gradient check는 analytic gradient 구현이 맞는지 검증하고, framework의 autograd는 같은 개념을 일반 tensor graph로 확장한다.
- `artifacts/scratch-manual/loss_curve.svg`는 현재 weight와 gradient step 이후 weight가 loss 곡선에서 어떻게 이동하는지 보여준다.

## 확인 질문
- analytic gradient와 finite-difference gradient가 거의 같다는 사실은 무엇을 보장하는가?
- backpropagation에서 `d(loss)/d(prediction)`이 앞단 gradient와 어떻게 결합되는가?
- 이번 실행에서 loss 감소와 gradient norm은 `artifacts/analysis-manual/latest_report.md`에 어떻게 기록되었는가?

## 관련 이론
- [THEORY.md](./THEORY.md): chain rule, finite-difference check, autograd 흐름을 다시 확인한다.
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

    grad_error_w = float(scratch['grad_error_w'])
    grad_error_b = float(scratch['grad_error_b'])
    if max(grad_error_w, grad_error_b) < 1e-6:
        grad_check_comment = 'analytic gradient와 finite-difference gradient가 거의 완전히 일치해 scratch backward 구현을 신뢰할 수 있다.'
    else:
        grad_check_comment = 'gradient check 오차가 커서 eps 설정이나 analytic gradient 식을 다시 점검해야 한다.'

    loss_before = float(framework['loss_before_step'])
    loss_after = float(framework['loss_after_step'])
    if loss_after < loss_before:
        step_comment = '한 번의 optimizer step 뒤 loss가 감소해, autograd가 만든 gradient가 실제 감소 방향을 가리켰다.'
    else:
        step_comment = '이번 step에서 loss가 줄지 않았으므로 learning rate나 toy network 설정을 다시 점검해야 한다.'

    observed_report = f'''# 03 Gradients and Backpropagation 실행 관측

## 관측 결과
- scratch loss: `{scratch["loss"]}`
- scratch grad_w / finite_diff_grad_w: `{scratch["grad_w"]}` / `{scratch["finite_diff_grad_w"]}`
- scratch grad_b / finite_diff_grad_b: `{scratch["grad_b"]}` / `{scratch["finite_diff_grad_b"]}`
- scratch updated_loss: `{scratch["updated_loss"]}`
- scratch figure: `{scratch["figure_path"]}`
- framework loss_before_step: `{framework["loss_before_step"]}`
- framework loss_after_step: `{framework["loss_after_step"]}`
- framework total_grad_norm: `{framework["total_grad_norm"]}`

## 한국어 해석
- {grad_check_comment}
- scratch에서 `d(loss)/d(prediction) = prediction - target`이 `x`와 곱해져 `grad_w`가 되었고, bias 쪽은 local derivative가 1이라 `grad_b`로 바로 전달됐다.
- {step_comment}
- framework total grad norm `{framework["total_grad_norm"]}`은 네 개 파라미터 텐서의 gradient 크기를 합쳐 읽은 값이다. 즉 autograd는 단순히 loss 하나만 반환하는 것이 아니라, 각 파라미터가 얼마나 민감한지도 함께 남긴다.
- 더 자세한 개념 정리는 [THEORY.md](../../THEORY.md)를, 그림 확인은 `artifacts/scratch-manual/loss_curve.svg`를 함께 보면 가장 빠르다.
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
