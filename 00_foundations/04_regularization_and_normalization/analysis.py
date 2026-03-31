from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 04 Regularization and Normalization 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 normalization / regularization / training dynamics를 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- normalization은 입력/표현의 scale을 정리해 gradient 크기와 optimization 경로를 더 예측 가능하게 만든다.
- regularization은 loss만 빠르게 줄이는 대신, weight norm이나 특정 경로 의존도가 과도하게 커지는 것을 억제한다.
- scratch의 `training_dynamics.svg`는 같은 learning rate에서도 raw feature와 normalized feature가 얼마나 다른 loss 곡선을 만드는지 보여준다.
- framework 관측에서는 `LayerNorm`, `Dropout`, `weight_decay`가 각각 다른 방식으로 안정화/제약을 주는지 확인한다.

## 확인 질문
- normalization이 initial gradient scale을 얼마나 바꿨는가?
- weight decay를 켰을 때 loss 감소와 weight norm 사이 trade-off는 어떻게 읽어야 하는가?
- 이번 실행에서 LayerNorm, dropout, weight decay 관측은 `artifacts/analysis-manual/latest_report.md`에 어떻게 정리되었는가?

## 관련 이론
- [THEORY.md](./THEORY.md): normalization, regularization, LayerNorm, dropout, weight decay의 역할을 다시 확인한다.
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

    raw_initial_grad = float(scratch['raw_initial_grad_norm'])
    normalized_initial_grad = float(scratch['normalized_initial_grad_norm'])
    grad_ratio = round(raw_initial_grad / normalized_initial_grad, 4)

    raw_final_loss = float(scratch['raw_final_loss'])
    normalized_final_loss = float(scratch['normalized_final_loss'])
    if raw_final_loss > normalized_final_loss:
        scratch_comment = (
            f'raw feature의 initial gradient norm이 normalized 대비 약 `{grad_ratio}`배 컸고, '
            '같은 learning rate에서도 raw/no-regularization 실험은 발산하는 방향으로 움직였다.'
        )
    else:
        scratch_comment = '이번 scratch 설정에서는 raw feature도 안정적이었으므로 feature scale과 learning rate를 다시 점검해야 한다.'

    weight_decay_after = float(framework['weight_decay_weight_norm_after_step'])
    no_weight_decay_after = float(framework['no_weight_decay_weight_norm_after_step'])
    if weight_decay_after < no_weight_decay_after:
        regularization_comment = (
            'framework step에서 weight decay를 켠 쪽의 weight norm이 더 작게 남아, '
            'regularization이 같은 gradient 조건에서 더 작은 norm 해를 선호한다는 점을 보여줬다.'
        )
    else:
        regularization_comment = 'framework step에서 weight decay가 weight norm을 줄이지 못했으므로 실험 설정을 다시 봐야 한다.'

    observed_report = f'''# 04 Regularization and Normalization 실행 관측

## 관측 결과
- scratch raw_initial_grad_norm: `{scratch["raw_initial_grad_norm"]}`
- scratch normalized_initial_grad_norm: `{scratch["normalized_initial_grad_norm"]}`
- scratch raw_final_loss: `{scratch["raw_final_loss"]}`
- scratch normalized_final_loss: `{scratch["normalized_final_loss"]}`
- scratch normalized_l2_weight_norm: `{scratch["normalized_l2_weight_norm"]}`
- scratch figure: `{scratch["figure_path"]}`
- framework layernorm_row_means: `{framework["layernorm_row_means"]}`
- framework layernorm_row_vars: `{framework["layernorm_row_vars"]}`
- framework dropout_train_zero_fraction: `{framework["dropout_train_zero_fraction"]}`
- framework weight_decay_weight_norm_after_step: `{framework["weight_decay_weight_norm_after_step"]}`
- framework no_weight_decay_weight_norm_after_step: `{framework["no_weight_decay_weight_norm_after_step"]}`

## 한국어 해석
- {scratch_comment}
- normalized + L2 실험의 최종 weight norm이 `{scratch["normalized_l2_weight_norm"]}`로 기록되어, regularization이 loss만 보는 것이 아니라 파라미터 크기도 함께 제어한다는 점을 보여줬다.
- LayerNorm 출력 row mean/variance가 `{framework["layernorm_row_means"]}` / `{framework["layernorm_row_vars"]}`로 남아, 샘플 내부 feature scale이 정렬되는 모습을 tiny tensor에서도 확인했다.
- dropout train zero fraction이 `{framework["dropout_train_zero_fraction"]}`이고 eval에서는 입력 보존이 확인되어, stochastic regularization은 train/eval 해석을 분리해야 함을 다시 보여줬다.
- {regularization_comment}
- 더 자세한 개념 정리는 [THEORY.md](../../THEORY.md)를, loss 곡선은 `artifacts/scratch-manual/training_dynamics.svg`를 함께 보면 가장 빠르다.
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
