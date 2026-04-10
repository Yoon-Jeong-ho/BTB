from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 07 학습 레시피와 디버깅 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 남긴다.
- 이 문서는 learning rate / batch size / weight decay / scheduler / sanity check를 읽는 고정된 해석 프레임만 유지해, 반복 실행에도 안정적인 기준점을 제공한다.

## 해석 프레임
- learning rate는 단순한 속도 조절값이 아니라, loss 곡선이 매끈하게 내려갈지 발산할지를 결정하는 안정성 레버다.
- batch size는 gradient noise와 step 빈도를 바꾸므로, 같은 epoch budget에서도 fit 속도와 generalization gap 해석이 달라진다.
- weight decay와 scheduler는 모두 late-stage training을 다듬지만, 하나는 파라미터 크기를 누르고 다른 하나는 step size를 줄인다는 점에서 역할이 다르다.
- data bug는 보통 “train loss는 움직이는데 validation이 비정상적으로 망가지는가?”와 “single-batch overfit조차 실패하는가?” 같은 sanity check로 가장 빨리 드러난다.

## 확인 질문
- scratch와 framework에서 baseline 대비 weight decay + scheduler recipe가 어떤 validation trade-off를 만들었는가?
- large batch recipe는 train loss와 validation gap을 어떻게 바꿨는가?
- high learning rate probe에서 first bad epoch와 alert는 무엇이었는가?
- shifted-label bug probe는 baseline보다 얼마나 큰 validation loss를 남겼는가?
- 이번 실행의 sanity check 결과는 `artifacts/analysis-manual/latest_report.md`에 어떻게 정리되었는가?

## 관련 이론
- [THEORY.md](./THEORY.md): learning rate, batch size, weight decay, scheduler, overfit/underfit, divergence, data bug 해석을 다시 연결한다.
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

    scratch_baseline = scratch['recipes']['small_batch_baseline']
    scratch_regularized = scratch['recipes']['weight_decay_scheduler']
    scratch_large_batch = scratch['recipes']['large_batch_constant_lr']
    scratch_high_lr = scratch['recipes']['high_lr_divergence']
    scratch_shifted = scratch['debug_probes']['shifted_label_bug']

    framework_baseline = framework['recipes']['baseline_tiny_mlp']
    framework_regularized = framework['recipes']['weight_decay_scheduler_tiny_mlp']
    framework_large_batch = framework['recipes']['large_batch_tiny_mlp']
    framework_high_lr = framework['recipes']['high_lr_tiny_mlp']
    framework_shifted = framework['debug_probes']['shifted_label_bug_tiny_mlp']

    scratch_regularized_delta = round(
        float(scratch_baseline['final_val_loss']) - float(scratch_regularized['final_val_loss']),
        6,
    )
    framework_regularized_delta = round(
        float(framework_baseline['final_val_loss']) - float(framework_regularized['final_val_loss']),
        6,
    )

    scratch_gap = float(scratch_baseline['generalization_gap'])
    framework_gap = float(framework_baseline['generalization_gap'])
    scratch_regularized_gap = float(scratch_regularized['generalization_gap'])
    framework_regularized_gap = float(framework_regularized['generalization_gap'])

    if scratch_gap > 0.02:
        scratch_baseline_comment = (
            'scratch baseline은 train loss보다 validation loss가 더 크게 남아, '
            '작은 데이터에서의 mild overfit 패턴을 보여줬다.'
        )
    elif scratch_gap < -0.001:
        scratch_baseline_comment = (
            'scratch baseline은 noisy train target을 직접 맞추느라 train loss가 더 높고, '
            'clean validation split이 더 쉽게 맞는 패턴을 보였다.'
        )
    else:
        scratch_baseline_comment = (
            'scratch baseline은 train/validation loss가 모두 안정적으로 내려가며 '
            '큰 불안정성 없이 수렴했다.'
        )

    if framework_gap > 0.02:
        framework_baseline_comment = (
            'framework baseline tiny MLP는 train loss가 매우 낮아진 반면 validation loss가 더 크게 남아, '
            '이 단위에서 가장 읽기 쉬운 overfit 패턴을 남겼다.'
        )
    else:
        framework_baseline_comment = (
            'framework baseline은 작은 MLP에서도 큰 발산 없이 수렴했지만, '
            'validation 곡선을 함께 봐야 recipe 차이를 읽을 수 있었다.'
        )

    if scratch_regularized_delta < 0:
        scratch_regularized_comment = (
            f'scratch regularized recipe는 final validation loss가 baseline보다 `{abs(scratch_regularized_delta)}`만큼 높았지만, '
            f'gap은 `{scratch_gap}` -> `{scratch_regularized_gap}`로 바뀌어 더 보수적인 수렴을 보여줬다.'
        )
    else:
        scratch_regularized_comment = (
            f'scratch regularized recipe는 final validation loss를 `{abs(scratch_regularized_delta)}`만큼 낮춰 '
            'weight decay + scheduler 조합의 이점을 보여줬다.'
        )

    if framework_regularized_delta > 0:
        framework_regularized_comment = (
            f'framework regularized recipe는 final validation loss를 `{framework_regularized_delta}`만큼 낮추고, '
            f'gap도 `{framework_gap}` -> `{framework_regularized_gap}`로 줄였다.'
        )
    else:
        framework_regularized_comment = (
            'framework regularized recipe는 absolute validation loss 개선은 크지 않았지만, '
            'late-stage step size를 줄이며 overfit drift를 완화했다.'
        )

    observed_report = f'''# 07 학습 레시피와 디버깅 실행 관측

## Scratch 요약
- baseline final train/val loss: `{scratch_baseline["final_train_loss"]}` / `{scratch_baseline["final_val_loss"]}`
- large batch final train/val loss: `{scratch_large_batch["final_train_loss"]}` / `{scratch_large_batch["final_val_loss"]}`
- weight decay + scheduler final val loss: `{scratch_regularized["final_val_loss"]}` (`baseline` 대비 `{scratch_regularized_delta}` 개선)
- high learning rate alert: `{scratch_high_lr["alerts"]}` (first bad epoch: `{scratch_high_lr["first_bad_epoch"]}`)
- shifted-label bug final val loss: `{scratch_shifted["final_val_loss"]}`
- scratch figure: `{scratch["figure_path"]}`

## Framework 요약
- baseline tiny MLP final train/val loss: `{framework_baseline["final_train_loss"]}` / `{framework_baseline["final_val_loss"]}`
- large batch tiny MLP final train/val loss: `{framework_large_batch["final_train_loss"]}` / `{framework_large_batch["final_val_loss"]}`
- weight decay + scheduler tiny MLP final val loss: `{framework_regularized["final_val_loss"]}` (`baseline` 대비 `{framework_regularized_delta}` 개선)
- high learning rate alert: `{framework_high_lr["alerts"]}` (first bad epoch: `{framework_high_lr["first_bad_epoch"]}`)
- shifted-label bug final val loss: `{framework_shifted["final_val_loss"]}`

## 한국어 해석
- {scratch_baseline_comment}
- {framework_baseline_comment}
- large batch recipe는 같은 epoch budget에서 train loss를 덜 낮춰, batch size가 단순 throughput이 아니라 fit 속도와 gradient noise까지 바꾼다는 점을 드러냈다.
- {scratch_regularized_comment}
- {framework_regularized_comment}
- high learning rate probe는 scratch에서는 `{scratch_high_lr["alerts"]}`, framework에서는 `{framework_high_lr["alerts"]}`로 끝나, **발산/gradient explosion은 구현체가 달라도 같은 recipe 실수에서 반복**된다는 점을 확인했다.
- shifted-label bug는 scratch `{scratch_shifted["final_val_loss"]}`, framework `{framework_shifted["final_val_loss"]}`의 큰 validation loss로 나타나, data bug를 성능 문제와 구분하는 가장 쉬운 신호가 되었다.
- sanity check 결과는 scratch `{scratch["sanity_checks"]}` / framework `{framework["sanity_checks"]}`로 남아, single-batch overfit과 label-bug probe를 먼저 보는 습관을 굳혀 준다.
- 더 자세한 개념 정리는 [THEORY.md](../../THEORY.md), 실행별 숫자는 두 metrics JSON, 시각화는 `artifacts/scratch-manual/recipe_comparison.svg`에서 다시 확인한다.
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
