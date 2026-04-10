from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 03 Domain Adaptive Pretraining 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 domain shift, continued pretraining, catastrophic forgetting, data selection, stopping concern을 읽는 **안정적인 프레임**만 남겨 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- DAPT는 objective를 바꾸는 실험이 아니라 **같은 pretraining objective를 유지한 채 데이터 분포를 다시 가중**하는 실험이다.
- pure-domain continued pretraining은 in-domain validation loss를 빠르게 낮출 수 있지만 general-domain retention loss를 함께 악화시킬 수 있다.
- replay mixture는 general corpus를 일부 다시 보여 주어 forgetting을 늦추지만, domain adaptation 속도를 낮출 수 있다.
- data selection은 문서 수보다 품질, 중복, contamination risk, 목표 분포 적합도를 먼저 봐야 한다.
- stopping은 training loss 최저점 하나가 아니라 in-domain gain, general regression guardrail, downstream probe를 동시에 보는 Pareto decision이다.

## 확인 질문
- domain shift가 vocabulary 차이보다 더 넓은 문체·형식·최신성 차이라는 점을 어떻게 관측할 수 있는가?
- pure domain schedule이 왜 adaptation speed와 catastrophic forgetting을 동시에 키울 수 있는가?
- replay mixture가 retention을 지키는 대신 specialization 속도를 늦추는 이유는 무엇인가?
- data selection에서 noisy large corpus보다 curated small corpus가 나을 수 있는 조건은 무엇인가?
- stop step을 정할 때 in-domain loss와 general retention loss가 충돌하면 어떤 기준을 우선할 것인가?

## 관련 이론
- [THEORY.md](./THEORY.md): domain shift, continued pretraining, replay mixture, forgetting/stopping trade-off를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
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


def _strategy_line(name: str, strategy: dict[str, object]) -> str:
    return (
        f'- `{name}`: domain_share=`{strategy.get("domain_share")}`, '
        f'in-domain gain=`{strategy.get("in_domain_gain_final", strategy.get("in_domain_gain"))}`, '
        f'general regression=`{strategy.get("general_regression_final", strategy.get("general_regression"))}`, '
        f'recommended stop step=`{strategy.get("recommended_stop_step")}`'
    )


def run() -> None:
    _ensure_metrics_exist()
    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)
    scratch_strategies = scratch['strategies']
    framework_strategies = framework['strategies']
    assert isinstance(scratch_strategies, dict)
    assert isinstance(framework_strategies, dict)

    pure_framework = framework_strategies['pure_domain']
    replay_framework = framework_strategies['replay_mixture']
    data_selection = framework.get('data_selection', {})
    curated = data_selection.get('curated_domain', {}) if isinstance(data_selection, dict) else {}
    noisy = data_selection.get('noisy_domain', {}) if isinstance(data_selection, dict) else {}

    observed_report = f'''# 03 Domain Adaptive Pretraining 실행 관측

## 관측 결과
- objective kept constant: `{scratch.get("setup", {}).get("objective_kept_constant", "unknown")}` / `{framework.get("objective_kept_constant", "unknown")}`
- scratch baseline domain loss: `{scratch.get("setup", {}).get("baseline_losses", {}).get("domain", "unknown")}`
- scratch baseline general loss: `{scratch.get("setup", {}).get("baseline_losses", {}).get("general", "unknown")}`
- framework base domain loss: `{framework.get("base_losses", {}).get("domain", "unknown")}`
- framework base general loss: `{framework.get("base_losses", {}).get("general", "unknown")}`
- balanced scratch recommendation: `{scratch.get("comparison", {}).get("balanced_recommendation", "unknown")}`
- framework comparison: `{framework.get("comparison", {})}`
- preferred data selection profile: `{data_selection.get("preferred", "unknown") if isinstance(data_selection, dict) else "unknown"}`

## 전략별 요약
{_strategy_line('pure domain', scratch_strategies['pure_domain'])}
{_strategy_line('replay mixture', scratch_strategies['replay_mixture'])}
{_strategy_line('framework pure domain', pure_framework)}
{_strategy_line('framework replay mixture', replay_framework)}

## 한국어 해석
- 이 toy 실험은 모델 구조나 objective를 바꾸지 않고, **domain shift**가 있는 데이터 분포를 더 자주 보여 주는 continued pretraining 상황만 비교한다.
- pure domain schedule은 domain validation loss를 더 빠르게 낮추지만, framework 관측에서도 general regression이 `{pure_framework.get("general_regression", "unknown")}`로 replay mixture의 `{replay_framework.get("general_regression", "unknown")}`보다 크다. 즉 catastrophic forgetting trade-off가 숫자로 드러난다.
- replay mixture는 general replay를 섞어 adaptation 속도를 일부 포기하는 대신 general retention을 지킨다. balanced recommendation이 replay mixture인 이유도 이 guardrail 때문이다.
- data selection 관점에서는 curated_domain score `{curated.get("selection_score", "unknown")}`가 noisy_domain score `{noisy.get("selection_score", "unknown")}`보다 높다. 문서 수가 많아도 중복과 contamination risk가 크면 DAPT 신호가 나빠질 수 있다.
- stopping은 마지막 step을 자동 선택하는 문제가 아니다. pure domain의 guardrail exceeded step `{pure_framework.get("guardrail_exceeded_step", "unknown")}`와 recommended stop step `{pure_framework.get("recommended_stop_step", "unknown")}`를 나눠 읽어야 한다.

## 이론 다시 연결하기
- stable 프레임은 `analysis.md`에 유지한다.
- 핵심 개념 복습: [THEORY.md](./THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
