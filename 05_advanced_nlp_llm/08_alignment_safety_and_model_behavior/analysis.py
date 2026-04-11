from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 08 Alignment, Safety, and Model Behavior 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 toy alignment / safety behavior eval 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 alignment vs capability, refusal vs over-refusal, harmlessness / robustness, behavioral eval slice analysis, policy vs system-level safety를 읽는 **안정적인 프레임**만 남긴다.
- 따라서 `analysis.py`를 반복 실행해도 이 파일은 같은 내용으로 유지되고, 관측 보고서만 최신 값으로 갱신된다.

## 해석 프레임
- capability는 모델이 할 수 있는 일을 말하고, alignment는 그 능력이 정책 경계 안에서 어떤 행동으로 나타나는지를 말한다.
- refusal은 harmful request에서 필요하지만, benign request를 막으면 over-refusal이 되어 usefulness를 깎는다.
- harmlessness는 안전한 거절뿐 아니라 safe alternative, 범위 제한, 불확실성 표현을 포함한다.
- robustness는 paraphrase, noisy prompt, jailbreak-style phrasing에도 행동 계약이 유지되는지를 본다.
- behavioral eval은 하나의 scalar가 아니라 benign / harmful / borderline / robustness slice analysis로 읽어야 한다.
- policy vs system-level safety를 분리해야 model policy로 해결할 문제와 tool permission gating, moderation, audit logging으로 해결할 문제를 혼동하지 않는다.

## 확인 질문
- capability score가 높은데 behavior_contract_score가 낮다면 어떤 unsafe compliance나 policy drift가 숨어 있는가?
- harmful refusal rate가 높을 때 benign over-refusal rate도 같이 확인했는가?
- borderline request에서 safe alternative가 아니라 과잉 거절이나 우회 도움을 주고 있지는 않은가?
- robustness probe의 paraphrase / noisy / jailbreak variant 중 어느 표현이 behavior를 가장 많이 흔드는가?
- model policy 책임과 system guardrail 책임을 분리하지 않으면 어떤 product safety failure가 남는가?

## 관련 이론
- [THEORY.md](./THEORY.md): alignment vs capability, refusal / over-refusal, harmlessness, robustness, behavioral eval, system-level safety를 다시 확인한다.
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


def _ensure_stable_analysis_exists() -> None:
    if not ANALYSIS_PATH.exists():
        ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')


def run() -> None:
    _ensure_metrics_exist()
    _ensure_stable_analysis_exists()
    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)

    alignment = scratch.get('alignment_vs_capability', {})
    behavior = scratch.get('behavior_slices', {})
    refusal = scratch.get('refusal_confusion_matrix', {})
    robustness = scratch.get('robustness_probe', {})
    boundary = scratch.get('policy_vs_system_level_safety', {})
    aggregate = framework.get('aggregate_scores', {})
    slices = framework.get('slice_analysis', {})
    behavior_eval = framework.get('behavior_eval', {})
    policy_boundary = framework.get('policy_vs_system_level', {})

    observed_report = f'''# 08 Alignment, Safety, and Model Behavior 실행 관측

## 관측 결과
- alignment vs capability: `{alignment}`
- scratch behavior slices: `{behavior}`
- refusal confusion matrix: `{refusal}`
- robustness probe: `{robustness}`
- scratch policy vs system-level safety: `{boundary}`
- framework aggregate scores: `{aggregate}`
- framework slice analysis: `{slices}`
- behavior eval note: `{behavior_eval}`
- framework policy boundary: `{policy_boundary}`

## 한국어 해석
- **alignment vs capability**는 같은 축이 아니다. capability score가 높아도 unsafe compliance가 남으면 배포 행동은 실패한다.
- **refusal**은 harmful request에서 필요하지만 benign request에서 늘어나면 **over-refusal**이 되어 helpfulness를 깎는다. 그래서 benign answer rate와 harmful refusal rate를 함께 읽어야 한다.
- **harmlessness**는 무조건 거절이 아니라 safe alternative, 범위 제한, 불확실성 표현까지 포함한다. borderline slice에서 safe alternative rate를 따로 보는 이유가 여기에 있다.
- **robustness**는 jailbreak만이 아니라 paraphrase와 noisy prompt에서도 같은 behavior contract가 유지되는지를 본다. min stability `{robustness.get('min_stability', 'unknown')}`가 낮아지면 prompt variation regression set을 추가해야 한다.
- **behavioral eval**은 `{behavior_eval.get('slice_analysis_note', 'slice-based analysis required')}`이다. 단일 judge score는 refusal vs over-refusal, unsafe compliance, safe alternative 품질을 숨길 수 있다.
- **policy vs system-level safety** 관점에서 model policy는 unsafe content refusal과 safe alternative phrasing을 맡고, system guardrail은 tool permission gating, moderation and audit logging, access control을 맡는다. missing guardrail failure는 `{policy_boundary.get('missing_guardrail_failure', 'unknown')}`로 요약된다.

## 이론 다시 연결하기
- 안정적인 해석 프레임은 `analysis.md`에 유지한다.
- 핵심 개념 복습: [THEORY.md](./THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
