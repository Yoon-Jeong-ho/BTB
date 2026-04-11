from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 05 Preference Optimization: DPO, ORPO, KTO 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 chosen/rejected pair, log-prob margin, DPO/ORPO/KTO contrast, policy update without full RL, alignment/eval tradeoff를 읽는 **안정적인 프레임**만 남겨 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- preference data의 chosen 응답은 절대 정답이 아니라 rejected보다 선호된 응답이다.
- log-prob margin은 같은 prompt에서 policy가 chosen 쪽에 얼마나 더 높은 확률을 주는지 보는 최소 관찰값이다.
- DPO는 reference-relative chosen/rejected margin을 직접 키우는 pairwise objective로 읽는다.
- ORPO는 chosen likelihood anchor와 odds-ratio preference term을 함께 보는 one-stage preference objective로 읽는다.
- KTO는 strict pair가 없어도 desirable/undesirable label을 비대칭 utility처럼 사용할 수 있다는 점을 강조한다.
- full RL loop 없이도 offline preference objective로 policy를 움직일 수 있지만, eval은 win rate 하나가 아니라 factuality, refusal balance, verbosity, style bias를 나눠 봐야 한다.

## 확인 질문
- chosen/rejected pair가 정답/오답과 다르다는 사실이 loss 해석을 어떻게 바꾸는가?
- reference-relative margin을 쓰면 policy drift는 줄지만 어떤 보수성이 생길 수 있는가?
- ORPO의 chosen likelihood anchor는 imitation과 preference separation 사이에서 어떤 균형을 만든다고 볼 수 있는가?
- KTO처럼 label-only signal을 쓰면 pairwise ranking보다 무엇이 유연해지고 무엇이 약해지는가?
- offline alignment eval에서 length bias, style over factuality, over-refusal을 어떻게 따로 감시할 것인가?

## 관련 이론
- [THEORY.md](./THEORY.md): DPO / ORPO / KTO의 데이터 요구사항, anchor, alignment trade-off를 다시 확인한다.
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


def run() -> None:
    _ensure_metrics_exist()
    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)

    objective_views = scratch.get('objective_views', {})
    margin_summary = scratch.get('margin_summary', {})
    policy_update = framework.get('policy_update', {})
    eval_tradeoffs = framework.get('eval_tradeoffs', {})

    observed_report = f'''# 05 Preference Optimization 실행 관측

## 관측 결과
- preference batch: `{scratch.get("preference_batch", {})}`
- average policy log-prob margin: `{margin_summary.get("avg_policy_margin", "unknown")}`
- average DPO advantage: `{margin_summary.get("avg_dpo_advantage", "unknown")}`
- scratch pair accuracy / judge win rate: `{margin_summary.get("pair_accuracy", "unknown")}`
- framework avg margin before → after: `{policy_update.get("avg_margin_before", "unknown")}` → `{policy_update.get("avg_margin_after", "unknown")}`
- framework pair accuracy before → after: `{policy_update.get("pair_accuracy_before", "unknown")}` → `{policy_update.get("pair_accuracy_after", "unknown")}`
- reference drift after update: `{policy_update.get("reference_drift_after", "unknown")}` / guardrail `{policy_update.get("reference_drift_guardrail", "unknown")}`
- objective losses: `{framework.get("objective_losses", {})}`
- eval tradeoffs: `{eval_tradeoffs}`

## Objective별 요약
- DPO: `{objective_views.get("dpo", {}).get("signal", "unknown")}` — chosen/rejected pair와 reference policy를 함께 본다.
- ORPO: `{objective_views.get("orpo", {}).get("signal", "unknown")}` — chosen likelihood anchor와 odds-ratio term을 함께 본다.
- KTO: `{objective_views.get("kto", {}).get("signal", "unknown")}` — strict pair 없이 desirable/undesirable label도 다룬다.

## 한국어 해석
- scratch 실험의 평균 log-prob margin이 `{margin_summary.get("avg_policy_margin", "unknown")}`로 양수이므로, toy policy는 대체로 chosen 응답을 rejected보다 더 높게 평가한다. 다만 한 pair는 아직 음수 margin을 남겨 offline preference objective가 모든 사례를 자동 해결하지 않는다는 점을 보여 준다.
- DPO advantage는 reference-relative chosen/rejected margin이다. 평균 advantage `{margin_summary.get("avg_dpo_advantage", "unknown")}`는 policy가 reference보다 chosen 쪽으로 더 이동했음을 뜻하지만, drift guardrail과 함께 읽어야 한다.
- framework simulation에서는 full RL rollout이나 reward model 없이 margin을 직접 키워 pair accuracy가 `{policy_update.get("pair_accuracy_before", "unknown")}`에서 `{policy_update.get("pair_accuracy_after", "unknown")}`로 오른다. 이것이 policy update without full RL의 핵심 감각이다.
- DPO/ORPO는 strict pair 신호가 자연스럽고, KTO는 desirable/undesirable label-only setup이 더 유연하다. 따라서 annotation 비용과 label noise를 같이 봐야 한다.
- alignment eval은 win rate 하나로 끝나지 않는다. 이 관측은 length bias, style over factuality, refusal overreach, verbosity delta를 함께 보라는 trade-off 체크리스트를 남긴다.

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
