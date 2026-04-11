from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 06 RLHF and Reasoning RL 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 toy RLHF / reasoning RL 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 reward model intuition, PPO/RLHF high-level loop, verifier/judge signal, reasoning-oriented reward shaping, failure modes를 읽는 **안정적인 프레임**만 남긴다.
- 따라서 `analysis.py`를 반복 실행해도 이 파일은 같은 내용으로 유지되고, 관측 보고서만 최신 값으로 갱신된다.

## 해석 프레임
- reward model은 truth engine이 아니라 annotation rubric, verifier, judge가 압축된 preference proxy다.
- RLHF loop는 prompt sampling → policy rollout → reward scoring → PPO-family policy update → regression eval의 피드백 경로로 읽는다.
- policy update를 볼 때 reward mean만 보지 말고 KL anchor, reference drift, held-out safety/factuality slice를 같이 본다.
- reasoning RL은 긴 trace를 무조건 보상하는 것이 아니라 outcome reward와 process reward를 섞어 검증 가능성, self-correction, final answer quality를 함께 shaping한다.
- verifier는 좁고 체크리스트적인 signal을, judge는 넓고 비교적인 signal을 주지만 둘 다 reward hacking, length bias, over-refusal에 취약하다.

## 확인 질문
- reward model이 높은 점수를 준 응답은 어떤 rubric proxy에 맞았는가, 그리고 어떤 truth/factuality 축은 놓칠 수 있는가?
- PPO-family update sketch에서 reward가 오르더라도 KL guardrail을 같이 보는 이유는 무엇인가?
- verifier pass rate와 judge win rate가 서로 불일치하면 어떤 failure slice를 먼저 조사해야 하는가?
- reasoning-oriented reward shaping에서 trace length가 아니라 verifier consistency와 answer accuracy를 같이 보는 이유는 무엇인가?
- reward hacking, verbosity inflation, over-refusal을 관찰하려면 어떤 held-out regression prompts를 따로 유지해야 하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): reward model, RLHF, PPO-family update, verifier/judge, reasoning RL failure mode를 다시 확인한다.
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

    reward_batch = scratch.get('reward_model_batch', {})
    loop = scratch.get('rlhf_loop_view', {})
    reasoning_signal = scratch.get('reasoning_signal', {})
    scratch_failures = scratch.get('failure_modes', {})
    policy_update = framework.get('policy_update', {})
    reasoning_eval = framework.get('reasoning_eval', {})
    probes = framework.get('failure_mode_probes', {})

    observed_report = f'''# 06 RLHF and Reasoning RL 실행 관측

## 관측 결과
- reward model batch: `{reward_batch}`
- RLHF loop steps: `{loop.get('steps', [])}`
- policy update style: `{loop.get('policy_update_style', 'unknown')}`
- verifier pass rate: `{reasoning_signal.get('verifier_pass_rate', 'unknown')}`
- judge preference win rate: `{reasoning_signal.get('judge_preference_win_rate', 'unknown')}`
- process reward weight: `{reasoning_signal.get('process_reward_weight', 'unknown')}`
- reward mean before → after: `{policy_update.get('reward_mean_before', 'unknown')}` → `{policy_update.get('reward_mean_after', 'unknown')}`
- advantage mean before → after: `{policy_update.get('advantage_mean_before', 'unknown')}` → `{policy_update.get('advantage_mean_after', 'unknown')}`
- KL after / guardrail: `{policy_update.get('kl_after', 'unknown')}` / `{policy_update.get('kl_guardrail', 'unknown')}`
- answer accuracy before → after: `{reasoning_eval.get('answer_accuracy_before', 'unknown')}` → `{reasoning_eval.get('answer_accuracy_after', 'unknown')}`
- verifier consistency before → after: `{reasoning_eval.get('verifier_consistency_before', 'unknown')}` → `{reasoning_eval.get('verifier_consistency_after', 'unknown')}`
- failure probes: `{probes}`

## 한국어 해석
- scratch 실험의 **reward model**은 `{reward_batch.get('reward_model_intuition', 'unknown')}`이다. 즉 reward는 truth engine이 아니라 verifier, judge, rubric이 섞인 preference proxy로 읽어야 한다.
- toy RLHF loop는 `{loop.get('policy_update_style', 'unknown')}`로 요약된다. 여기서 PPO-family라는 말은 실제 대형 학습을 돌렸다는 뜻이 아니라 advantage 방향으로 policy update를 상상할 수 있게 하는 high-level framing이다.
- reasoning RL 관점에서는 outcome reward만 보지 않고 verifier signal과 judge signal을 함께 본다. verifier pass rate `{reasoning_signal.get('verifier_pass_rate', 'unknown')}`, judge win rate `{reasoning_signal.get('judge_preference_win_rate', 'unknown')}`가 모두 높아도 length bias나 over-refusal probe를 별도로 봐야 한다.
- framework simulation에서는 reward mean과 advantage mean이 올라가지만 KL guardrail `{policy_update.get('kl_guardrail', 'unknown')}` 안에 남는지 확인한다. reward를 높이는 policy update가 항상 안전한 행동 변화라는 뜻은 아니기 때문이다.
- primary failure watch는 `{scratch_failures.get('primary_watch', probes.get('highest_risk', 'unknown'))}`다. reward hacking, verbosity inflation, over-refusal은 RLHF와 reasoning-oriented reward shaping에서 늘 따로 관찰해야 한다.

## 이론 다시 연결하기
- 안정적인 해석 프레임은 `analysis.md`에 유지한다.
- 핵심 개념 복습: [THEORY.md](./THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
