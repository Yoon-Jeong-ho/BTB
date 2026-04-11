from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 04 Instruction Tuning and SFT 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 toy SFT 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 instruction format, supervised fine-tuning, input-output template, role framing, imitation vs helpfulness tradeoff를 해석하는 **안정적인 프레임**만 남긴다.
- 따라서 `analysis.py`를 반복 실행해도 이 파일은 같은 내용으로 유지되고, 관측 보고서만 최신 값으로 갱신된다.

## 해석 프레임
- SFT는 base LM objective를 마법처럼 교체하기보다, serialized instruction example 위에서 assistant response target을 강조하는 supervised imitation 단계다.
- input-output template는 role boundary, generation 시작점, EOS/stop marker, loss target 위치를 결정하는 학습 신호다.
- system/user/assistant role framing은 모델 밖 메타데이터가 아니라 token sequence 안의 conditioning signal이다.
- assistant-only loss mask는 prompt 복창보다 응답 생성에 학습 신호를 집중시킨다.
- imitation score가 높아도 helpfulness가 자동으로 충분해지지는 않으므로, preference optimization으로 이어질 문제를 남겨야 한다.

## 확인 질문
- plain instruction format과 chat template는 같은 예시를 어떤 다른 token boundary로 보여 주는가?
- loss mask에서 prompt tokens를 ignored label로 둔다는 것은 어떤 학습 신호를 제거한다는 뜻인가?
- system message가 있을 때 role framing score가 올라간다면, 이것을 어떤 제품 행동으로 연결할 수 있는가?
- SFT training curve에서 format imitation은 빠르게 좋아지지만 helpfulness proxy는 느리게 움직이는 이유는 무엇인가?
- 다음 preference optimization 단계에서 SFT가 남긴 imitation bias를 어떻게 다시 평가할 것인가?

## 관련 이론
- [THEORY.md](./THEORY.md): instruction format, supervised fine-tuning, input-output template, role framing, imitation/helpfulness tradeoff를 다시 확인한다.
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
    template_views = scratch.get('template_views', {})
    chat_template = template_views.get('chat_template', {}) if isinstance(template_views, dict) else {}
    masking = scratch.get('loss_masking', {})
    role = scratch.get('role_framing', {})
    tradeoff = scratch.get('imitation_vs_helpfulness', {})
    curve = framework.get('training_curve', [])
    first_curve = curve[0] if isinstance(curve, list) and curve else {}
    last_curve = curve[-1] if isinstance(curve, list) and curve else {}
    loss_summary = framework.get('loss_mask_summary', {})
    observed_report = f'''# 04 Instruction Tuning and SFT 실행 관측

## 관측 결과
- scratch unit: `{scratch.get('setup', {}).get('unit', 'unknown')}`
- template names: `{list(template_views.keys()) if isinstance(template_views, dict) else []}`
- chat roles: `{chat_template.get('roles', []) if isinstance(chat_template, dict) else []}`
- target region: `{masking.get('target_region', 'unknown') if isinstance(masking, dict) else 'unknown'}`
- prompt tokens masked out: `{masking.get('prompt_tokens_masked_out', 'unknown') if isinstance(masking, dict) else 'unknown'}`
- assistant loss tokens: `{masking.get('assistant_loss_tokens', 'unknown') if isinstance(masking, dict) else 'unknown'}`
- framework: `{framework.get('framework', 'unknown')}` on `{framework.get('device', 'unknown')}`
- batch shape: `{framework.get('batch_shape', {})}`
- assistant loss: `{first_curve.get('assistant_loss', 'unknown')}` → `{last_curve.get('assistant_loss', 'unknown')}`
- template adherence: `{first_curve.get('template_adherence', 'unknown')}` → `{last_curve.get('template_adherence', 'unknown')}`
- next step: `{framework.get('next_step', {}).get('why_sft_is_not_enough', 'unknown')}`

## 한국어 해석
- 이 toy 실험은 **instruction format**과 chat template가 같은 예시를 다른 input-output template로 직렬화한다는 점을 보여 준다.
- **supervised fine-tuning** 관점에서 system/user/assistant prompt tokens `{loss_summary.get('masked_prompt_tokens', 'unknown') if isinstance(loss_summary, dict) else 'unknown'}`개는 ignored label로 처리되고, assistant response tokens `{loss_summary.get('assistant_loss_tokens', 'unknown') if isinstance(loss_summary, dict) else 'unknown'}`개만 loss target으로 남는다.
- system/user/assistant role framing은 외부 메타데이터가 아니라 serialized token 안의 conditioning signal이다. scratch 관측의 system constraint delta `{role.get('system_constraint_delta', 'unknown') if isinstance(role, dict) else 'unknown'}`가 이 차이를 보여 준다.
- SFT는 reference answer에 대한 **imitation**을 빠르게 높인다. 하지만 scratch helpfulness proxy `{tradeoff.get('helpfulness_proxy_score', 'unknown') if isinstance(tradeoff, dict) else 'unknown'}`는 format imitation score `{tradeoff.get('format_imitation_score', 'unknown') if isinstance(tradeoff, dict) else 'unknown'}`보다 낮아, helpfulness와 모방이 같지 않음을 남긴다.
- 그래서 이 단위의 결론은 “SFT가 assistant 형식의 첫 정렬을 만든다”이지, “선호되는 답변 선택까지 끝났다”가 아니다. 다음 preference optimization 단위에서 이 간극을 다시 평가한다.

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
