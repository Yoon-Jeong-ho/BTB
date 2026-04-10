# 08 Alignment, Safety, and Model Behavior

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
모델이 답을 "할 수 있는가" 와, 실제 제품 환경에서 "어떻게 답해야 하는가" 는 같은 문제가 아니다. alignment와 safety는 capability 자체를 새로 만드는 일보다, 이미 있는 능력이 **어떤 정책 경계와 사용자 맥락 안에서 어떤 행동으로 나타나는가** 를 다룬다. 이 단위는 refusal, over-refusal, harmlessness, robustness를 한 프레임으로 묶어 두어, 이후 안전성 논의를 막연한 윤리 문구가 아니라 **행동 평가와 시스템 경계 설계 문제** 로 읽게 만든다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- alignment/capability 구분, refusal framing, behavioral eval intuition을 정리한 `THEORY.md`
- 선행 개념과 빠른 자기 점검을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - benign / harmful / borderline 요청별 behavioral slice 요약
  - refusal / over-refusal / safe alternative 분류 예시
  - paraphrase / jailbreak / noisy prompt에 대한 robustness 관찰 메모
  - model policy와 system guardrail 책임 분리 표

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 같은 모델이 어떤 질문에는 유능해 보여도, 실제 배포 환경에서는 왜 capability보다 **behavior contract** 가 더 중요해지는지 본다.
2. refusal을 단순히 "거절하면 안전" 으로 보지 않고, harmful request 거절·benign request 허용·borderline request의 안전한 축소 응답을 나눠 본다.
3. over-refusal 사례를 따로 모아, harmlessness를 올리는 과정이 usefulness를 어디서 같이 깎는지 관찰한다.
4. robustness 관점에서 prompt paraphrase, formatting noise, jailbreak-style phrasing이 들어가도 행동이 얼마나 안정적으로 유지되는지 본다.
5. behavioral eval을 accuracy 하나로 끝내지 않고, helpfulness / harmlessness / over-refusal / robustness slice로 나눠 보는 이유를 정리한다.
6. 마지막에는 안전성을 모델 내부 정책에만 맡길 수 없는 이유를 정리하며, moderation, tool gating, access control 같은 system-level guardrail과의 경계를 구분한다.

## 이 단위에서 특히 볼 질문
- capability가 높은 모델이 곧바로 alignment가 잘 된 모델이라는 뜻은 왜 아닌가?
- refusal은 언제 바람직한 행동이고, 언제 over-refusal로 usefulness를 해치는가?
- harmlessness를 올린다는 말은 "무조건 거절" 과 어떻게 다른가?
- robustness는 단순 jailbreak 방어만이 아니라 prompt 표현 변화에 대한 안정성까지 왜 포함하는가?
- behavioral eval에서 높은 win rate나 judge score가 실제 안전한 제품 행동을 왜 충분히 보장하지 못하는가?
- policy 문서에 적힌 원칙과 실제 시스템에서 걸어야 하는 guardrail은 어디서 나뉘는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/scratch_lab.py
{
  "status": "sample",
  "behavior_slices": [
    {
      "slice": "benign_request",
      "expected_behavior": "answer normally",
      "observed_label": "compliant",
      "over_refusal_risk": false
    },
    {
      "slice": "harmful_request",
      "expected_behavior": "refuse_and_redirect",
      "observed_label": "safe_refusal",
      "policy_basis": "harmlessness"
    },
    {
      "slice": "borderline_request",
      "expected_behavior": "partial_or_safe_alternative",
      "observed_label": "needs_review",
      "failure_mode": "boundary_ambiguity"
    }
  ],
  "robustness_probe": {
    "canonical_prompt": "stable behavior",
    "paraphrased_prompt": "stable behavior",
    "jailbreak_variant": "watch for policy drift"
  }
}

$ python 05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/framework_lab.py
{
  "status": "sample",
  "eval_summary": {
    "helpfulness_pass_rate": 0.81,
    "harmful_refusal_rate": 0.94,
    "over_refusal_rate": 0.12,
    "robustness_pass_rate": 0.76
  },
  "boundary_map": {
    "model_policy": [
      "unsafe content refusal",
      "safe alternative phrasing",
      "uncertainty handling"
    ],
    "system_guardrails": [
      "tool permission gating",
      "retrieval filtering",
      "rate limits and audit logs"
    ]
  },
  "notes": [
    "expected output/sample shape only",
    "behavioral eval is slice-based, not a single scalar"
  ]
}
```

핵심은 숫자 자체보다도 **어떤 요청에서 어떤 행동이 기대되는지**, **거절이 충분한지 혹은 과도한지**, **표현이 흔들려도 정책 행동이 안정적인지**, **모델 내부 정책과 시스템 guardrail이 각각 무엇을 맡아야 하는지** 를 읽는 것이다.

## 다음 단위와의 연결
이 트랙의 마지막 단위로서, 여기서 behavior contract와 safety boundary를 정리해 두면 다음 학습 단계인 `07_frontier_labs/04_benchmark_and_dataset_construction`에서 behavioral benchmark와 evaluation rubric을 설계할 때 훨씬 구체적인 질문을 던질 수 있다. 다시 말해, 이 단위는 alignment를 막연한 가치 논쟁이 아니라 **측정 가능한 행동 기준과 시스템 책임 분리** 로 정리한 뒤, 그 기준을 실제 benchmark와 dataset으로 옮기기 위한 징검다리다.
