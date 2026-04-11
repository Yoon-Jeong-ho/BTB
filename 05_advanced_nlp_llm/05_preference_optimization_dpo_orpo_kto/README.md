# 05 Preference Optimization: DPO, ORPO, KTO

> Status: runnable
>
> 이 단위는 CPU에서 바로 실행되는 deterministic toy preference-optimization 실습이다. 실제 LLM을 fine-tune하지 않고, chosen/rejected pair와 desirable/undesirable label 위에서 log-prob margin이 어떻게 policy update without full RL 신호로 바뀌는지 관찰한다.

## 왜 이 단위를 배우는가
SFT는 assistant가 어떤 형식으로 답해야 하는지 모방하게 만들지만, 같은 prompt에 대한 여러 답 중 **어느 응답이 더 선호되는지** 를 직접 비교하지는 않는다. Preference optimization은 reward model + online PPO 같은 full RL loop를 전부 구현하지 않고도, offline preference data 위에서 policy log-prob를 chosen 쪽으로 조금 더 밀 수 있게 해 준다.

이 단위의 목표는 숫자를 크게 만드는 것이 아니라 다음 감각을 남기는 것이다.
- chosen/rejected pair는 정답/오답이 아니라 상대 선호 신호다.
- log-prob margin은 policy가 chosen 응답을 rejected보다 얼마나 더 선호하는지 보는 최소 단위다.
- DPO / ORPO / KTO는 모두 preference를 loss에 직접 넣지만, 데이터 요구사항과 anchor 방식이 다르다.
- alignment eval은 win rate 하나가 아니라 factuality, refusal balance, verbosity, style bias를 함께 봐야 한다.

## 이번 단위에서 남길 것
- `scratch_lab.py`: 손으로 만든 preference table에서 DPO/ORPO/KTO 신호와 `artifacts/scratch-manual/preference_margin.svg`를 생성한다.
- `framework_lab.py`: tiny deterministic numeric policy simulation으로 margin update, pair accuracy, reference drift를 생성한다.
- `analysis.py`: metrics가 없으면 명확히 실패하고, 있으면 stable `analysis.md`를 유지한 채 관측 리포트를 `artifacts/analysis-manual/latest_report.md`에 쓴다.
- `THEORY.md`, `PREREQS.md`, `reflection.md`, `lesson.yaml`: 한국어 우선 개념·선행지식·회고 질문·메타데이터를 고정한다.

## 실행 방법
프로젝트 루트(`/data_x/aa007878/projects/BTB`)에서 아래 순서로 실행한다.

```bash
python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/scratch_lab.py
python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/framework_lab.py
python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/analysis.py
```

생성되는 파일은 다음과 같다.

```text
05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/artifacts/
├── scratch-manual/
│   ├── metrics.json
│   └── preference_margin.svg
├── framework-manual/
│   └── metrics.json
└── analysis-manual/
    └── latest_report.md
```

## 실행 결과 예시
아래 예시는 이 저장소의 deterministic toy script가 실제로 쓰는 핵심 필드 모양이다. 값은 코드의 고정 테이블에서 계산된다.

```text
$ python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/scratch_lab.py
{
  "preference_batch": {"prompt_count": 4, "pair_count": 4, "desirable_labels": 2, "undesirable_labels": 2},
  "margin_summary": {
    "avg_policy_margin": 0.38,
    "avg_dpo_advantage": 0.115,
    "policy_update_without_full_rl": "log-prob margin을 offline objective로 직접 이동시킨다."
  },
  "objective_views": {
    "dpo": {"signal": "reference-relative chosen/rejected log-prob margin"},
    "orpo": {"signal": "chosen likelihood anchor plus odds-ratio preference term"},
    "kto": {"requires_chosen_rejected_pairs": false}
  },
  "figure_path": "artifacts/scratch-manual/preference_margin.svg"
}

$ python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/framework_lab.py
{
  "device": "cpu",
  "simulation": "tiny_numeric_policy",
  "policy_update": {
    "avg_margin_before": 0.015,
    "avg_margin_after": 0.17,
    "pair_accuracy_before": 0.5,
    "pair_accuracy_after": 1.0,
    "without_full_rl_loop": true
  },
  "contrast": {"pairwise_reference_method": "DPO", "label_only_method": "KTO"}
}

$ python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/analysis.py
# 05 Preference Optimization 실행 관측
...
## 한국어 해석
- scratch 실험의 평균 log-prob margin이 양수이므로 toy policy는 대체로 chosen 응답을 rejected보다 더 높게 평가한다.
```

## 실습 흐름
1. 같은 prompt에 chosen 응답과 rejected 응답을 붙인 toy preference batch를 읽는다.
2. policy log-prob margin과 reference log-prob margin을 나눠 계산한다.
3. DPO를 reference-relative chosen/rejected margin으로, ORPO를 chosen likelihood anchor + odds-ratio term으로, KTO를 desirable/undesirable utility signal로 비교한다.
4. tiny numeric policy simulation에서 average margin과 pair accuracy가 올라가는지, reference drift가 guardrail 안에 남는지 확인한다.
5. analysis report에서 alignment eval tradeoff를 다시 읽는다. length bias, style over factuality, over-refusal은 win rate가 좋아져도 따로 감시해야 한다.

## 이 단위에서 특히 볼 질문
- chosen/rejected pair는 왜 정답/오답 label과 다르며, 이 차이가 loss 해석을 어떻게 바꾸는가?
- log-prob margin을 직접 키우는 방식은 reward model + PPO 기반 full RL loop와 어디서 갈라지는가?
- DPO는 왜 reference policy가 중요하고, ORPO는 왜 chosen likelihood anchor를 함께 말하는가?
- KTO는 strict pair 없이 desirable/undesirable label을 쓸 수 있지만 어떤 label noise 위험을 더 크게 받는가?
- alignment eval에서 win rate가 올라도 factuality, refusal balance, verbosity, style bias가 악화될 수 있는 이유는 무엇인가?

## 다음 단위와의 연결
이 단위는 "preference를 loss 설계로 직접 넣으면 어디까지 갈 수 있는가"를 먼저 보여 준다. 다음 `05_advanced_nlp_llm/06_rlhf_and_reasoning_rl`에서는 이 offline objective의 한계를 넘어 online rollouts, reward modeling, reasoning-specific RL signal이 왜 다시 등장하는지 본다.
