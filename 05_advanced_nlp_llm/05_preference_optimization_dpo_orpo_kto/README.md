# 05 Preference Optimization: DPO, ORPO, KTO

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
instruction tuning과 SFT만으로도 모델은 형식을 꽤 잘 따르지만, **어떤 답이 더 낫고 어떤 답은 피해야 하는가** 라는 선호 기준까지 자동으로 정교해지지는 않는다. 선호 최적화(preference optimization)는 사람/규칙/모델이 만든 선호 신호를 직접 objective에 넣어, online RL loop를 모두 구현하지 않아도 post-training behavior를 더 바람직한 방향으로 미는 방법들이다. 이 단위는 chosen-vs-rejected pair, desirable-vs-undesirable label, reference policy 같은 요소를 한 프레임에서 비교해 두어, 다음 단위의 RLHF와 reasoning-oriented RL로 넘어가기 전에 **정책 업데이트를 loss 설계 문제로 읽는 감각** 을 만든다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- preference data intuition, DPO / ORPO / KTO 대비, alignment trade-off를 정리한 `THEORY.md`
- 선행 개념과 빠른 자기 점검을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - preference batch schema와 chosen/rejected 길이·점수 요약
  - DPO / ORPO / KTO별 objective signal 비교 표
  - reference model 사용 여부와 KL/regularization 관찰 메모
  - offline alignment evaluation에서 볼 편향·회귀 체크리스트

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. `05_advanced_nlp_llm/04_instruction_tuning_and_sft`까지 거친 SFT policy를 출발점으로 두고, 같은 프롬프트에 대해 chosen / rejected 응답 또는 desirable / undesirable 응답이 어떻게 기록되는지 preference data schema를 본다.
2. pairwise preference 관점에서 "모델이 더 높은 확률을 줘야 하는 응답"과 "낮춰야 하는 응답"을 비교하며, reward model 없이도 log-prob 차이를 직접 다루는 이유를 정리한다.
3. DPO에서는 reference policy 대비 chosen 쪽 margin을 키우는 관점, ORPO에서는 SFT anchor 위에 odds-ratio 신호를 얹는 관점, KTO에서는 binary desirability signal을 비대칭 효용으로 다루는 관점을 나란히 비교한다.
4. 각 objective가 요구하는 데이터 형태가 무엇인지 본다. strict pair가 꼭 필요한지, desirable/undesirable label만 있어도 되는지, reference model을 따로 들고 있어야 하는지를 체크한다.
5. alignment trade-off를 관찰한다. helpfulness를 올리다 verbosity를 키우는지, refusal/safety 분포를 과도하게 밀어 버리는지, style preference가 factual quality보다 과대평가되는지 본다.
6. 마지막에는 "offline preference objective로 충분한가, 아니면 online RLHF loop가 언제 필요한가?" 라는 질문을 남기며 다음 단위 `05_advanced_nlp_llm/06_rlhf_and_reasoning_rl`로 연결한다.

## 이 단위에서 특히 볼 질문
- preference data는 정답 데이터와 어떻게 다르고, chosen / rejected라는 말은 무엇을 보장하지 않는가?
- DPO, ORPO, KTO는 모두 "선호를 직접 loss로 넣는다"는 공통점이 있지만, 데이터 요구사항과 regularization 방식은 어떻게 다른가?
- reference policy는 왜 필요할 때도 있고 필요하지 않을 때도 있는가?
- full RL framing 없이 policy를 업데이트한다는 말은 정확히 무엇이며, reward model + PPO와는 어디서 갈리는가?
- alignment 성능이 올랐다고 말할 때 win rate 외에 어떤 regression과 편향을 같이 봐야 하는가?
- 이 단위를 이해하면 다음 RLHF 단위에서 online data collection, reward hacking, reasoning-specific RL signal을 어떤 질문으로 보게 되는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/scratch_lab.py
{
  "status": "sample",
  "preference_batch": {
    "prompt_count": 6,
    "pair_count": 6,
    "desirable_labels": 4,
    "undesirable_labels": 2,
    "avg_chosen_tokens": 118,
    "avg_rejected_tokens": 96
  },
  "objective_views": {
    "dpo": {
      "reference_handling": "explicit reference comparison",
      "signal": "chosen-vs-rejected log-prob margin relative to reference",
      "pairwise_data": true
    },
    "orpo": {
      "reference_handling": "chosen-likelihood anchor",
      "signal": "chosen NLL anchor + odds-ratio preference term",
      "pairwise_data": true
    },
    "kto": {
      "reference_handling": "implementation-dependent / anchored utility",
      "signal": "desirable vs undesirable utility-style update",
      "pairwise_data": false
    }
  },
  "offline_eval": {
    "judge_win_rate": 0.67,
    "length_bias_flag": true,
    "safety_regression_flags": ["over-refusal-risk"],
    "notes": "expected output/sample shape only"
  }
}

$ python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/framework_lab.py
{
  "status": "sample",
  "tensor_shapes": {
    "prompt_input_ids": [4, 256],
    "chosen_input_ids": [4, 320],
    "rejected_input_ids": [4, 304],
    "desirability_labels": [8]
  },
  "loss_terms": {
    "dpo_loss": 0.54,
    "orpo_nll": 1.21,
    "orpo_odds_ratio": 0.18,
    "kto_loss": 0.49
  },
  "eval_summary": {
    "pair_accuracy": 0.75,
    "format_following": 0.88,
    "factuality_watch": "manual review required"
  }
}
```

핵심은 숫자 자체보다도 **preference 신호가 어떤 데이터 구조에서 오고**, **각 objective가 어떤 비교를 loss로 직접 밀어 넣는지**, **offline 지표가 실제 alignment 품질을 어디까지밖에 보장하지 못하는지** 를 읽는 것이다.

## 다음 단위와의 연결
이 단위에서 DPO / ORPO / KTO를 policy-loss 설계 관점으로 정리해 두면, 다음 단위 `05_advanced_nlp_llm/06_rlhf_and_reasoning_rl`에서 왜 reward model, online rollouts, PPO류 update, reasoning-specific RL signal이 다시 등장하는지를 더 선명하게 이해할 수 있다. 다시 말해, 이 단위는 "선호 최적화를 RL 없이 어디까지 가져갈 수 있는가" 를 먼저 정리한 뒤, 그 한계를 넘기 위해 RLHF가 무엇을 추가하는지 보는 징검다리다.
