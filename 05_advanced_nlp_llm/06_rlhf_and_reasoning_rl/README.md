# 06 RLHF and Reasoning RL

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
`05_preference_optimization_dpo_orpo_kto`에서 본 DPO / ORPO / KTO는 **offline preference objective로 policy를 직접 미는 방식** 을 보여 준다. 하지만 실제 assistant를 더 일관되게, 더 안전하게, 더 잘 검증하며, 더 안정적으로 추론하게 만들려면 종종 **reward model + online rollout + policy update** 프레임이 다시 필요해진다. 이 단위는 RLHF를 "복잡한 강화학습 레시피"로 외우는 대신, reward model이 무엇을 대신 측정하고 reasoning-oriented RL이 어떤 행동을 더 밀어 주려 하는지를 high-level에서 정리해 두어, 이후 retrieval/eval과 alignment/safety 단위를 더 정확한 질문으로 보게 만든다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- reward model intuition, RLHF high-level loop, reasoning-oriented RL framing, verifier/judge interaction을 정리한 `THEORY.md`
- 선행 개념과 빠른 자기 점검을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - preference / reward batch 요약과 chosen-vs-rejected score 비교
  - online rollout / update 단계별 샘플 로그 구조
  - reasoning candidate와 verifier / judge signal 비교 표
  - reward hacking, length bias, over-refusal, verbosity drift 관찰 메모

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. SFT 혹은 preference-optimized policy를 출발점으로 두고, 사람/규칙/모델 선호에서 **reward model이 무엇을 대신 근사하려 하는지** 정리한다.
2. 같은 프롬프트에 대한 여러 응답 후보를 두고, chosen / rejected 비교가 scalar reward 신호로 바뀌는 과정을 본다. 여기서 reward는 진실의 점수가 아니라 **선호 proxy** 라는 점을 강조한다.
3. online RLHF loop를 high-level로 따라간다. 프롬프트 샘플링 → policy rollout → reward / preference score 부여 → policy update → regression evaluation의 순서를 잡는다.
4. reasoning-oriented RL에서는 최종 답만 보상하는지, intermediate reasoning behavior·self-correction·verifier pass를 함께 신호로 쓰는지 비교한다.
5. verifier와 judge가 어떤 역할을 나눠 갖는지 본다. verifier는 더 좁고 체크리스트적인 signal, judge는 더 넓고 비교적인 signal을 주지만 둘 다 편향과 gaming 가능성이 있다는 점을 정리한다.
6. 마지막에는 "내부 policy를 더 잘 정렬한 뒤에도 왜 외부 retrieval과 grounding eval이 다시 필요한가?" 를 질문으로 남기며 다음 단위 `05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval`로 연결한다.

## 이 단위에서 특히 볼 질문
- reward model은 무엇을 학습하는가? 정답성 자체인가, 아니면 특정 annotation / rubric / judge의 선호 proxy인가?
- offline preference optimization으로 충분하지 않은 상황에서는 왜 online rollout과 policy update가 다시 필요해지는가?
- reasoning-oriented RL은 "정답 보상"과 무엇이 다른가? 긴 chain-of-thought를 보상하는 것과 좋은 추론 행동을 보상하는 것은 어떻게 다른가?
- verifier와 judge는 각각 어떤 종류의 신호를 주며, 둘이 서로 다른 방향으로 오답을 밀 수 있는 지점은 어디인가?
- RLHF 과정에서 reward hacking, verbosity inflation, over-refusal, style bias는 어떤 형태로 관찰되는가?
- reasoning 품질이 올랐다고 말할 때 최종 정답률 외에 어떤 과정 지표와 failure slice를 같이 봐야 하는가?
- 이 단위를 이해하면 다음 RAG / eval 단위에서 retrieval grounding과 judge-based evaluation을 어떤 더 좋은 질문으로 보게 되는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/scratch_lab.py
{
  "status": "sample",
  "reward_model_batch": {
    "prompt_count": 8,
    "candidate_count": 24,
    "chosen_rejected_pairs": 8,
    "avg_reward_chosen": 1.42,
    "avg_reward_rejected": -0.37,
    "notes": "expected output/sample shape only"
  },
  "rlhf_loop_view": {
    "rollout_rounds": 3,
    "reward_source": "preference-trained reward model + verifier bonus",
    "policy_update_style": "advantage-style / PPO-family sketch",
    "kl_anchor": "enabled",
    "regression_watch": ["verbosity", "refusal", "format drift"]
  },
  "reasoning_signal": {
    "final_answer_reward": 0.71,
    "verifier_pass_rate": 0.62,
    "judge_preference_win_rate": 0.66,
    "process_observation": "longer traces are not always better"
  }
}

$ python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/framework_lab.py
{
  "status": "sample",
  "tensor_shapes": {
    "prompt_input_ids": [4, 256],
    "response_input_ids": [4, 384],
    "reward_scores": [4],
    "advantages": [4],
    "verifier_scores": [4]
  },
  "update_summary": {
    "policy_loss": 0.48,
    "kl_penalty": 0.09,
    "reward_mean": 0.63,
    "reasoning_eval": {
      "answer_accuracy": 0.58,
      "verifier_consistency": 0.61,
      "judge_length_bias_flag": true
    }
  }
}
```

핵심은 숫자 자체보다도 **reward가 무엇의 proxy인지**, **online RLHF loop가 어떤 피드백 경로를 여는지**, **reasoning-oriented RL에서 verifier / judge signal이 어디서 도움이 되고 어디서 왜곡될 수 있는지** 를 읽는 것이다.

## 다음 단위와의 연결
이 단위에서 RLHF와 reasoning RL을 policy-shaping 관점으로 정리해 두면, 다음 단위 `05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval`에서 왜 **좋아 보이는 답변** 과 **실제로 근거를 갖고 있는 답변** 을 다시 분리해서 봐야 하는지가 더 선명해진다. 즉, RLHF는 모델의 내부 행동과 선호를 밀어 주는 단계이고, 다음 단위는 그 위에 retrieval과 grounding evaluation을 얹어 **모델이 무엇을 알고 있는가** 와 **어떻게 답을 검증할 것인가** 를 더 구조적으로 다룬다.
