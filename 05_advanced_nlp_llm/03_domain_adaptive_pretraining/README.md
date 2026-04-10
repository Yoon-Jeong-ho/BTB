# 03 Domain Adaptive Pretraining

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
base LM이 일반 텍스트 분포를 넓게 익혔다고 해서, 곧바로 의료·법률·금융·사내 문서 같은 특정 도메인을 잘 읽거나 쓰는 것은 아니다. 실제 현장에서는 **같은 pretraining objective를 유지한 채 특정 도메인 데이터를 조금 더 먹이는 continued pretraining** 으로 분포 간극(domain shift)을 줄이려는 시도를 자주 한다. 이 단위는 DAPT(domain-adaptive pretraining)를 "그냥 데이터를 더 넣는 단계"가 아니라 **어떤 분포를 더 강하게 밀어 주고, 그 대가로 무엇을 잃을 수 있는지 판단하는 적응 설계 문제** 로 읽게 만든다.

## 이번 단위에서 남길 것
- outline 상태의 학습 안내 문서 `README.md`
- continued pretraining intuition, forgetting trade-off, stopping concern을 정리한 `THEORY.md`
- 선행 개념과 자기 점검 질문을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - base model 대비 in-domain / general-domain validation 변화 요약
  - pure-domain vs replay mixture 비교 표
  - catastrophic forgetting 징후 메모
  - data selection / stopping decision log 초안

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 먼저 base LM이 이미 어떤 일반 분포를 배운 상태인지 가정하고, 새로 들어오는 domain corpus가 vocabulary, 문체, 문서 구조, 사실 밀도에서 무엇이 다른지 정리한다.
2. domain corpus를 그대로 몰아 넣는 경우와 general replay를 섞는 경우를 나란히 두고, adaptation 목표를 "전문화 gain"과 "기존 능력 유지" 두 축으로 본다.
3. continued pretraining 동안 in-domain validation loss와 general-domain validation loss를 함께 추적하며, specialization이 실제로 생기는지와 forgetting이 얼마나 빠르게 커지는지 본다.
4. batch 구성 관점에서는 pure-domain sampling, weighted mixture, replay buffer 같은 선택이 gradient를 어떻게 바꾸는지 observation point 위주로 적는다.
5. stopping 관점에서는 "더 오래 돌리면 무조건 좋다"가 아니라 in-domain 개선이 둔화되는 시점, general regression이 guardrail을 넘는 시점, downstream probe가 포화되는 시점을 함께 본다.
6. 마지막에는 DAPT가 instruction tuning 이전에 왜 필요한지 정리하며, 다음 단위 `05_advanced_nlp_llm/04_instruction_tuning_and_sft`에서 "도메인 지식을 가진 base LM을 어떻게 assistant behavior로 바꿀 것인가" 로 연결한다.

## 이 단위에서 특히 볼 질문
- domain adaptive pretraining은 from-scratch pretraining이나 일반 fine-tuning과 무엇이 다른가?
- domain shift가 큰 corpus일수록 항상 DAPT 효과가 큰가, 아니면 noise·format mismatch 때문에 오히려 불안정해질 수 있는가?
- pure-domain continued pretraining은 왜 빠르게 specialization을 주면서도 catastrophic forgetting 위험을 키우는가?
- general replay나 mixed curriculum을 넣으면 forgetting은 줄어들 수 있는데, 그 대신 adaptation 속도는 얼마나 느려질 수 있는가?
- domain corpus selection에서 문서 수보다 더 먼저 봐야 하는 것은 품질, 중복, 라이선스, 평가셋 오염, 최신성 중 무엇인가?
- stop 시점은 validation loss 하나로 정하면 되는가, 아니면 retention metric과 downstream probe를 함께 봐야 하는가?

## 실행 결과 예시
아래는 **아직 완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 05_advanced_nlp_llm/03_domain_adaptive_pretraining/scratch_lab.py
{
  "status": "sample",
  "setup": {
    "base_objective": "causal_lm",
    "domain_name": "biomedical_ko_en",
    "sampling_plan": {"domain": 0.75, "general_replay": 0.25},
    "train_steps": 1800
  },
  "validation": {
    "base": {
      "in_domain_loss": 2.91,
      "general_loss": 2.34
    },
    "adapted": {
      "in_domain_loss": 2.28,
      "general_loss": 2.47
    },
    "delta": {
      "in_domain": -0.63,
      "general": 0.13
    }
  },
  "stopping_signal": {
    "recent_in_domain_gain": 0.02,
    "general_regression_guardrail": 0.15,
    "decision": "stop_or_increase_replay"
  }
}

$ python 05_advanced_nlp_llm/03_domain_adaptive_pretraining/framework_lab.py
{
  "status": "sample",
  "batch_shape": {
    "input_ids": [4, 2048],
    "attention_mask": [4, 2048],
    "labels": [4, 2048]
  },
  "domain_batch_share": 0.75,
  "general_replay_share": 0.25,
  "observation_points": [
    "in-domain validation improves before downstream probes stabilize",
    "general retention must be tracked in parallel",
    "stopping is a trade-off, not a single best-loss step"
  ]
}
```

핵심은 숫자 자체보다도 **in-domain gain과 general regression을 동시에 읽는 것**, **mixture가 실제 batch 구성에 어떻게 반영되는지 보는 것**, **언제 멈춰야 하는지를 손실 곡선 하나가 아니라 여러 guardrail로 해석하는 것** 이다.

## 다음 단위와의 연결
이 단위에서 DAPT를 통해 base LM의 분포 감각을 특정 도메인 쪽으로 먼저 당겨 두면, 다음 단위 `05_advanced_nlp_llm/04_instruction_tuning_and_sft`에서는 그 지식을 실제 instruction-response 행동으로 바꾸는 문제를 더 분리해서 볼 수 있다. 즉 DAPT는 "무엇을 더 잘 알게 만들 것인가" 에 가깝고, SFT는 "그 지식을 어떤 형식으로 드러내게 만들 것인가" 에 더 가깝다.
