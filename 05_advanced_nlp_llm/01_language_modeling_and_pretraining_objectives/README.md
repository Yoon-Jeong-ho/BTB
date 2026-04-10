# 01 Language Modeling and Pretraining Objectives

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 runnable/applied 단계에서 구현될 예상 구조** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
언어모델이 "언어를 안다"고 말할 때, 실제로는 **무엇을 입력으로 보고 무엇을 정답으로 맞히도록 학습했는가** 가 먼저 정해져 있다. 같은 transformer 계열 구조라도 causal LM인지, masked LM인지, span corruption인지에 따라 모델이 보는 문맥, loss가 걸리는 위치, 이후 잘하는 행동이 달라진다. 이 단위는 pretraining objective를 하나의 표면적인 용어가 아니라 **예측 타깃 설계 문제** 로 이해하게 만들어, 다음 단위의 corpus/tokenizer/data mixture 설계와 이후 domain-adaptive pretraining·instruction tuning의 출발점을 세운다.

## 이번 단위에서 남길 것
- outline 상태의 안내 문서 `README.md`
- causal LM / masked LM / span corruption 비교 관점을 정리한 `THEORY.md`
- 선행 개념과 빠른 자기 점검을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - objective별 input-target pair 비교 표
  - loss mask / supervision density 관찰 요약
  - context window별 예측 가능 범위 메모
  - corruption strategy별 common confusion 정리

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 같은 문장을 하나 고정하고, causal LM / masked LM / span corruption이 각각 **입력을 어떻게 바꾸고 정답을 어디에 두는지** 를 나란히 만든다.
2. causal LM에서는 한 칸 shift된 next-token target을, masked LM에서는 일부 위치에만 걸리는 loss를, span corruption에서는 sentinel token 기반 span 복원 target을 비교한다.
3. context window를 고정한 채 objective별로 "각 위치가 무엇을 볼 수 있고 무엇은 직접 예측해야 하는가" 를 정리한다.
4. supervision density를 본다. 모든 시점에 loss가 걸리는지, 일부 mask 위치에만 걸리는지, span 단위 decoder target에 걸리는지를 비교한다.
5. common confusion을 따로 정리한다. 예를 들어 objective와 architecture를 같은 것으로 착각하는지, MLM이 곧 생성 모델이라고 오해하는지, context window를 장기 기억과 같은 것으로 착각하는지 본다.
6. 마지막에는 이 비교가 왜 다음 단위 `05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture`에서 token budget, tokenizer granularity, data mixture 설계로 이어지는지 연결한다.

## 이 단위에서 특히 볼 질문
- 같은 문장을 보더라도 causal LM, masked LM, span corruption은 각각 무엇을 정답으로 삼는가?
- "prediction target framing"이 달라지면 loss가 걸리는 위치와 supervision density는 어떻게 달라지는가?
- context window는 objective마다 어떤 식으로 작동하고, 무엇을 보장하지는 않는가?
- masked LM은 양방향 문맥을 보면서도 왜 정답을 몰래 보는 치팅으로 이해하면 안 되는가?
- span corruption은 token 하나 맞히기와 무엇이 다르고, 왜 encoder-decoder pretraining과 잘 맞는가?
- 이 단위를 이해하면 다음 단위의 tokenizer/data mixture와 이후 domain-adaptive pretraining을 어떤 질문으로 보게 되는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/scratch_lab.py
{
  "status": "sample",
  "objective_views": {
    "causal_lm": {
      "input_shape": [2, 11],
      "target_shape": [2, 11],
      "loss_positions": "all shifted next-token positions",
      "visible_context": "left-only"
    },
    "masked_lm": {
      "input_shape": [2, 12],
      "mask_count": 4,
      "target_shape": [2, 12],
      "loss_positions": "masked positions only",
      "visible_context": "bidirectional within window"
    },
    "span_corruption": {
      "encoder_input_shape": [2, 10],
      "decoder_target_shape": [2, 5],
      "sentinel_count": 2,
      "loss_positions": "decoder targets for corrupted spans"
    }
  },
  "context_window_note": {
    "window_tokens": 12,
    "tokens_scored": {
      "causal_lm": 11,
      "masked_lm": 4,
      "span_corruption": 5
    }
  }
}

$ python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/analysis.py
{
  "status": "sample",
  "observation_points": [
    "target alignment differs by objective",
    "loss mask density changes supervision",
    "context window is not long-term memory"
  ],
  "notes": "expected output/sample shape only"
}
```

핵심은 숫자 자체보다도 **입력-정답 짝을 어떻게 구성했는지**, **loss가 어디에만 걸리는지**, **같은 context window라도 objective가 바뀌면 관찰 가능한 문맥이 어떻게 달라지는지** 를 읽는 것이다.

## 다음 단위와의 연결
이 단위에서 pretraining objective를 예측 타깃 설계 문제로 정리해 두면, 다음 단위 `05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture`에서 "어떤 tokenizer와 corpus mixture가 이 objective를 더 잘 지지하는가" 를 더 구체적으로 보게 된다. 또 이후 `05_advanced_nlp_llm/03_domain_adaptive_pretraining`에서는 같은 objective를 유지한 채 데이터 분포만 바꿨을 때 무엇이 달라지는지도 더 자연스럽게 해석할 수 있다.
