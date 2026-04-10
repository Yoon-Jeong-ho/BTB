# 01 Language Modeling and Pretraining Objectives 이론 노트

## 핵심 개념

### 1. target framing: 무엇을 정답으로 만들 것인가
pretraining objective는 모델에게 “무엇을 보게 하고 무엇을 맞히게 할까?”를 정하는 규칙이다. 이 선택 하나가
- visible context,
- loss-mask density,
- decoder 사용 여부,
- downstream에서 더 쉽게 얻는 행동
을 함께 바꾼다.

그래서 objective는 단순한 구현 디테일이 아니라 **학습 신호의 방향을 미리 정하는 설계 선택**으로 읽어야 한다.

### 2. causal LM: 다음 토큰을 계속 맞히는 framing
causal LM은 보통 `[x_0, x_1, ..., x_{t-1}] -> x_t` 구조를 반복한다.
- 입력: `<bos> 연구자는 긴 문맥을 천천히 읽는다`
- 타깃: `연구자는 긴 문맥을 천천히 읽는다 <eos>`
- loss-mask density: 거의 모든 시점에 supervision이 걸린다.
- context window 직관: 현재 위치는 **왼쪽 prefix**만 볼 수 있다.

generation과 학습 규칙이 비슷하므로, causal LM은 text continuation 감각과 잘 맞는다.

### 3. masked LM: 일부 위치만 복원하는 framing
masked LM은 원문 일부를 `[MASK]`로 바꾸고 그 위치의 원래 token만 맞힌다.
- 입력: `<bos> 연구자는 긴 [MASK] [MASK] 읽는다 <eos>`
- 타깃: `문맥을`, `천천히` (mask 위치에만 존재)
- loss-mask density: 전체보다 훨씬 희박하다.
- context window 직관: mask된 위치는 **좌우 문맥**을 함께 참고할 수 있다.

여기서 “양방향을 본다”는 것은 정답 token을 그대로 보는 치팅이 아니라, **빈칸 주변 단서**를 함께 활용한다는 뜻이다.

### 4. span corruption: token 하나가 아니라 span을 복원하는 framing
span corruption은 연속된 span을 sentinel token으로 치환하고, decoder가 빠진 조각을 순서대로 복원한다.
- encoder 입력: `연구자는 긴 <extra_id_0> 읽는다 <eos>`
- decoder 타깃: `<extra_id_0> 문맥을 천천히 <extra_id_1>`
- loss-mask density: masked LM보다 촘촘하고, causal LM보다는 보통 더 희박하다.
- context window 직관: encoder는 손상된 문서를 보고, decoder는 **이전 decoder target만** 보며 autoregressive하게 복원한다.

span corruption은 text-to-text, denoising pretraining, encoder-decoder intuition과 잘 이어진다.

### 5. loss-mask density를 어떻게 읽을까
loss-mask density는 “전체 prediction slot 대비 실제로 loss가 걸리는 비율” 정도로 이해할 수 있다.
- causal LM: 높다 → supervision이 촘촘하다.
- masked LM: 낮다 → 일부 token에만 직접 supervision이 있다.
- span corruption: 중간 → encoder는 손상 입력을 보고, decoder에서 빠진 span을 복원한다.

하지만 density가 높다고 objective가 자동으로 더 우월한 것은 아니다. **무엇을 예측하게 했는가**, **downstream과 얼마나 alignment가 있는가**, **corruption 난이도가 어떤가**를 함께 봐야 한다.

### 6. context window intuition: 창문 크기와 예측 규칙은 다르다
context window는 “한 번의 forward에서 참고할 수 있는 토큰 budget”이고, objective는 “그 토큰 중 무엇을 보게 하고 무엇을 맞히게 하는가”다.
- causal LM: window 안에서도 미래는 못 본다.
- masked LM: mask 위치는 window 안의 좌우 문맥을 함께 본다.
- span corruption: encoder는 손상된 전체 문서를 보고, decoder는 sentinel 이후 span을 순차 복원한다.

즉 **같은 context window라도 objective가 달라지면 실제 학습 신호는 전혀 다를 수 있다.**

## 자주 헷갈리는 지점
- objective와 architecture를 같은 것으로 생각하는 실수
- masked LM을 곧 free-form generation objective로 생각하는 실수
- span corruption을 “mask를 많이 친 MLM” 정도로 축소하는 실수
- context window를 곧 long-term memory라고 생각하는 실수
- loss-mask density가 높으면 항상 더 좋다고 단정하는 실수

## 실행 결과 예시
```text
$ python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/scratch_lab.py
{
  "objectives": {
    "causal_lm": {"loss_mask_density": 1.0, "target_framing": "next-token prediction"},
    "masked_lm": {"loss_mask_density": 0.333333, "target_framing": "recover masked tokens only"},
    "span_corruption": {"loss_mask_density": 0.666667, "target_framing": "decoder reconstructs missing span"}
  }
}

$ python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/framework_lab.py
{
  "density_ranking": ["causal_lm", "span_corruption", "masked_lm"],
  "context_window": {
    "causal_future_blocked": true,
    "masked_middle_token_sees_both_sides": true,
    "span_decoder_reads_previous_targets_only": true
  }
}
```

## 이 단위에서 끝까지 남겨야 할 질문
- 같은 문장을 causal LM, masked LM, span corruption으로 바꾸면 target framing은 각각 어떻게 달라지는가?
- loss-mask density 차이는 supervision 감각을 어떻게 바꾸는가?
- 같은 context window여도 objective별 visible context는 왜 다르게 느껴지는가?
- 이 차이가 tokenizer/data mixture나 later domain-adaptive pretraining 해석에 어떤 질문을 남기는가?
