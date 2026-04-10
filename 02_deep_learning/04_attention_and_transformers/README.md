# 04 Attention and Transformers

> Status: outlined

> 이 단위는 아직 outline 단계다. 아래 실습 흐름과 출력 예시는 **구현 목표를 설명하는 설계 스케치**이며, 현재 이 디렉터리에는 runnable lab 코드가 없다.

## 왜 이 단위를 배우는가
`02_deep_learning/03_sequence_models_rnn_lstm_gru`에서 recurrent family가 시퀀스를 시간축으로 읽는 방식을 봤다면, 이제는 **토큰들을 순서대로 하나씩 넘기지 않고도 서로 섞을 수 있는 방법**을 봐야 한다. 이 단위는 attention을 "점수 계산"이 아니라 **sequence mixing 규칙**으로 읽고, transformer가 왜 recurrent bottleneck을 줄이며 현대 NLP 모델의 기본 블록이 되었는지 모델 패밀리 관점에서 다시 묶는다.

또한 `03_nlp_bridge/02_attention_and_transformer_block`에서 배운 attention weight / mask / block shape 감각을, 이제는 **encoder-only / decoder-only / encoder-decoder 계열을 구분하는 상위 시야**로 확장한다.

## 이번 단위에서 남길 것
- outline 상태의 학습 문서 `README.md`
- 핵심 개념과 관찰 포인트를 정리한 `THEORY.md`
- 선행 개념과 자기 점검을 담은 `PREREQS.md`
- 단위 목표와 질문을 구조화한 `lesson.yaml`
- 향후 실습 산출물을 받을 `artifacts/`
- 이후 구현 단계에서 채울 예정인 attention / transformer 관찰 결과
  - `artifacts/scratch-manual/metrics.json` 예상
  - `artifacts/framework-manual/metrics.json` 예상
  - `artifacts/analysis-manual/latest_report.md` 예상

## 실습 흐름
1. **scratch 관점에서 sequence mixing 다시 보기**
   아주 작은 query / key / value 예제로, 한 토큰이 다른 토큰 표현을 얼마나 섞는지와 `softmax(QK^T)`의 각 행이 왜 "참조 비율"처럼 읽히는지 확인한다.
2. **multi-head intuition 붙이기**
   head를 여러 개 둘 때 각 head가 같은 시퀀스를 서로 다른 기준으로 읽을 수 있다는 점을 관찰하고, "하나의 넓은 attention"과 "여러 개의 좁은 attention"이 직관상 어떻게 다른지 비교한다.
3. **encoder block vs decoder block 구분하기**
   encoder는 보통 bidirectional self-attention으로 전체 문맥을 본다는 점, decoder는 causal mask로 미래를 막고 필요하면 cross-attention으로 encoder 출력을 참조한다는 점을 블록 수준에서 정리한다.
4. **recurrent bottleneck과 비교하기**
   RNN류는 시간축 순차 업데이트가 필요하지만 transformer는 시점별 hidden update를 병렬화할 수 있다는 점, 대신 attention cost가 시퀀스 길이에 따라 커진다는 trade-off를 함께 본다.
5. **NLP bridge와 model family track 연결하기**
   여기서 정리한 encoder / decoder / encoder-decoder 감각이 이후 BERT류, GPT류, seq2seq transformer를 읽는 공통 분류틀이 된다는 점을 정리한다.

## 이 단위에서 특히 볼 질문
- attention output을 왜 "토큰 선택"이 아니라 **sequence mixing 결과**라고 부르는가?
- multi-head는 단순히 파라미터를 늘린 것과 무엇이 다른가?
- encoder block과 decoder block은 둘 다 transformer인데, 어떤 정보 접근 규칙이 가장 다르게 설계되는가?
- transformer는 왜 recurrent family의 시간축 bottleneck을 줄여 주는가?
- NLP bridge에서 본 attention block 감각이 모델 패밀리 단위 분류로 올라가면 무엇이 더 선명해지는가?

## 실행 결과 예시
아래는 **구현 후 기대하는 출력 형태의 예시**다. 완료된 실행 기록이 아니라, 어떤 관찰값을 남기면 좋은지 보여 주는 sample shape이다.

```text
$ python 02_deep_learning/04_attention_and_transformers/scratch_lab.py
{
  "status": "sample",
  "sequence_length": 5,
  "head_count": 1,
  "attention_row_sums": [1.0, 1.0, 1.0, 1.0, 1.0],
  "mixed_token_example": {
    "query_token": "ate",
    "top_keys": ["cat", "fish"]
  },
  "recurrent_steps": 5,
  "parallel_attention_steps": 1
}

$ python 02_deep_learning/04_attention_and_transformers/framework_lab.py
{
  "status": "sample",
  "encoder_hidden_shape": [2, 6, 32],
  "decoder_hidden_shape": [2, 6, 32],
  "self_attention_heads": 4,
  "cross_attention_used": true,
  "causal_mask_blocked_future": true
}
```

핵심은 숫자 하나를 맞히는 것이 아니라, **row sum이 1로 유지되는지**, **head별로 다른 mixing 패턴이 보이는지**, **encoder/decoder가 같은 shape를 유지하면서도 다른 정보 접근 규칙을 쓰는지**를 읽는 것이다.

## 다음 단위와의 연결
이 단위를 통해 transformer를 "attention 있는 블록"이 아니라 **모델 패밀리를 구분하는 기본 골격**으로 읽게 되면, 이후 NLP 트랙에서 encoder-only(BERT류), decoder-only(GPT류), encoder-decoder(seq2seq) 구분이 훨씬 자연스러워진다. 또한 뒤에서 학습 recipe와 추론 비용을 볼 때도, 왜 transformer가 recurrent 병목을 줄이는 대신 메모리/길이 비용을 새로 가져오는지 더 선명하게 해석할 수 있다.
