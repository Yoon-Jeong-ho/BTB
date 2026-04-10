# 04 Attention and Transformers

> Status: runnable
>
> 이 단위는 **CPU-safe toy attention 실험을 직접 실행해 보는 runnable 단계**다. attention을 “점수 계산 공식”이 아니라 **토큰들이 서로를 섞는 sequence mixing 규칙**으로 읽고, multi-head intuition, encoder/decoder 구분, recurrent bottleneck 완화를 한 번에 묶는다.

## 왜 이 단위를 배우는가
`02_deep_learning/03_sequence_models_rnn_lstm_gru`에서 recurrent family가 시퀀스를 시간축으로 압축하는 방식을 봤다면, 이제는 **각 위치가 필요한 다른 위치를 직접 참조하는 방식**으로 넘어와야 한다. transformer는 recurrence를 완전히 지운 마법 블록이 아니라, **정보 전달 경로를 짧게 만들고 병렬 계산을 가능하게 하는 attention family**로 읽는 것이 핵심이다.

또한 `03_nlp_bridge/02_attention_and_transformer_block`에서 attention row sum, mask, transformer block shape를 보았다면, 여기서는 그 감각을 **encoder-only / decoder-only / encoder-decoder** 모델 패밀리 시야로 올려서 다시 정리한다.

## 이번 단위에서 남길 것
- scratch attention 관측치 `artifacts/scratch-manual/metrics.json`
- scratch attention heatmap `artifacts/scratch-manual/attention_patterns.svg`
- framework attention/transformer 관측치 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자 회고 질문 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 아주 작은 query/key/value 예제로 attention row가 왜 1로 합쳐지고, output이 왜 value들의 가중합인지 계산한다.
2. 같은 시퀀스를 두 개 head로 읽어 보며, **한 head는 가까운 문맥**, 다른 head는 **조금 더 긴 문맥**을 강조하도록 만들어 multi-head intuition을 잡는다.
3. 같은 attention score를 encoder 규칙(전체 문맥 허용)과 decoder 규칙(causal mask로 미래 차단)에 각각 적용해, “누구를 볼 수 있는가”가 어떻게 달라지는지 비교한다.
4. `framework_lab.py`에서 PyTorch `MultiheadAttention`과 작은 transformer-ish decoder block을 CPU에서 실행하며, encoder self-attention / decoder masked self-attention / cross-attention을 한 번에 관찰한다.
5. `analysis.py`로 row sum, head diversity, encoder/decoder 접근 규칙, recurrent bottleneck relief를 한국어 문장으로 묶는다.

## 이번 단위에서 특히 볼 질문
- attention output을 왜 “토큰 하나 선택”이 아니라 **sequence mixing 결과**라고 읽어야 하는가?
- multi-head는 단순히 파라미터를 늘린 것과 무엇이 다른가?
- encoder block과 decoder block은 둘 다 transformer인데, **정보 접근 규칙**은 어떻게 다른가?
- transformer는 recurrent family의 어떤 bottleneck을 줄이고, 대신 어떤 비용을 새로 가져오는가?
- encoder-only / decoder-only / encoder-decoder 구분을 할 때 attention mask와 cross-attention 유무가 왜 중요한가?

## 실행 결과 예시
아래 예시는 이 디렉터리에서 **실제로 실행되는 command/output shape**를 보여 준다.

```text
$ python 02_deep_learning/04_attention_and_transformers/scratch_lab.py
{
  "sequence_length": 5,
  "max_row_sum_error": 0.0,
  "multi_head": {
    "head_count": 2,
    "distinct_top_key_counts": [1, 1, 2, 2, 1]
  },
  "encoder_decoder": {
    "encoder_future_access_mass": 0.465313,
    "causal_mask_future_blocked": true
  },
  "figure_path": "artifacts/scratch-manual/attention_patterns.svg"
}

$ python 02_deep_learning/04_attention_and_transformers/framework_lab.py
{
  "device": "cpu",
  "num_heads": 2,
  "encoder_hidden_shape": [2, 5, 8],
  "decoder_hidden_shape": [2, 5, 8],
  "cross_attention_used": true,
  "encoder_future_attention_mean": 0.18210457,
  "decoder_future_attention_max": 0.0
}

$ python 02_deep_learning/04_attention_and_transformers/analysis.py
# 04 Attention and Transformers 실행 관측
- row sum 오차, head별 top-key 차이, encoder/decoder 접근 규칙,
  recurrent bottleneck relief를 한국어 관측 리포트로 저장한다.
```

실행 후에는 `attention_patterns.svg`를 눈으로 보면서 **head마다 어떤 토큰을 더 강하게 섞는지**, `metrics.json`을 읽으면서 **row sum / causal mask / cross-attention 사용 여부**를 바로 확인할 수 있다.

## 다음 단위와의 연결
이 단위에서 “attention = sequence mixing”, “decoder = 미래 차단”, “encoder-decoder = cross-attention 추가” 감각을 잡아 두면, 이후 NLP/model-family 단위에서 BERT류 / GPT류 / seq2seq transformer를 훨씬 빠르게 분류할 수 있다. 또한 training recipe를 볼 때도, 왜 transformer가 recurrent bottleneck을 줄이는 대신 **길이 제곱 비용**을 가져오는지 더 선명하게 이해하게 된다.
