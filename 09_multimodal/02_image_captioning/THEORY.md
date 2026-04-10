# 02 Image Captioning 이론 노트

## 핵심 개념
- **image captioning**은 이미지 representation을 조건으로 두고 다음 토큰을 순차적으로 생성하는 multimodal generation 문제다.
- **decoder-style generation**은 이전 토큰과 이미지 조건을 함께 받아 다음 토큰 분포를 예측한다.
- **teacher forcing**은 학습 시 모델이 직전 예측 대신 정답 이전 토큰을 입력으로 받아 더 안정적으로 학습하게 하는 방법이다.
- **greedy decoding**은 매 시점마다 확률이 가장 높은 토큰 하나만 고르는 단순 추론 전략이다.
- **hallucination**은 이미지에 없는 객체/속성을 caption에 넣는 현상이다.

## 수식 / 직관
- 이미지 특징을 `v`, 이전 토큰들을 `y_{<t}` 라고 하면 captioning은 `P(y_t | y_{<t}, v)` 를 반복해서 계산하는 문제로 볼 수 있다.
- teacher forcing 학습에서는 `y_{t-1}` 자리에 모델 예측이 아니라 정답 토큰을 넣는다.
- 그래서 학습 loss가 낮아져도, 실제 추론에서 모델이 한 번 틀린 토큰을 내면 그 이후 문맥이 달라져 오류가 커질 수 있다.
- BLEU-1 같은 unigram 지표는 토큰 겹침을 빠르게 볼 수 있지만, 문장 전체 자연스러움이나 hallucination severity를 완전히 설명하지는 못한다.

## 이 단위에서 꼭 볼 것
- scratch captioner는 왜 `dog` prior 때문에 해변의 연 장면을 잘못 설명했는가?
- exact match와 unigram precision이 서로 다른 메시지를 줄 때 무엇을 더 확인해야 하는가?
- framework decoder의 teacher forcing token accuracy가 높아도, greedy decode exact match를 따로 봐야 하는 이유는 무엇인가?
- hallucination token 수와 caption length를 같이 보면 어떤 qualitative 힌트를 얻을 수 있는가?

## Common Confusion
- captioning을 retrieval처럼 “가장 가까운 reference 하나 찾기”로만 오해하는 실수
- BLEU/CIDEr 같은 자동 지표만 높으면 caption 품질이 완벽하다고 생각하는 실수
- teacher forcing 학습 지표와 실제 greedy generation 결과를 같은 것으로 보는 실수
- hallucination을 단순 오타 수준으로 과소평가하는 실수

## PyTorch tiny decoder demo에서 보는 구조
- 이 unit의 `framework_lab.py`는 대형 VisionEncoderDecoder가 아니라, **작은 이미지 projection + token embedding + GRU decoder** 조합으로 captioning의 핵심 흐름만 재현한다.
- 즉 image feature는 hidden state 초기값과 매 step conditioning에 들어가고, decoder는 토큰 시퀀스를 따라 다음 단어를 예측한다.
- CPU-safe toy demo라도 “image condition + autoregressive decoder + teacher forcing / greedy decode 차이”는 충분히 볼 수 있다.

## 실행 결과 예시
```text
scratch metrics
- exact_match_rate: 0.75
- corpus_unigram_precision: 0.875
- hallucinated_content_tokens_total: 1
- figure_path: artifacts/scratch-manual/caption_diagnostics.svg

framework metrics
- device: cpu
- token_accuracy: 1.0
- exact_match_rate: 1.0
- corpus_unigram_precision: 1.0
- hallucinated_content_tokens_total: 0
```
이 숫자는 “토큰 몇 개가 맞았는가”를 넘어, **실제 caption 문장이 사람 눈에 그럴듯한가, hallucination이 줄었는가, teacher forcing이 greedy decode까지 이어졌는가**를 같이 읽어야 image captioning을 제대로 해석할 수 있음을 보여 준다.
