# 02 Image Captioning 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 image captioning을 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- captioning은 retrieval처럼 “가장 가까운 정답 하나”를 찾는 문제가 아니라, 여러 가능한 문장 중 **그럴듯한 다음 토큰을 순차적으로 고르는 생성 문제**다.
- 그래서 exact match나 BLEU-1 같은 자동 지표만으로는 충분하지 않다. 실제로 어떤 content token이 hallucination 되었는지, 길이가 과하게 짧아지거나 길어지지 않았는지를 함께 읽어야 한다.
- scratch figure `artifacts/scratch-manual/caption_diagnostics.svg`는 샘플별 caption 길이와 hallucination 수를 바로 보여 주고, 실행별 관측 리포트는 이번 실행에서 어떤 캡션이 틀렸는지 구체적 사례를 남긴다.
- framework decoder는 teacher forcing으로 학습하지만, 평가는 greedy decoding으로 읽는다. 따라서 “학습 loss가 낮아졌는데도 추론 시 문장이 흔들리는가?”를 항상 따로 확인해야 한다.

## 확인 질문
- 이번 실행에서 자동 지표가 좋아져도 사람이 읽었을 때 어색하거나 hallucination 된 caption은 무엇이었는가?
- scratch와 framework의 exact match / unigram precision 차이는 무엇을 말해 주는가?
- decoder가 teacher forcing에서는 맞아도 greedy decode에서는 틀릴 수 있다는 말을 이번 결과로 설명할 수 있는가?

## 관련 이론
- [THEORY.md](./THEORY.md): decoder-style 생성, teacher forcing, hallucination 해석을 다시 확인한다.
