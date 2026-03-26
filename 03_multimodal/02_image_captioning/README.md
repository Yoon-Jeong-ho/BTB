# 02 Image Captioning

## 목표

이미지를 받아 텍스트를 생성하는 과정에서 `decoder behavior`, `hallucination`, `metric vs human judgment 차이` 를 익힌다.

## 추천 데이터셋

- `MS COCO Captions`

## 실습 파이프라인

1. pretrained caption model로 inference baseline
2. caption length, vocabulary, reference diversity 분석
3. VisionEncoderDecoder 또는 BLIP 계열의 소규모 finetuning
4. BLEU, CIDEr, SPICE 등 자동 지표 비교
5. 좋은 caption과 hallucination caption 예시 정리

## 결과로 남길 figure

- metric comparison table
- caption length distribution
- qualitative caption panel

## 분석으로 남길 figure

- hallucination 사례 모음
- object count / scene complexity별 성능
- reference diversity 대비 score 분석

## 승격 기준

- 자동 지표뿐 아니라 사람이 봐도 caption quality 변화가 납득된다.
- hallucination 유형이 정리되어 있다.
