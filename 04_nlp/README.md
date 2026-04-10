# 03 NLP

이 트랙은 현재 `01_text_classification`, `02_named_entity_recognition`, `03_machine_reading_comprehension` 세 unit로 채워져 있으며, `텍스트 전처리 -> bag-of-words baseline -> pretrained LM finetuning -> error analysis` 흐름을 실제 태스크로 반복한다.

## 선행 / 다음 단계

- 선행 권장: [02_nlp_bridge](../02_nlp_bridge/README.md) 의 2개 unit를 먼저 보고 온다.
- 다음 단계: [04_multimodal_bridge](../04_multimodal_bridge/README.md) 로 넘어가며 text-only 표현을 multimodal 표현으로 확장한다.

## 읽는 순서

1. [01_text_classification](01_text_classification/README.md) — 가장 쉬운 NLP applied baseline으로 tokenizer와 classifier를 연결한다.
2. [02_named_entity_recognition](02_named_entity_recognition/README.md) — token-level prediction과 boundary error를 본다.
3. [03_machine_reading_comprehension](03_machine_reading_comprehension/README.md) — span extraction, no-answer, evidence reading을 본다.

한국어 실습을 바로 하고 싶다면 `NSMC` 와 `KLUE` 를 중심으로 시작하고, 영어 표준 벤치마크를 병행하고 싶다면 `IMDb`, `CoNLL-2003`, `SQuAD 2.0` 을 같이 본다.

## 단계 구성

| Stage | 목적 | 추천 데이터셋 | 약한 베이스라인 | 강한 베이스라인 | 남길 figure |
| --- | --- | --- | --- | --- | --- |
| [01_text_classification](01_text_classification/README.md) | 분류 기본기 | NSMC, IMDb, KLUE-TC | TF-IDF + Linear | BERT / RoBERTa finetuning | length histogram, confusion matrix, calibration |
| [02_named_entity_recognition](02_named_entity_recognition/README.md) | 토큰 단위 예측 | KLUE-NER, CoNLL-2003 | CRF / BiLSTM-CRF | Transformer token classification | entity F1, boundary error summary |
| [03_machine_reading_comprehension](03_machine_reading_comprehension/README.md) | span extraction과 불답 처리 | KLUE-MRC, SQuAD 2.0 | BM25 + heuristic / small QA head | pretrained QA finetuning | EM/F1, answerable breakdown, span failure analysis |

## 이 트랙에서 꼭 남길 것

- 문장 길이 분포
- OOV / rare token / subword 특성
- baseline과 transformer의 차이
- 어떤 클래스/엔티티/질문 유형에서 약한지
- 잘못된 예측 예시와 그 원인

## 선택형 확장

- `Transfer learning`: MultiNLI로 intermediate finetuning 후 분류/QA로 옮겨 본다.
- `Cross-lingual evaluation`: XNLI로 zero-shot 일반화와 비용 대비 성능을 본다.
- `Efficiency`: full finetuning, frozen encoder, LoRA/adapter를 비교한다.

실험 운영 규칙은 [../docs/01_experiment_playbook.md](../docs/01_experiment_playbook.md) 를 따른다.
