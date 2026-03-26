# 02 NLP

이 트랙의 목표는 `텍스트 전처리 -> bag-of-words baseline -> pretrained LM finetuning -> error analysis` 흐름을 익히는 것이다.

한국어 실습을 바로 하고 싶다면 `NSMC` 와 `KLUE` 를 중심으로 시작하고, 영어 표준 벤치마크를 병행하고 싶다면 `IMDb`, `CoNLL-2003`, `SQuAD 2.0` 을 같이 본다.

## 단계 구성

| Stage | 목적 | 추천 데이터셋 | 약한 베이스라인 | 강한 베이스라인 | 남길 figure |
| --- | --- | --- | --- | --- | --- |
| [01_text_classification](01_text_classification/README.md) | 분류 기본기 | NSMC, IMDb, KLUE-TC | TF-IDF + Linear | BERT / RoBERTa finetuning | length histogram, confusion matrix, calibration |
| [02_named_entity_recognition](02_named_entity_recognition/README.md) | 토큰 단위 예측 | KLUE-NER, CoNLL-2003 | CRF / BiLSTM-CRF | Transformer token classification | entity F1, boundary error summary |
| [03_machine_reading_comprehension](03_machine_reading_comprehension/README.md) | span extraction과 불답 처리 | KLUE-MRC, SQuAD 2.0 | BM25 + heuristic / small QA head | pretrained QA finetuning | EM/F1, answerable breakdown, span failure analysis |

## 추천 데이터셋

| Dataset | Task | 규모/형태 | 왜 좋은가 | 공식 출처 |
| --- | --- | --- | --- | --- |
| NSMC | sentiment classification | 한국어 영화 리뷰 | 가장 빠르게 한국어 분류 실습 가능 | https://github.com/e9t/nsmc |
| IMDb Reviews | sentiment classification | 25k train + 25k test | 영어 분류의 고전적 baseline 벤치마크 | https://www.tensorflow.org/datasets/catalog/imdb_reviews |
| KLUE | Korean NLU benchmark | 8개 태스크 | 한국어 분류, NER, MRC까지 한 벤치마크 안에서 확장 가능 | https://github.com/KLUE-benchmark/KLUE |
| CoNLL-2003 | NER | shared task benchmark | 영어 NER의 표준 출발점 | https://aclanthology.org/W03-0419/ |
| SQuAD 2.0 | reading comprehension | 100k+ answerable + 50k+ unanswerable | span QA와 abstention을 함께 연습 가능 | https://rajpurkar.github.io/SQuAD-explorer/ |
| MultiNLI | natural language inference | 433k sentence pairs | 전이학습용 intermediate task로 좋고, genre generalization을 보기 좋다 | https://cims.nyu.edu/~sbowman/multinli/ |
| XNLI | cross-lingual NLI | 15개 언어 dev/test | 영어 학습 후 다국어 zero-shot 일반화를 점검하기 좋다 | https://github.com/facebookresearch/XNLI |

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
