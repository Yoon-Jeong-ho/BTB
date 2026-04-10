# 02 Study Guide

이 문서는 BTB를 **한 unit씩 차례대로 공부할 때** 바로 따라갈 수 있는 체크리스트다.

## 1. 가장 먼저 읽을 것

1. 루트 [README.md](../README.md)
2. [00_program_map.md](00_program_map.md)
3. 이 문서
4. foundations/bridge/applied unit는 `README -> THEORY -> (있으면) PREREQS -> scratch/framework -> analysis -> reflection` 순서로, `01_ml` stage는 `README -> THEORY -> 최신 artifact README` 순서로 본다.

## 2. 추천 공부 루트

### 루트 A. 전체 커리큘럼을 순서대로

| 순번 | 경로 | 메모 |
| --- | --- | --- |
| 1 | `00_foundations/01_tensor_shapes` | tensor shape와 broadcasting 감각 |
| 2 | `00_foundations/02_activation_and_loss` | activation, logits, loss |
| 3 | `00_foundations/03_gradients_and_backpropagation` | gradient, chain rule, autograd |
| 4 | `00_foundations/04_regularization_and_normalization` | LayerNorm, dropout, weight decay |
| 5 | `00_foundations/05_gpu_memory_runtime` | runtime, dtype, training vs inference |
| 6 | `01_ml/01_tabular_classification` | 가장 쉬운 applied baseline |
| 7 | `01_ml/02_tabular_regression` | residual과 calibration |
| 8 | `01_ml/03_model_selection_and_interpretation` | validation과 해석 |
| 9 | `01_ml/04_large_scale_tabular` | 비용-성능 trade-off |
| 10 | `02_nlp_bridge/01_tokenization_and_embeddings` | 문장을 id와 embedding으로 바꾸기 |
| 11 | `02_nlp_bridge/02_attention_and_transformer_block` | attention, mask, transformer block |
| 12 | `03_nlp/01_text_classification` | NLP 첫 applied task |
| 13 | `03_nlp/02_named_entity_recognition` | token-level prediction |
| 14 | `03_nlp/03_machine_reading_comprehension` | span extraction과 no-answer |
| 15 | `04_multimodal_bridge/01_contrastive_alignment` | 이미지-텍스트 공동 표현 공간 |
| 16 | `05_multimodal/01_image_text_retrieval` | alignment를 retrieval로 읽기 |
| 17 | `05_multimodal/02_image_captioning` | retrieval에서 generation으로 이동 |
| 18 | `05_multimodal/03_visual_question_answering` | 멀티모달 reasoning |

### 루트 B. 딥러닝 코어를 먼저 굳히기

딥러닝 쪽이 비어 있다고 느끼면 아래 7개만 먼저 본다.

1. `00_foundations/01_tensor_shapes`
2. `00_foundations/02_activation_and_loss`
3. `00_foundations/03_gradients_and_backpropagation`
4. `00_foundations/04_regularization_and_normalization`
5. `00_foundations/05_gpu_memory_runtime`
6. `02_nlp_bridge/01_tokenization_and_embeddings`
7. `02_nlp_bridge/02_attention_and_transformer_block`

이 루트는 **activation / loss / gradient / backprop / attention / runtime** 을 한 축으로 이해하게 만드는 데 초점을 둔다. 멀티모달 bridge는 이 다음에 붙인다.

## 3. 한 unit를 읽는 기본 루틴

### 1단계: 개념 잡기
- `README.md`를 읽으며 이 unit의 목표와 대표 figure를 본다.
- `THEORY.md`에서 용어와 직관을 먼저 잡는다.
- `PREREQS.md`가 있으면 모르는 개념이 있는지 바로 메모한다.

### 2단계: 숫자 흐름 보기
- `scratch_lab.py`를 먼저 본다.
- shape, logits, probability, attention weight, similarity 같은 **중간 숫자**를 확인한다.
- 가능하면 figure와 함께 본다.

### 3단계: 실제 프레임워크와 연결하기
- `framework_lab.py`를 읽는다.
- PyTorch / Transformers / vision-text encoder 같은 실제 모듈 이름과 연결한다.
- `README`에 있는 실행 결과 예시를 같이 본다.

### 4단계: 해석하고 자기 말로 정리하기
- `analysis.md`를 읽고 왜 이런 결과가 나왔는지 정리한다.
- `reflection.md`를 짧게 적으며 스스로 설명 가능한지 확인한다.

## 4. 추천 실행 명령

```bash
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode scratch
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode framework
python scripts/build_lesson_report.py --unit 00_foundations/01_tensor_shapes
```

- 처음에는 foundations unit로 실행 흐름을 익히는 것을 권장한다.
- applied track으로 갈수록 README의 예시 figure와 analysis를 먼저 읽고 실행 여부를 결정해도 된다.

## 5. 공부하면서 꼭 남길 메모

각 unit마다 최소한 아래 질문에 답해 본다.

1. 입력 shape는 무엇이고, 출력 shape는 무엇인가?
2. 중간 representation은 어디서 바뀌는가?
3. loss는 어떤 실수를 벌점으로 주는가?
4. figure가 보여주는 실패 패턴은 무엇인가?
5. 다음 unit로 넘어가기 전에 아직 헷갈리는 개념은 무엇인가?

## 6. 다음 문서

- 전체 프로그램 설명: [00_program_map.md](00_program_map.md)
- 실험 운영 규칙: [01_experiment_playbook.md](01_experiment_playbook.md)
- 시작점: [../00_foundations/README.md](../00_foundations/README.md)
