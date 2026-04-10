# 05 Multimodal

이 트랙은 현재 `01_image_text_retrieval`, `02_image_captioning`, `03_visual_question_answering` 세 unit로 채워져 있으며, `이미지와 텍스트를 같은 표현 공간에서 다루는 법` 과 `생성/추론 태스크에서 멀티모달 모델을 평가하는 법` 을 실제 태스크로 반복한다.

## 선행 / 들어오는 길

- 기본 루트: [04_multimodal_bridge](../04_multimodal_bridge/README.md) 를 먼저 읽고 들어온다.
- 딥러닝 코어 보강이 더 필요하면 [00_foundations](../00_foundations/README.md) 과 [02_nlp_bridge](../02_nlp_bridge/README.md) 를 다시 본다.

## 읽는 순서

1. [01_image_text_retrieval](01_image_text_retrieval/README.md) — alignment가 실제 검색 성능으로 어떻게 보이는지 본다.
2. [02_image_captioning](02_image_captioning/README.md) — retrieval에서 generation으로 넘어가며 caption quality와 failure case를 읽는다.
3. [03_visual_question_answering](03_visual_question_answering/README.md) — 질문 유형별 성능과 qualitative failure를 읽는다.

처음부터 거대한 모델을 직접 끝까지 학습하기보다, 작은 데이터 subset 또는 parameter-efficient finetuning으로 시작하는 것을 기본 원칙으로 한다.

## 단계 구성

| Stage | 목적 | 추천 데이터셋 | 약한 베이스라인 | 강한 베이스라인 | 남길 figure |
| --- | --- | --- | --- | --- | --- |
| [01_image_text_retrieval](01_image_text_retrieval/README.md) | 이미지-텍스트 정렬 | COCO, CxC | frozen CLIP retrieval | VisionTextDualEncoder finetuning | Recall@K, retrieval grid |
| [02_image_captioning](02_image_captioning/README.md) | 이미지 설명 생성 | COCO Captions | pretrained captioner inference | VisionEncoderDecoder / BLIP 계열 finetuning | BLEU/CIDEr table, caption examples |
| [03_visual_question_answering](03_visual_question_answering/README.md) | 시각적 질의응답과 추론 | VQA v2, VizWiz, ScienceQA, NLVR2 | frozen VLM prompting | PEFT finetuning / task head | answer-type breakdown, qualitative panel |

## 이 트랙에서 꼭 남길 것

- qualitative figure를 반드시 남긴다.
- 성능 숫자뿐 아니라 retrieval 사례, caption 예시, 질문별 실패 패턴을 같이 저장한다.
- 대형 모델은 전체 finetuning보다 먼저 frozen encoder, linear probe, LoRA/adapter부터 시도한다.

## 선택형 확장

- `Retrieval 평가 강화`: COCO 학습 후 CxC로 similarity-aware retrieval를 평가한다.
- `Robustness`: VQA v2 다음에 VizWiz로 저화질/실사용 상황을 본다.
- `Reasoning`: ScienceQA나 NLVR2로 단순 matching을 넘어 reasoning failure를 분석한다.

실험 운영 규칙은 [../docs/01_experiment_playbook.md](../docs/01_experiment_playbook.md) 를 따른다.
