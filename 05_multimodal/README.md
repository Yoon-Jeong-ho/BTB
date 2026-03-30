# 03 Multimodal

이 트랙의 목표는 `이미지와 텍스트를 같은 표현 공간에서 다루는 법` 과 `생성/추론 태스크에서 멀티모달 모델을 평가하는 법` 을 익히는 것이다.

처음부터 거대한 모델을 직접 끝까지 학습하기보다, 작은 데이터 subset 또는 parameter-efficient finetuning으로 시작하는 것을 기본 원칙으로 한다.

## 단계 구성

| Stage | 목적 | 추천 데이터셋 | 약한 베이스라인 | 강한 베이스라인 | 남길 figure |
| --- | --- | --- | --- | --- | --- |
| [01_image_text_retrieval](01_image_text_retrieval/README.md) | 이미지-텍스트 정렬 | COCO, CxC | frozen CLIP retrieval | VisionTextDualEncoder finetuning | Recall@K, retrieval grid |
| [02_image_captioning](02_image_captioning/README.md) | 이미지 설명 생성 | COCO Captions | pretrained captioner inference | VisionEncoderDecoder / BLIP 계열 finetuning | BLEU/CIDEr table, caption examples |
| [03_visual_question_answering](03_visual_question_answering/README.md) | 시각적 질의응답과 추론 | VQA v2, VizWiz, ScienceQA, NLVR2 | frozen VLM prompting | PEFT finetuning / task head | answer-type breakdown, qualitative panel |

## 추천 데이터셋

| Dataset | Task | 규모/형태 | 왜 좋은가 | 공식 출처 |
| --- | --- | --- | --- | --- |
| MS COCO | captioning / retrieval / detection | 대규모 image-caption benchmark | 캡셔닝과 retrieval의 공통 출발점 | https://cocodataset.org/ |
| Crisscrossed Captions (CxC) | retrieval evaluation / similarity | COCO 확장 human similarity labels | retrieval를 더 정교하게 평가하기 좋다 | https://github.com/google-research-datasets/Crisscrossed-Captions |
| VQA v2 | visual question answering | 204,721 COCO images, 1.1M+ questions | 멀티모달 질의응답의 대표 벤치마크 | https://visualqa.org/ |
| VizWiz-VQA | robust VQA / answerability | 실제 촬영 이미지와 사용자 질문 | 저화질, framing 문제, unanswerable case 분석에 좋다 | https://vizwiz.org/tasks-and-datasets/vqa/ |
| ScienceQA | multimodal reasoning | 21,208 science questions | explanation과 reasoning 분석까지 가능 | https://scienceqa.github.io/ |
| NLVR2 | visual reasoning | image + sentence truth judgment | 정답률뿐 아니라 reasoning 오류 분석에 좋음 | https://github.com/lil-lab/nlvr |

## 이 트랙에서 꼭 남길 것

- qualitative figure를 반드시 남긴다.
- 성능 숫자뿐 아니라 retrieval 사례, caption 예시, 질문별 실패 패턴을 같이 저장한다.
- 대형 모델은 전체 finetuning보다 먼저 frozen encoder, linear probe, LoRA/adapter부터 시도한다.

## 선택형 확장

- `Retrieval 평가 강화`: COCO 학습 후 CxC로 similarity-aware retrieval를 평가한다.
- `Robustness`: VQA v2 다음에 VizWiz로 저화질/실사용 상황을 본다.
- `Reasoning`: ScienceQA나 NLVR2로 단순 matching을 넘어 reasoning failure를 분석한다.

실험 운영 규칙은 [../docs/01_experiment_playbook.md](../docs/01_experiment_playbook.md) 를 따른다.
