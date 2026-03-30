# 04 Multimodal Bridge

이 구간은 `03_nlp`에서 `05_multimodal`로 넘어가기 전에 필요한 연결 고리다. 텍스트만 보던 표현 학습 감각을 **이미지-텍스트 공동 표현 공간**으로 옮기는 데 초점을 둔다.

## 핵심 목표

- contrastive alignment가 무엇을 맞추는지 이해한다.
- retrieval와 generation이 요구하는 표현 차이를 구분한다.
- cross-attention, grounding, failure case를 멀티모달 실습 전에 익힌다.

## 첫 번째 브리지 단위

1. [01_contrastive_alignment](01_contrastive_alignment/README.md) — 이미지 임베딩과 텍스트 임베딩을 같은 공간에 놓고, 대각선 정답 쌍이 왜 중요해지는지 tiny 예제로 확인한다.

## 학습 태도

텍스트와 이미지가 언제 함께 쓰이고 언제 엇갈리는지 예시 중심으로 확인한다. 특히 이 브리지에서는 “문장 뜻이 맞다”가 아니라 “같은 장면을 가리키는 벡터가 가까워진다”는 감각을 먼저 만든다.

## `05_multimodal`로 이어지는 질문

- retrieval 모델은 왜 caption 전체를 생성하지 않고도 좋은 검색을 할 수 있을까?
- 한 이미지와 여러 문장 설명이 있을 때, 어떤 문장이 hard negative가 될까?
- 같은 표현 공간 정렬이 무너지면 `05_multimodal/01_image_text_retrieval`의 Recall@K가 어떻게 흔들릴까?
