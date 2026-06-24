# 07 Multimodal Generation Bridge

이 문서는 `08_multimodal_bridge/01_contrastive_alignment`의 shared embedding retrieval 감각에서 `09_multimodal`의 captioning, VQA, multimodal reasoning으로 넘어가기 위한 다리다. retrieval은 “같은 장면을 가까운 벡터로 놓는 일”이고, generation/VQA는 “이미지 정보를 token 생성 또는 답변 선택 과정에 계속 주입하는 일”이다.

## Shared embedding만으로 부족한 이유

Contrastive retrieval은 이미지 벡터와 텍스트 벡터의 거리만 잘 맞추면 된다. 하지만 captioning과 VQA는 다음을 요구한다.

- 이미지의 여러 부분을 순서대로 참고한다.
- 질문 token과 이미지 region이 상호작용한다.
- 답변이나 caption token을 생성하면서 이전 token과 시각 정보를 함께 본다.
- 실패가 retrieval failure가 아니라 grounding failure일 수 있다.

## Cross-attention vs shared embedding retrieval

- **shared embedding retrieval**: 이미지 전체와 문장 전체를 각각 하나의 벡터로 요약한 뒤 similarity를 계산한다.
- **cross-attention**: text token이 image patch/region feature를 바라보며 필요한 정보를 골라 온다.

예를 들어 “빨간 컵의 왼쪽에 무엇이 있나?”라는 질문은 전체 이미지-문장 similarity보다, `빨간 컵`, `왼쪽`, `무엇` token이 어떤 region을 보는지가 중요하다.

## Encoder-decoder multimodal generation

Captioning에서는 보통 다음 흐름을 생각한다.

1. vision encoder가 image feature를 만든다.
2. text decoder가 지금까지 만든 caption token을 본다.
3. decoder cross-attention이 image feature를 참조한다.
4. 다음 caption token을 생성한다.
5. stop token까지 반복한다.

따라서 caption 품질은 language fluency만이 아니라 image grounding과 decoding policy에 의해 같이 결정된다.

## VQA fusion

VQA는 captioning보다 짧은 답을 만들 수 있지만, 질문과 이미지의 결합이 더 중요하다.

- 질문이 색, 위치, 개수, 행위 중 무엇을 묻는지 분류한다.
- image feature에서 해당 evidence를 찾는다.
- answer vocabulary 또는 decoder가 답을 낸다.
- answer-type별 정확도를 따로 본다.

숫자 하나의 accuracy만 보면 color question은 잘 맞히지만 counting question은 계속 틀리는 문제를 놓친다.

## Grounding failure vs retrieval failure

- **retrieval failure**: 맞는 image-text pair를 가까이 두지 못한다.
- **grounding failure**: 이미지는 봤지만 질문/문장 속 특정 대상, 위치, 관계를 잘못 연결한다.
- **generation failure**: 시각 정보는 맞게 봤지만 decoder가 반복, hallucination, generic answer로 무너진다.

`09_multimodal`에서는 이 세 failure를 같은 것으로 취급하지 말고, figure와 예시 표에서 분리해 적는다.

## `09_multimodal`에 들어가기 전 체크리스트

- cross-attention이 왜 retrieval similarity보다 더 세밀한 연결인지 설명할 수 있다.
- caption hallucination과 retrieval mismatch를 구분할 수 있다.
- VQA answer-type breakdown이 왜 필요한지 말할 수 있다.
- qualitative panel에서 이미지, 질문/캡션, 모델 출력, 실패 원인을 함께 볼 수 있다.

## 최소 실험 아이디어

- 같은 이미지에 retrieval caption, generated caption, VQA question을 하나씩 붙인다.
- 실패를 `retrieval`, `grounding`, `generation` 중 하나로 라벨링한다.
- held-out 또는 adversarial 예시 2~4개를 만들어 “학습 데이터에서는 맞지만 새 조합에서는 틀리는” 경우를 기록한다.
