# 02 CNN and Image Classification 돌아보기

## 1. 한 문장 요약
- convolution을 “작은 pattern detector”라는 표현으로 한 문장으로 다시 써 보자.
- pooling이 하는 일을 “남기는 것 / 버리는 것” 구조로 한 문장으로 적어 보자.

## 2. 관측 기록
- `scratch_lab.py`에서 어떤 detector가 세로 막대에 더 크게 반응했는가?
- feature map이 4×4에서 2×2로 줄어들 때 무엇이 더 요약되었다고 느꼈는가?
- `framework_lab.py`의 `conv_weight_shape`, `feature_map_shape`, `logits_shape`를 보고 CNN 흐름을 말로 설명해 보자.

## 3. 헷갈렸던 점
- 입력 channel과 출력 feature map을 처음에는 어떻게 헷갈렸는가?
- pooling이 “정보를 버리는 연산”이라는 말이 처음에는 왜 불편했는가?
- fully connected baseline과 비교했을 때 local receptive field가 왜 더 이미지 친화적인지 아직 막연한 부분이 있는가?

## 4. 다음 단위로 가져갈 질문
- sequence 데이터에서는 local receptive field 대신 어떤 inductive bias가 더 중요해질까?
- recurrent hidden state는 CNN feature map과 어떤 점이 비슷하고 어떤 점이 다를까?
