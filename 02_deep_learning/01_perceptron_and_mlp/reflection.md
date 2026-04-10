# 01 Perceptron and MLP 돌아보기

## 1. 한 문장 요약
- perceptron의 decision rule을 내 말로 다시 써 보자.
- tiny MLP가 왜 필요한지 한 문장으로 적어 보자.

## 2. 관측 기록
- `scratch_lab.py`에서 본 선형 분리 가능 데이터의 핵심 관측은 무엇이었는가?
- XOR에서 single neuron이 실패한 이유를 "직선 하나"라는 표현을 써서 설명해 보자.
- `framework_lab.py`에서 tiny MLP가 single neuron보다 좋아진 수치를 적고, 그 차이를 어떻게 해석할지 적어 보자.

## 3. 헷갈렸던 점
- 처음에는 optimizer 문제처럼 보였지만, 다시 보니 표현력 문제였던 부분이 있었는가?
- hidden layer가 추가되면 어떤 종류의 decision boundary가 가능해진다고 느꼈는가?

## 4. 다음 단위로 가져갈 질문
- 이미지에서는 왜 공간 구조를 읽는 CNN이 더 유리할까?
- sequence에서는 hidden state나 attention이 어떤 식으로 perceptron/MLP의 아이디어를 확장할까?
