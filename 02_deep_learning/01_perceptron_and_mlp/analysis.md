# 01 Perceptron and MLP 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 perceptron과 tiny MLP를 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- perceptron의 decision rule은 `w·x + b`의 부호 하나로 클래스를 가른다. 따라서 decision boundary는 한 줄의 직선(또는 고차원에서는 hyperplane)이다.
- 어떤 toy 데이터가 linear separable하면 single neuron 하나로도 완벽하게 맞출 수 있다. 이 경우 모델이 약해서가 아니라, 문제 자체가 직선 하나로 충분한 것이다.
- XOR처럼 대각선 패턴을 요구하는 데이터는 single neuron의 표현력 한계에 걸린다. 이때 accuracy가 안 나오는 이유를 optimizer 탓으로만 보면 안 된다.
- hidden layer와 nonlinearity가 들어간 tiny MLP는 입력을 한 번 더 재표현해서, 직선 하나로는 안 되던 문제를 풀 수 있다.

## 확인 질문
- single neuron이 잘 되는 경우와 안 되는 경우를 decision boundary 관점에서 어떻게 구분할 수 있는가?
- XOR 실패는 학습률 문제라기보다 표현력 문제라는 말을 어떤 관측으로 설명할 수 있는가?
- 이번 실행의 실제 숫자는 왜 `analysis.md`가 아니라 `artifacts/analysis-manual/latest_report.md`에 남겨야 하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): perceptron decision rule, linear separability, hidden layer의 역할을 다시 확인한다.
