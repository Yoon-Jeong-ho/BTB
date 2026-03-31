# 01 Text Classification 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 bag-of-words baseline과 tiny neural classifier를 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- text classification의 첫 질문은 "무슨 거대한 모델을 쓸까"보다 먼저, 문장을 어떤 token 단위와 어떤 feature 표현으로 읽을지 정하는 것이다.
- bag-of-words baseline이 잘 맞힌다면, dataset 안에 강한 lexical cue가 있다는 뜻이다. 이 신호는 해석 가능하고 빠르게 확인할 수 있다.
- tiny neural classifier는 dense embedding 평균으로 문장 표현을 만든다. 따라서 같은 label이라도 표면 token이 조금 달라진 예문에 더 유연해질 여지가 있다.
- accuracy는 전체 정답률이고, macro F1은 각 클래스를 동등 비중으로 본다. 둘을 같이 읽어야 특정 클래스 collapse를 놓치지 않는다.
- 오분류 문장을 읽을 때는 모델이 틀렸다는 사실만 보지 말고, 어떤 token cue를 과신했는지 또는 어떤 표현을 vocabulary 밖으로 흘려 보냈는지를 함께 봐야 한다.

## 확인 질문
- baseline의 top token signal은 무엇이며, 그것이 왜 분류 근거가 되는가?
- neural classifier가 baseline보다 좋아 보인다면 그것은 어떤 representation 차이를 시사하는가?
- accuracy와 macro F1을 함께 읽을 때 어떤 failure pattern이 더 잘 드러나는가?

## 관련 이론
- [THEORY.md](./THEORY.md): bag-of-words, tiny neural classifier, accuracy, macro F1 핵심 개념을 다시 확인한다.
