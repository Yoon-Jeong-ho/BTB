# 01 Language Modeling and Pretraining Objectives 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 target framing, loss-mask density, context window intuition을 읽는 **안정적인 프레임**만 남겨 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- causal LM / masked LM / span corruption의 핵심 차이는 architecture 이름보다 **무엇을 정답으로 삼는가**에 있다.
- loss-mask density는 supervision이 얼마나 촘촘한지 보여 주지만, density 하나만으로 objective 우열을 정하면 안 된다.
- 같은 context window라도 causal LM은 왼쪽 prefix만, masked LM은 mask 주변 양쪽 문맥을, span corruption은 encoder 입력과 decoder prefix를 다르게 본다.
- span corruption은 sentinel token 덕분에 “빠진 span의 시작과 끝”을 decoder target 안에서 안정적으로 bookkeeping할 수 있다.

## 확인 질문
- target framing만 바뀌어도 model behavior intuition이 달라진다고 왜 말할 수 있는가?
- loss-mask density가 높은 causal LM과 sparse한 masked LM은 각각 어떤 학습 신호를 준다고 볼 수 있는가?
- 같은 context window=4라도 objective별 visible context를 어떻게 다르게 설명할 수 있는가?
- span corruption을 단순한 MLM 확장으로 축소하면 무엇을 놓치게 되는가?

## 관련 이론
- [THEORY.md](./THEORY.md): causal LM, masked LM, span corruption, target framing, context window intuition을 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
