# 02 CNN and Image Classification 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 CNN을 해석하는 **안정적인 프레임**만 남겨, 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- convolution은 작은 patch를 읽는 local rule이다. 따라서 feature map은 “이미지 전체 의미”보다 “어느 위치에서 어떤 detector가 켜졌는가”를 먼저 보여 준다.
- local receptive field는 출력 하나가 입력 전체가 아니라 작은 이웃만 본다는 뜻이다. 이미지에서는 이 제한이 오히려 inductive bias가 된다.
- parameter sharing은 같은 kernel이 여러 위치에서 재사용된다는 뜻이다. 같은 막대 패턴이 왼쪽/오른쪽 어디에 있어도 같은 detector가 반응할 수 있는 이유가 여기 있다.
- pooling은 중요한 반응을 남기고 해상도를 줄인다. 따라서 위치 정보 일부는 버리지만 class score baseline으로 넘어갈 때 더 압축된 표현을 만든다.
- 입력 channel 수와 출력 feature map 수는 서로 다른 개념이다. 전자는 데이터 관측 축, 후자는 detector 개수에 가깝다.

## 확인 질문
- local receptive field가 이미지 문제에서는 왜 도움이 되는가?
- parameter sharing이 translation-like robustness와 어떻게 연결되는가?
- pooling이 남기는 정보와 버리는 정보를 구분해서 설명할 수 있는가?
- 입력 channel과 feature map을 서로 다른 말로 설명할 수 있는가?
- 실행별 숫자를 왜 `analysis.md`가 아니라 `latest_report.md`에 남겨야 하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): convolution, pooling, channel/feature map, toy classification baseline을 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
