# 02 Activation and Loss 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 activation/loss를 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- activation은 중간 표현을 비선형으로 바꿔 모델이 더 복잡한 패턴을 표현하게 한다.
- loss는 예측과 정답 사이의 차이를 scalar로 압축해, optimizer가 따라갈 방향을 만든다.
- BCE와 cross entropy는 각각 다른 target 형식과 입력 가정을 가진다. 실행 결과를 볼 때 “확률을 넣었는지, logits를 넣었는지”를 먼저 확인한다.
- scratch figure `artifacts/scratch-manual/activation_curves.svg`는 곡선의 모양 차이를 눈으로 보여주고, observed report는 이번 실행의 수치 차이를 문장으로 풀어준다.

## 확인 질문
- ReLU / sigmoid / tanh 중 어떤 activation이 입력을 가장 강하게 잘랐는가?
- BCE와 cross entropy는 각각 어떤 정답 표현을 기대하는가?
- 이번 실행에서 관측한 구체적 loss 값과 확률 합은 `artifacts/analysis-manual/latest_report.md`에서 어떻게 해석되었는가?

## 관련 이론
- [THEORY.md](./THEORY.md): activation, logits, probability, loss 연결을 다시 확인한다.
