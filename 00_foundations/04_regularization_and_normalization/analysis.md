# 04 Regularization and Normalization 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 normalization / regularization / training dynamics를 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- normalization은 입력/표현의 scale을 정리해 gradient 크기와 optimization 경로를 더 예측 가능하게 만든다.
- regularization은 loss만 빠르게 줄이는 대신, weight norm이나 특정 경로 의존도가 과도하게 커지는 것을 억제한다.
- scratch의 `training_dynamics.svg`는 같은 learning rate에서도 raw feature와 normalized feature가 얼마나 다른 loss 곡선을 만드는지 보여준다.
- framework 관측에서는 `LayerNorm`, `Dropout`, `weight_decay`가 각각 다른 방식으로 안정화/제약을 주는지 확인한다.

## 확인 질문
- normalization이 initial gradient scale을 얼마나 바꿨는가?
- weight decay를 켰을 때 loss 감소와 weight norm 사이 trade-off는 어떻게 읽어야 하는가?
- 이번 실행에서 LayerNorm, dropout, weight decay 관측은 `artifacts/analysis-manual/latest_report.md`에 어떻게 정리되었는가?

## 관련 이론
- [THEORY.md](./THEORY.md): normalization, regularization, LayerNorm, dropout, weight decay의 역할을 다시 확인한다.
