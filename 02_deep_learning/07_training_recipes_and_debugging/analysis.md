# 07 학습 레시피와 디버깅 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 남긴다.
- 이 문서는 learning rate / batch size / weight decay / scheduler / sanity check를 읽는 고정된 해석 프레임만 유지해, 반복 실행에도 안정적인 기준점을 제공한다.

## 해석 프레임
- learning rate는 단순한 속도 조절값이 아니라, loss 곡선이 매끈하게 내려갈지 발산할지를 결정하는 안정성 레버다.
- batch size는 gradient noise와 step 빈도를 바꾸므로, 같은 epoch budget에서도 fit 속도와 generalization gap 해석이 달라진다.
- weight decay와 scheduler는 모두 late-stage training을 다듬지만, 하나는 파라미터 크기를 누르고 다른 하나는 step size를 줄인다는 점에서 역할이 다르다.
- data bug는 보통 “train loss는 움직이는데 validation이 비정상적으로 망가지는가?”와 “single-batch overfit조차 실패하는가?” 같은 sanity check로 가장 빨리 드러난다.

## 확인 질문
- scratch와 framework에서 baseline 대비 weight decay + scheduler recipe가 어떤 validation trade-off를 만들었는가?
- large batch recipe는 train loss와 validation gap을 어떻게 바꿨는가?
- high learning rate probe에서 first bad epoch와 alert는 무엇이었는가?
- shifted-label bug probe는 baseline보다 얼마나 큰 validation loss를 남겼는가?
- 이번 실행의 sanity check 결과는 `artifacts/analysis-manual/latest_report.md`에 어떻게 정리되었는가?

## 관련 이론
- [THEORY.md](./THEORY.md): learning rate, batch size, weight decay, scheduler, overfit/underfit, divergence, data bug 해석을 다시 연결한다.
