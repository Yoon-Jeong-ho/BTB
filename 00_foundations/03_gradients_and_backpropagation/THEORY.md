# 03 Gradients and Backpropagation 이론 노트

## gradient를 어떻게 읽을까
- gradient는 **어떤 파라미터를 아주 조금 바꿨을 때 loss가 얼마나 변하는지**를 나타내는 기울기다.
- 부호(sign)는 어느 방향으로 움직여야 loss가 줄어드는지 알려주고, 크기(magnitude)는 민감도를 보여준다.
- 그래서 gradient는 단순한 미분 결과가 아니라, optimizer가 따라갈 방향 정보다.

## backpropagation의 핵심
- forward에서는 `입력 -> 중간 표현 -> 예측 -> loss` 순서로 계산한다.
- backward에서는 그 반대로 `loss -> 예측 -> 중간 표현 -> 파라미터` 순서로 local gradient를 곱해 내려온다.
- 이 과정의 핵심은 **chain rule(연쇄법칙)** 이다.
- local gradient 하나만 보면 단순해도, 여러 층이 연결되면 곱셈이 연속되므로 값이 작아지거나 커질 수 있다.

## scratch 예제로 보는 한 단계
이번 scratch 실습은 다음 scalar 모델을 쓴다.

```text
prediction = w * x + b
loss = 0.5 * (prediction - target)^2
```

여기서
- `d(loss)/d(prediction) = prediction - target`
- `d(prediction)/d(w) = x`
- `d(prediction)/d(b) = 1`

이므로
- `d(loss)/d(w) = (prediction - target) * x`
- `d(loss)/d(b) = prediction - target`

처럼 읽을 수 있다. backpropagation은 이렇게 **마지막 오차 신호를 앞단 local derivative와 곱해** 파라미터까지 전달한다.

## finite-difference gradient check가 왜 필요한가
- analytic gradient를 손으로 적었더라도 구현이 틀릴 수 있다.
- finite difference는 `w + eps`, `w - eps` 두 지점의 loss 차이로 기울기를 근사한다.
- analytic gradient와 finite-difference gradient가 매우 비슷하면, backward 구현을 신뢰할 근거가 생긴다.
- 즉 gradient check는 “미분 공식을 외웠는가”보다 “내 구현이 chain rule을 올바르게 반영했는가”를 검증하는 도구다.

## PyTorch autograd에서는 무엇이 달라지나
- tensor에 `requires_grad=True`를 두면 연산 graph가 기록된다.
- `loss.backward()`를 호출하면 각 파라미터의 `.grad`가 채워진다.
- optimizer는 이 `.grad`를 읽어 파라미터를 갱신한다.
- 따라서 framework 단계에서는 수식을 손으로 모두 쓰지 않아도 되지만, **어떤 gradient가 어떤 경로로 생겼는지 해석하는 능력**은 여전히 필요하다.

## 흔한 혼동
- gradient가 크면 “좋다”고 단정하는 실수
- loss는 줄었는데 어떤 파라미터의 gradient는 왜 0에 가까운지 묻지 않는 습관
- finite-difference 근사 오차를 보고도 eps 설정 문제와 구현 오류를 구분하지 못하는 경우
- `optimizer.step()` 전에 `loss.backward()`와 `zero_grad()`의 순서를 헷갈리는 경우

## 실행에서 확인할 포인트
- scratch metrics에서 `grad_w`와 `finite_diff_grad_w`가 거의 같은지 본다.
- `loss_curve.svg`에서 현재 파라미터 위치와 gradient step 이후 위치를 함께 본다.
- framework metrics에서 `.backward()` 이후 gradient norm이 기록되는지 확인한다.
- 한 번의 step 뒤 `loss_after_step`이 `loss_before_step`보다 작아졌는지 본다.

## 실행 결과 예시
```text
scratch metrics
- loss: 0.125
- grad_w: 0.75
- finite_diff_grad_w: 0.75
- updated_loss: 0.056953

framework metrics
- loss_before_step: 0.40695
- loss_after_step: 0.119054
- total_grad_norm: 1.39978
```
이 숫자는 “gradient가 실제로 loss를 줄이는 방향을 가리킨다”는 사실을 작은 예제로 보여준다.
