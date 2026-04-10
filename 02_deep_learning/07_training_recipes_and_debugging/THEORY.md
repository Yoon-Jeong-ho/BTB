# 07 학습 레시피와 디버깅 이론 노트

## 1. learning rate는 속도보다 안정성 레버다
- learning rate는 파라미터를 얼마나 크게 움직일지 정하지만, 실제 체감은 “빠름/느림”보다 **안정성/불안정성**으로 먼저 나타난다.
- 너무 크면 loss가 진동하거나 gradient explosion이 나고, 결국 divergence나 NaN으로 이어질 수 있다.
- 너무 작으면 학습이 매우 느려져 같은 epoch budget 안에서는 underfit처럼 보인다.

## 2. batch size는 gradient noise와 fit 속도를 같이 바꾼다
- 작은 batch는 더 noisy한 gradient를 주지만 같은 epoch 안에서 더 자주 update되므로 빠르게 fit하는 경우가 많다.
- 큰 batch는 더 매끈한 step을 주지만, 같은 epoch budget에서는 update 횟수가 줄어 train loss가 덜 내려갈 수 있다.
- 따라서 batch size는 메모리/throughput만이 아니라 **optimization regime** 자체를 바꾸는 선택이다.

## 3. weight decay와 scheduler는 역할이 다르다
- weight decay는 큰 weight를 눌러 모델이 noisy sample까지 과하게 따라가는 것을 줄인다.
- scheduler는 학습이 진행될수록 step size를 줄여 late-stage oscillation을 완화한다.
- 둘 다 validation loss를 안정화할 수 있지만, 하나는 **파라미터 크기**, 다른 하나는 **시간축의 step 크기**를 다룬다는 점이 다르다.

## 4. overfit / underfit / divergence를 어떻게 읽을까
- overfit: train loss는 매우 낮은데 validation loss가 상대적으로 높게 남는다.
- underfit: train loss 자체가 충분히 낮아지지 못한다.
- divergence: alert, gradient explosion, 급격한 loss 폭증처럼 “학습이 깨지는 순간”이 보인다.
- data bug: train은 어느 정도 움직여도 validation이 비정상적으로 망가지거나, label shift probe처럼 특정 sanity check에서 바로 드러난다.

## 5. sanity check는 큰 sweep보다 먼저다
- single-batch overfit: 모델/optimizer/data path가 최소한 작동하는지 보는 첫 번째 체크다.
- tiny-subset replay: 작은 subset에서 loss가 꾸준히 내려가는지 확인한다.
- shifted/random label probe: 모델 자체 문제와 데이터 misalignment를 구분하는 데 매우 빠르다.
- 좋은 디버깅은 거대한 ablation보다 먼저 **작은 실패 분류 장치**를 갖는 데서 시작한다.

## 실행 결과 예시
```text
scratch 핵심 관측
- baseline final train/val loss: 0.002495 / 0.002083
- weight decay + scheduler final val loss: 0.003818
- high learning rate alert: ['grad_explosion', 'diverged']
- shifted-label bug final val loss: 0.683573

framework 핵심 관측
- baseline tiny MLP final train/val loss: 0.001591 / 0.031259
- weight decay + scheduler tiny MLP final val loss: 0.030674
- high learning rate alert: ['grad_explosion', 'diverged']
- single-batch overfit final loss: 0.000372
```
이 숫자는 실제 runnable lab에서 나오는 deterministic 결과 예시이며, 같은 질문이 scratch와 framework 양쪽에서 반복된다는 점이 핵심이다.

## 이 단위에서 꼭 남겨야 할 질문
- baseline과 regularized recipe의 validation 차이는 얼마나 났는가?
- large batch는 gradient noise를 줄인 대신 fit 속도를 얼마나 희생했는가?
- high learning rate probe는 어느 epoch에서 처음 깨졌는가?
- label shift probe는 overfit와 다른 어떤 흔적을 남겼는가?
- 이 질문들을 이후 LLM fine-tuning 로그에도 그대로 적용할 수 있는가?
