# 04 Regularization and Normalization 이론 노트

## 왜 normalization이 먼저 등장하는가
- 입력 feature의 스케일이 너무 크면 같은 learning rate에서도 gradient가 과도하게 커질 수 있다.
- 이 경우 optimizer가 “같은 방향으로 천천히 이동”하는 대신, 큰 step으로 튀면서 loss가 급격히 커질 수 있다.
- z-score normalization처럼 평균을 0 근처로 맞추고 표준편차를 1 근처로 맞추면, **gradient scale을 더 예측 가능하게** 만들 수 있다.

## regularization은 무엇을 억제하는가
- regularization은 모델이 training data에 너무 집착해 weight가 과도하게 커지거나, 특정 activation/path에 지나치게 의존하는 것을 막는 장치다.
- **L2 regularization / weight decay**는 큰 weight에 추가 비용을 주어, 같은 loss 감소를 만들더라도 더 작은 norm의 해를 선호하게 만든다.
- **dropout**은 train 단계에서 일부 activation을 무작위로 꺼, 특정 경로에만 의존하는 현상을 줄인다.

## normalization과 regularization은 같은가
아니다. 둘은 목적이 겹쳐 보이지만 초점이 다르다.

- normalization: 입력 또는 중간 표현의 **scale / distribution** 을 안정화해 optimization을 돕는다.
- regularization: 모델이 너무 복잡한 해로 치우치지 않도록 **자유도 / norm / reliance** 를 제어한다.

즉 normalization은 주로 **학습을 더 잘 되게** 만들고, regularization은 주로 **과적합과 과한 weight growth를 줄이게** 만든다. 실제 training dynamics에서는 둘이 함께 작동한다.

## LayerNorm은 무엇을 하나
- `LayerNorm`은 샘플별로 마지막 feature 축의 평균을 빼고 분산으로 나눠, row 단위로 mean 0 / variance 1 근처의 표현을 만든다.
- batch size에 덜 민감하고 sequence 모델에서도 잘 쓰이기 때문에 transformer 계열에서 자주 등장한다.
- 따라서 `LayerNorm`을 볼 때는 “batch 전체 통계”보다 “각 토큰/샘플 내부 feature scale 정렬”을 먼저 떠올리면 좋다.

## weight decay와 dropout을 실행에서 어떻게 읽을까
- weight decay를 켠 optimizer step 후에는, 같은 gradient 조건에서도 **weight norm이 더 작아지는지** 본다.
- dropout은 `train()` 모드에서만 랜덤하게 activation을 0으로 만들고, `eval()` 모드에서는 꺼진다.
- 그래서 dropout을 해석할 때는 “zero fraction이 얼마나 나왔는가”보다, **train/eval 모드가 정말 다르게 동작했는가** 를 먼저 확인해야 한다.

## 흔한 혼동
- normalization이 있으면 regularization이 필요 없다고 생각하는 실수
- weight decay를 “gradient clipping”과 같은 것으로 기억하는 실수
- dropout의 0 출력을 보고 계산 오류라고 오해하는 실수
- train/eval 모드 전환 없이 dropout 결과를 비교하는 실수
- mean 0 / variance 1만 맞으면 학습이 자동으로 잘 된다고 단정하는 실수

## 실행에서 확인할 포인트
- scratch metrics에서 raw feature와 normalized feature의 초기 gradient norm 차이를 본다.
- `training_dynamics.svg`에서 raw/no-regularization, normalized/no-regularization, normalized+L2의 log-loss 곡선을 함께 본다.
- framework metrics에서 `LayerNorm` 출력의 row mean/variance가 어떻게 기록되는지 본다.
- `weight_decay_weight_norm_after_step`과 `no_weight_decay_weight_norm_after_step`을 비교해 weight shrinkage를 확인한다.
- dropout이 `train()`에서는 0을 만들고 `eval()`에서는 입력을 그대로 통과시키는지 확인한다.

## 실행 결과 예시
```text
scratch metrics
- raw_initial_grad_norm: 1600.0
- normalized_initial_grad_norm: 11.18034
- raw_final_loss: 2.443508834485501e+27
- normalized_final_loss: 148.885694
- normalized_l2_weight_norm: 5.112655

framework metrics
- layernorm_row_means: [0.0, 0.0]
- layernorm_row_vars: [1.0, 1.0]
- dropout_train_zero_fraction: 0.5
- weight_decay_weight_norm_after_step: 0.805621
```
이 숫자는 **정규화가 update scale을 안정화하고, regularization이 weight growth를 억제한다** 는 점을 아주 작은 예제로 보여준다.
