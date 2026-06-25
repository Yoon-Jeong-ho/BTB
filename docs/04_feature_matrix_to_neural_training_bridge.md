# 04 Feature Matrix to Neural Training Bridge

`01_ml`에서 익힌 표형 데이터 baseline 감각을 `02_deep_learning`의 신경망 학습 루프로 옮기기 위한 짧은 다리다. 여기서 목표는 “sklearn 실험이 끝났고 이제 PyTorch가 시작된다”가 아니라, 같은 실험 규율이 어떤 이름으로 바뀌는지 확인하는 것이다.

## 무엇이 그대로 이어지는가

- **데이터 분리**: train/valid/test를 섞지 않는다.
- **baseline 비교**: 새 모델은 항상 더 단순한 기준선과 비교한다.
- **metric 해석**: accuracy, F1, RMSE 같은 숫자는 실패 사례와 함께 읽는다.
- **artifact 저장**: config, metrics, figure, summary를 남겨 다시 비교한다.

## 무엇이 바뀌는가

| 01 ML 언어 | 02 Deep Learning 언어 | 확인할 질문 |
| --- | --- | --- |
| feature matrix `X` | tensor batch `x` | shape가 `(batch, feature)`인지 먼저 확인했는가? |
| `fit()` | training loop | loss를 계산하고 `backward()`와 optimizer step을 호출하는가? |
| feature engineering | learned representation | hidden layer가 어떤 중간 표현을 배우는가? |
| model selection | validation loop | validation metric으로 과적합을 감지하는가? |
| predict/proba | logits/probabilities | logit, activation, probability를 구분하는가? |

## sklearn `fit/predict`와 PyTorch loop의 대응

```text
sklearn
  model.fit(X_train, y_train)
  y_pred = model.predict(X_valid)
  metric(y_valid, y_pred)

PyTorch
  for epoch in range(num_epochs):
    for x, y in train_loader:
      optimizer.zero_grad()
      logits = model(x)
      loss = criterion(logits, y)
      loss.backward()
      optimizer.step()

  with torch.no_grad():
      logits = model(x_valid)
      metric(y_valid, logits)
```

핵심 차이는 PyTorch에서는 학습 신호가 자동으로 숨겨지지 않는다는 점이다. `loss.backward()`가 어떤 parameter에 gradient를 만들고, `optimizer.step()`이 그 값을 어떻게 바꾸는지 직접 확인해야 한다.

`DataLoader`는 `01_ml`에서 한 번에 보던 feature matrix를 mini-batch로 잘라 주는 장치다. `epoch`은 train set 전체를 한 번 훑는 반복 단위이고, `optimizer.zero_grad()`는 이전 batch의 gradient가 다음 batch 계산에 섞이지 않도록 비우는 단계다. 즉 loop의 최소 순서는 보통 `batch 가져오기 → gradient 비우기 → forward → loss → backward → step`으로 읽으면 된다.

## 02 Deep Learning에 들어가기 전 체크

다음 문장을 말할 수 있으면 `02_deep_learning/01_perceptron_and_mlp`로 넘어가도 좋다.

1. feature matrix의 행은 sample이고 열은 feature다.
2. mini-batch tensor의 첫 축은 보통 batch dimension이다.
3. logit은 아직 확률이 아니며, activation/softmax를 거쳐 해석한다.
4. validation metric은 학습 중 선택 기준이고 test metric은 마지막 보고 기준이다.
5. 딥러닝 실험도 baseline, metric, failure analysis 없이는 해석할 수 없다.

## 같이 열어 볼 파일

- [01_ml/README.md](../01_ml/README.md)
- [02_deep_learning/README.md](../02_deep_learning/README.md)
- [00_foundations/03_gradients_and_backpropagation](../00_foundations/03_gradients_and_backpropagation/README.md)
