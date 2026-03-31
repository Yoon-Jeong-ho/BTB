# 04 Regularization and Normalization

## 왜 이 단위를 배우는가
activation과 gradient를 본 다음에는, **학습이 왜 흔들리고 어떤 장치가 그 흔들림을 줄이는지**를 봐야 한다. 이 단위는 입력 정규화(normalization), `L2 / weight decay` 같은 regularization, dropout, LayerNorm이 **training dynamics를 어떻게 바꾸는지**를 작은 숫자와 그림으로 확인하게 만든다.

## 이번 단위에서 남길 것
- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- scratch figure `artifacts/scratch-manual/training_dynamics.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 큰 스케일의 raw feature와 z-score 정규화 feature를 같은 learning rate로 학습시켜, gradient scale과 loss 곡선이 어떻게 달라지는지 본다.
2. 같은 scratch 실험 안에서 `L2 regularization`을 추가해, loss 감소 속도와 weight norm이 어떻게 달라지는지 비교한다.
3. `framework_lab.py`에서 PyTorch `LayerNorm`, `Dropout`, `weight_decay`가 tiny tensor / tiny linear model에서 어떤 수치 행동을 보이는지 확인한다.
4. `analysis.py`로 실행별 관측치를 한국어 문장으로 정리하고, 안정적인 해석 문서와 실행별 리포트를 분리한다.

## 실행 결과 예시
```text
$ python 00_foundations/04_regularization_and_normalization/scratch_lab.py
{
  "raw_initial_grad_norm": 1600.0,
  "normalized_initial_grad_norm": 11.18034,
  "raw_final_loss": 2.443508834485501e+27,
  "normalized_final_loss": 148.885694,
  "normalized_l2_weight_norm": 5.112655,
  "figure_path": "artifacts/scratch-manual/training_dynamics.svg"
}

$ python 00_foundations/04_regularization_and_normalization/framework_lab.py
{
  "layernorm_row_means": [0.0, 0.0],
  "layernorm_row_vars": [1.0, 1.0],
  "dropout_train_zero_fraction": 0.5,
  "weight_decay_weight_norm_after_step": 0.805621
}
```
실행 후에는 SVG figure와 metrics JSON이 `artifacts/` 아래에 쌓여, **정규화가 gradient scale을 줄이고, regularization이 weight norm을 눌러주는 흐름**을 바로 확인할 수 있다.

## 문서를 읽을 때 볼 포인트
- `README.md`: 무엇을 실행하고 어떤 산출물을 남기는지 먼저 본다.
- `THEORY.md`: normalization / regularization / dropout / weight decay가 각각 어떤 문제를 겨냥하는지 정리한다.
- `analysis.md`: 숫자가 바뀌어도 유지되는 해석 프레임을 본다.
- `artifacts/analysis-manual/latest_report.md`: 이번 실행에서 실제로 나온 loss, gradient, weight norm을 읽는다.

## 다음 단위와의 연결
이 감각은 이후 optimizer, attention 안정화, GPU batch tuning, multimodal training recipe를 읽을 때 그대로 쓰인다. 특히 **“loss가 흔들리는 원인이 입력 scale인지, 과한 weight growth인지, stochastic regularization 때문인지”** 를 구분하는 습관이 중요하다.
