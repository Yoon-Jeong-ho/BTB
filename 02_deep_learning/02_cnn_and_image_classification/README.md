# 02 CNN and Image Classification

> Status: runnable
>
> 이 단위는 **CPU-safe toy CNN 실험을 직접 실행하는 runnable 단계**다. convolution을 “커다란 이미지 전체를 한 번에 읽는 연산”이 아니라 **작은 patch를 반복 검사하는 local rule**로 보고, pooling / channel / feature map / 간단한 classification baseline까지 한 흐름으로 묶는다.

## 왜 이 단위를 배우는가
`02_deep_learning/01_perceptron_and_mlp`에서 tiny MLP가 feature space 위에 직선을 긋는 가장 작은 neural baseline이었다면, 이제는 **이미지처럼 위치 관계가 중요한 입력**에서 왜 다른 inductive bias가 필요한지 봐야 한다. CNN은 local receptive field, parameter sharing, pooling을 통해 “가까운 픽셀 패턴을 여러 위치에서 반복 감지한다”는 구조를 만든다.

이 감각이 있어야 뒤에서 attention이나 multimodal encoder를 배울 때도, feature map이 어떻게 쌓이고 왜 공간 해상도가 줄어드는지 덜 막연하게 읽을 수 있다.

## 이번 단위에서 남길 것
- scratch 관측치 `artifacts/scratch-manual/metrics.json`
- scratch figure `artifacts/scratch-manual/cnn_feature_maps.svg`
- framework 관측치 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자용 회고 워크시트 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 6×6 toy RGB-like 이미지 4장을 직접 convolution 한다.
2. vertical detector / horizontal detector 두 개가 **같은 kernel을 여러 위치에 재사용** 하며 feature map을 만든다는 점을 본다.
3. max pooling이 4×4 feature map을 2×2로 줄이면서, 세부 좌표는 일부 버리고 “강한 반응이 있었는가”를 더 압축해 남긴다는 점을 확인한다.
4. pooled feature map 평균값을 class score처럼 읽어, simple image classification baseline이 어떻게 가능한지 본다.
5. `framework_lab.py`에서 같은 toy dataset을 PyTorch `Conv2d` + `MaxPool2d`로 다시 실행해, scratch 직관과 framework shape가 연결되는지 확인한다.
6. `analysis.py`로 관측 숫자를 한국어 문장으로 묶고, 안정적인 해석 프레임(`analysis.md`)과 실행별 리포트(`latest_report.md`)를 분리한다.

## 이번 단위에서 특히 볼 질문
- convolution을 왜 **작은 pattern detector**로 읽을 수 있는가?
- local receptive field는 fully connected baseline과 무엇이 다르고, 이미지에서는 왜 자연스러운 inductive bias가 되는가?
- parameter sharing이 “위치가 달라도 비슷한 패턴을 같은 규칙으로 본다”는 말과 어떻게 연결되는가?
- pooling은 무엇을 남기고 무엇을 버리며, class score baseline에는 어떤 도움을 주는가?
- 입력 channel 수와 출력 feature map 수는 각각 무엇을 의미하는가?

## 실행 결과 예시
아래 예시는 이 디렉터리에서 **실제로 실행되는 command/output shape**를 보여 준다.

```text
$ python 02_deep_learning/02_cnn_and_image_classification/scratch_lab.py
{
  "dataset_shape": [4, 3, 6, 6],
  "conv_kernel_shape": [2, 3, 3, 3],
  "feature_map_shape": [4, 2, 4, 4],
  "pooled_shape": [4, 2, 2, 2],
  "classification_accuracy": 1.0,
  "figure_path": "artifacts/scratch-manual/cnn_feature_maps.svg"
}

$ python 02_deep_learning/02_cnn_and_image_classification/framework_lab.py
{
  "backend": "pytorch",
  "device": "cpu",
  "conv_weight_shape": [2, 3, 3, 3],
  "feature_map_shape": [4, 2, 4, 4],
  "pooled_shape": [4, 2, 2, 2],
  "logits_shape": [4, 2],
  "accuracy": 1.0
}

$ python 02_deep_learning/02_cnn_and_image_classification/analysis.py
# 02 CNN and Image Classification 실행 관측
- local receptive field, parameter sharing, pooling 압축,
  channel/feature map 구분, toy classification baseline을 한국어 리포트로 저장한다.
```

실행 후에는 `cnn_feature_maps.svg`를 눈으로 보면서 **어느 detector가 어느 위치에서 더 강하게 켜지는지** 확인하고, `metrics.json`을 통해 **feature map shape / pooled shape / class score baseline** 을 바로 읽을 수 있다.

## 문서를 읽을 때 볼 포인트
- `README.md`: 무엇을 실행하고 어떤 산출물을 남기는지 먼저 본다.
- `THEORY.md`: convolution, receptive field, pooling, channel vs feature map 직관을 다시 정리한다.
- `analysis.md`: 숫자가 바뀌어도 유지되는 해석 틀만 읽는다.
- `artifacts/analysis-manual/latest_report.md`: 이번 실행에서 실제로 나온 관측 숫자와 해석을 읽는다.

## 다음 단위와의 연결
이 단위에서 “작은 local pattern을 쌓아 큰 분류 신호로 만든다”는 감각을 잡아 두면, 다음 `02_deep_learning/03_sequence_models_rnn_lstm_gru`에서 순서 구조를 위한 recurrent inductive bias를 더 선명하게 대비해서 볼 수 있다. 하나는 **공간 이웃**, 다른 하나는 **시간 순서**를 우선적으로 보는 구조라는 점이 핵심이다.
