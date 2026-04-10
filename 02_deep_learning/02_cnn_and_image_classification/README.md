# 02 CNN and Image Classification

> Status: outlined
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 모습** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
`02_deep_learning/01_perceptron_and_mlp`에서 가장 작은 neural classifier를 봤다면, 이제는 **이미지처럼 공간 구조가 중요한 입력을 왜 다른 방식으로 읽어야 하는지** 봐야 한다. 픽셀을 그냥 길게 펴서(flatten) MLP에 넣으면 "서로 가까운 픽셀이 함께 패턴을 만든다"는 감각이 흐려지기 쉽다. 이 단위는 convolution, local receptive field, parameter sharing, pooling을 통해 CNN이 왜 이미지 분류의 기본 출발점이 되었는지 설명한다.

또한 feature map과 channel을 "중간 계층이 무엇을 보고 있는가"라는 관점으로 읽는 연습을 해 두면, 뒤의 multimodal·representation learning 단위에서도 시각 encoder를 더 덜 막연하게 받아들일 수 있다.

## 이번 단위에서 남길 것
- outline 상태의 안내 문서 `README.md`
- convolution / pooling / feature map 직관을 정리한 `THEORY.md`
- 선행 개념과 자기 점검을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 구조화한 `lesson.yaml`
- 후속 실습 산출물이 들어갈 자리 `artifacts/.gitkeep`
- runnable 승격 때 채울 예정인 shape 로그, feature map 관찰, class-logit 비교용 출력 계약

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 아주 작은 흑백 이미지 패치를 두고, kernel이 슬라이딩하면서 local pattern에 점수를 주는 과정을 손계산 또는 toy tensor로 확인한다.
2. 같은 kernel이 여러 위치에 재사용될 때 parameter sharing이 왜 "어디서 나타났는지와 상관없이 비슷한 패턴을 찾는다"는 성질을 만드는지 본다.
3. stride / padding / pooling을 바꿔 가며 feature map의 해상도와 정보 보존량이 어떻게 달라지는지 비교한다.
4. RGB 입력처럼 channel이 여러 개인 경우, convolution이 공간축뿐 아니라 channel 축 정보도 함께 섞는다는 점을 확인한다.
5. 마지막에는 중간 feature map이 분류기 head의 logits로 어떻게 이어지는지 읽고, "중간 표현"과 "최종 class 점수"를 구분하는 연습을 한다.

## 이 단위에서 특히 볼 질문
- convolution은 왜 "작은 패턴 탐지기"처럼 설명할 수 있는가?
- local receptive field는 fully connected layer와 무엇이 다르고, 이미지에서는 왜 더 자연스러운 inductive bias가 되는가?
- pooling은 무엇을 남기고 무엇을 버리며, stride를 키우는 것과 어떤 점이 비슷하고 다른가?
- 입력 channel 수와 출력 feature map 수는 각각 무엇을 의미하는가?
- feature map이 강하게 켜졌다고 해서 곧바로 특정 class가 확정된다고 말할 수 없는 이유는 무엇인가?
- 이미지 분류에서 최종 logits를 읽을 때 중간 표현과 예측 결과를 어떻게 구분해야 하는가?

## 실행 결과 예시
아래는 **아직 완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 02_deep_learning/02_cnn_and_image_classification/scratch_lab.py
{
  "input_shape": [1, 1, 8, 8],
  "conv_kernel_shape": [2, 1, 3, 3],
  "feature_map_shape": [1, 2, 6, 6],
  "pooled_shape": [1, 2, 3, 3],
  "top_activated_region": [2, 4],
  "classification_note": "sample output shape only"
}

$ python 02_deep_learning/02_cnn_and_image_classification/framework_lab.py
{
  "batch_shape": [4, 3, 32, 32],
  "conv1_output_shape": [4, 8, 30, 30],
  "pool_output_shape": [4, 8, 15, 15],
  "logits_shape": [4, 10],
  "predicted_classes": [3, 1, 7, 0],
  "confidence_note": "sample output shape only"
}
```

핵심은 숫자 하나를 맞히는 것이 아니라, **공간 해상도가 어느 단계에서 줄어드는지**, **feature map channel이 어떻게 늘어나는지**, **마지막 logits shape가 class 수와 어떻게 연결되는지**를 읽는 것이다.

## 다음 단위와의 연결
이 단위에서 "공간 구조를 보존한 채 패턴을 쌓아 올리는 법"을 이해해 두면, 다음 단위 `02_deep_learning/03_sequence_models_rnn_lstm_gru`에서 순서 구조를 다루는 recurrent family와 더 선명하게 대비할 수 있다. 하나는 이미지의 **가까운 위치 관계**를, 다른 하나는 시퀀스의 **시간 순서 관계**를 위한 inductive bias라는 점에서 나란히 읽으면 좋다.
