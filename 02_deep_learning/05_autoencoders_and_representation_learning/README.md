# 05 Autoencoders and Representation Learning

> Status: outlined
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 모습** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
지도 라벨이 없어도 입력 자체를 다시 복원해 보게 만들면, 모델은 **무엇을 버리고 무엇을 남겨야 하는가** 를 스스로 배우기 시작한다. autoencoder는 이 과정을 가장 작은 형태로 보여 주는 구조다. 이 단위는 encoder가 입력을 latent representation으로 압축하고, decoder가 그 요약본으로 원래 입력을 얼마나 복원할 수 있는지 보면서, 이후 VAE·self-supervised representation learning·retrieval embedding을 이해하는 바탕을 만든다.

## 이번 단위에서 남길 것
- 학습 목표와 후속 실습 방향을 정리한 `README.md`
- reconstruction objective와 bottleneck 직관을 묶은 `THEORY.md`
- 선행 개념과 빠른 자기 점검을 담은 `PREREQS.md`
- outlined 단계 메타데이터를 담은 `lesson.yaml`
- 후속 실습 산출물이 들어갈 자리만 먼저 만든 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`에 대한 명시적 빈자리

## 실습 흐름
1. 입력 벡터 또는 작은 이미지 patch를 encoder에 넣어 더 작은 latent dimension으로 압축했을 때 어떤 정보가 남고 어떤 정보가 사라지는지 본다.
2. decoder가 latent에서 원래 입력을 다시 복원하도록 두고, reconstruction loss가 줄어드는 방향이 무엇을 의미하는지 해석한다.
3. bottleneck 크기를 바꾸며 latent dimension이 너무 작을 때와 충분할 때 복원 품질이 어떻게 달라지는지 비교한다.
4. 입력에 노이즈를 섞고 복원하게 하는 denoising autoencoder 관점에서, 단순 복사가 아니라 구조를 잡는 representation이 왜 필요한지 본다.
5. 마지막에는 latent space가 단순 압축본을 넘어 retrieval, clustering, anomaly detection, generative modeling의 출발점이 될 수 있음을 질문으로 남긴다.

## 이 단위에서 특히 볼 질문
- reconstruction objective는 "입력을 외운다"는 것과 무엇이 다르고, 언제 유용한가?
- encoder / latent / decoder는 각각 어떤 역할을 맡고, 어디에서 병목이 생기는가?
- bottleneck dimension이 너무 작거나 너무 크면 representation learning 관점에서 어떤 문제가 생기는가?
- denoising autoencoder와 compression-oriented autoencoder는 무엇을 같게 보고 무엇을 다르게 보는가?
- 좋은 latent representation이라고 말하려면 복원 오차 외에 어떤 관찰이 더 필요할까?
- supervised label이 없어도 latent가 downstream task에 도움이 될 수 있는 이유는 무엇인가?

## 실행 결과 예시
아래는 **아직 완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 02_deep_learning/05_autoencoders_and_representation_learning/scratch_lab.py
{
  "input_shape": [4, 64],
  "latent_shape": [4, 8],
  "reconstruction_shape": [4, 64],
  "reconstruction_mse": 0.083,
  "bottleneck_dims_compared": [4, 8, 16]
}

$ python 02_deep_learning/05_autoencoders_and_representation_learning/framework_lab.py
{
  "model": "tiny_autoencoder",
  "encoder_hidden": [32, 8],
  "decoder_hidden": [8, 32],
  "noisy_input_shape": [8, 1, 28, 28],
  "reconstruction_shape": [8, 1, 28, 28],
  "denoising_loss": 0.041,
  "latent_norm_mean": 1.27
}
```

핵심은 숫자 자체보다도 **입력 → latent → reconstruction shape 흐름**, **bottleneck 크기에 따른 복원 변화**, **denoising 설정에서 latent가 노이즈보다 구조를 더 잘 보존하는지** 를 읽는 것이다.

## 다음 단위와의 연결
이 단위에서 latent representation의 의미를 잡아 두면, 다음 단위 `02_deep_learning/06_generative_models_vae_gan`에서 "latent를 단순히 압축에 쓸 것인가, 샘플링 가능한 생성 공간으로 다룰 것인가" 를 더 자연스럽게 비교할 수 있다. 다시 말해 autoencoder는 representation learning의 가장 작은 출발점이고, VAE/GAN은 그 latent를 더 강하게 구조화하거나 생성적으로 활용하는 다음 단계다.
