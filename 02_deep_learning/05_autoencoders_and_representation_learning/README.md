# 05 Autoencoders and Representation Learning

> Status: runnable
>
> 이 단위는 **CPU-safe / deterministic / toy reconstruction data only** 조건에서 autoencoder를 직접 실행해 보는 runnable 단계다. reconstruction objective가 encoder → latent → decoder 흐름을 어떻게 만들고, bottleneck·denoising·compression 변형이 어떤 차이를 만드는지 숫자와 SVG로 바로 확인한다.

## 왜 이 단위를 배우는가
지도 라벨이 없어도 입력 자체를 다시 복원하게 만들면, 모델은 **무엇을 버리고 무엇을 남겨야 하는가** 를 스스로 배우기 시작한다. autoencoder는 이 과정을 가장 작은 형태로 보여 주는 구조다. 이 단위는 encoder가 입력을 latent representation으로 압축하고, decoder가 그 요약본으로 원래 입력을 얼마나 복원할 수 있는지 보면서, 이후 VAE·self-supervised representation learning·retrieval embedding을 이해하는 바탕을 만든다.

## 이번 단위에서 남길 것
- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- scratch figure `artifacts/scratch-manual/autoencoder_bottleneck.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 손으로 만든 저차원 subspace toy 데이터를 basis projection 방식으로 압축/복원해, reconstruction objective와 bottleneck 직관을 먼저 잡는다.
2. latent dimension을 1, 2, 3으로 바꾸면서 reconstruction mse가 어떻게 달라지는지 본다.
3. noisy input을 같은 bottleneck에 통과시켜 denoising variant가 raw noisy baseline보다 얼마나 나아지는지 확인한다.
4. `framework_lab.py`에서 tiny PyTorch autoencoder를 full-batch CPU 학습으로 다시 관찰해, narrow bottleneck / compression bottleneck / denoising bottleneck을 비교한다.
5. `analysis.py`로 reconstruction objective, encoder/latent/decoder 역할, bottleneck intuition, denoising/compression variant를 한국어 문장으로 묶는다.

## 이번 단위에서 특히 볼 질문
- reconstruction objective는 "입력을 외운다"는 것과 무엇이 다르고, 언제 유용한가?
- encoder / latent / decoder는 각각 어떤 역할을 맡고, 어디에서 병목이 생기는가?
- bottleneck dimension이 너무 작거나 너무 크면 representation learning 관점에서 어떤 문제가 생기는가?
- denoising autoencoder와 compression-oriented autoencoder는 무엇을 같게 보고 무엇을 다르게 보는가?
- 좋은 latent representation이라고 말하려면 복원 오차 외에 어떤 관찰이 더 필요할까?

## 실행 결과 예시
아래 예시는 이 디렉터리에서 **실제로 실행되는 command/output shape**를 보여 준다.

```text
$ python 02_deep_learning/05_autoencoders_and_representation_learning/scratch_lab.py
{
  "input_dim": 8,
  "sample_count": 8,
  "bottleneck_dims_compared": [1, 2, 3],
  "compression_variant": {
    "selected_latent_dim": 3,
    "compression_ratio": 0.375
  },
  "denoising_variant": {
    "raw_noisy_mse": 0.00434219,
    "denoised_mse": 0.0,
    "denoising_improves_over_noisy_input": true
  },
  "figure_path": "artifacts/scratch-manual/autoencoder_bottleneck.svg"
}

$ python 02_deep_learning/05_autoencoders_and_representation_learning/framework_lab.py
{
  "device": "cpu",
  "input_dim": 8,
  "sample_count": 8,
  "compression_autoencoder": {
    "latent_dim": 3,
    "final_loss": 0.0,
    "reconstruction_shape": [8, 8]
  },
  "narrow_bottleneck_autoencoder": {
    "latent_dim": 1,
    "final_loss": 0.04680542
  },
  "denoising_autoencoder": {
    "final_loss": 0.0,
    "raw_noisy_baseline_loss": 0.00434219,
    "denoising_gain": 0.00434219
  }
}

$ python 02_deep_learning/05_autoencoders_and_representation_learning/analysis.py
# 05 Autoencoders and Representation Learning 실행 관측
- reconstruction objective, bottleneck, denoising/compression variant를
  한국어 관측 리포트로 저장한다.
```

실행 후에는 `autoencoder_bottleneck.svg`에서 **bottleneck 차원별 복원 오차와 denoising bar**를 눈으로 비교하고, JSON metrics에서 **encoder/latent/decoder 역할**, **compression ratio**, **denoising gain** 을 바로 확인할 수 있다.

## 문서를 읽을 때 볼 포인트
- `README.md`: 무엇을 실행하고 어떤 산출물을 남기는지 먼저 본다.
- `THEORY.md`: reconstruction objective, bottleneck, denoising/compression variant를 개념적으로 정리한다.
- `analysis.md`: 숫자가 바뀌어도 유지되는 해석 프레임을 본다.
- `artifacts/analysis-manual/latest_report.md`: 이번 실행에서 실제로 나온 reconstruction mse와 denoising gain을 읽는다.

## 다음 단위와의 연결
이 단위에서 latent representation의 의미를 잡아 두면, 다음 단위 `02_deep_learning/06_generative_models_vae_gan`에서 "latent를 단순히 압축에 쓸 것인가, 샘플링 가능한 생성 공간으로 다룰 것인가" 를 더 자연스럽게 비교할 수 있다. 다시 말해 autoencoder는 representation learning의 가장 작은 출발점이고, VAE/GAN은 그 latent를 더 강하게 구조화하거나 생성적으로 활용하는 다음 단계다.
