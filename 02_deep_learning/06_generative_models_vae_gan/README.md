# 06 Generative Models: VAE, GAN

> Status: runnable
>
> 이 단위는 **CPU-safe / deterministic / toy generative examples only** 조건에서 VAE와 GAN을 직접 실행해 보는 runnable 단계다. VAE의 latent sampling, GAN의 adversarial intuition, posterior collapse / mode collapse 관찰을 작은 숫자와 SVG 한 장으로 바로 연결한다.

## 왜 이 단위를 배우는가
`02_deep_learning/05_autoencoders_and_representation_learning`에서 latent가 “압축용 코드”로 쓰이는 모습을 봤다면, 이제는 그 latent를 **샘플링 가능한 생성 공간**으로 다루는 관점과, reconstruction 없이 **진짜 같은 샘플을 밀어 올리는 adversarial 관점**을 대비해서 읽어야 한다. VAE와 GAN은 이후 diffusion model, multimodal generation, synthetic data augmentation을 이해할 때 계속 다시 등장하는 두 출발점이다.

## 이번 단위에서 남길 것
- scratch 관측 결과 `artifacts/scratch-manual/metrics.json`
- scratch figure `artifacts/scratch-manual/vae_gan_diagnostics.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자 회고 질문 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 아주 작은 2차원 toy distribution을 만들고, VAE 쪽에서는 `mu/logvar -> z -> reconstruction` 흐름을 손계산에 가까운 방식으로 따라간다.
2. 같은 scratch 실험에서 latent interpolation과 prior decoding을 보며, VAE가 왜 “샘플링 가능한 latent geometry”를 만들려 하는지 확인한다.
3. posterior collapse probe를 따로 만들어, latent usage가 줄어들면 reconstruction이 어떻게 나빠지는지 비교한다.
4. 같은 데이터에 대해 GAN 쪽에서는 balanced generator와 collapsed generator를 대비해, adversarial loss만으로는 mode coverage를 다 읽기 어렵다는 점을 본다.
5. `framework_lab.py`에서는 tiny PyTorch modules로 같은 질문을 다시 실행해, CPU에서 deterministic하게 VAE/GAN 핵심 관찰만 재현한다.
6. `analysis.py`는 매 실행 관측을 `latest_report.md`에 쓰고, `analysis.md`는 안정적인 해석 프레임만 유지한다.

## 이번 단위에서 특히 볼 질문
- VAE는 reconstruction term과 KL term을 같이 보면서 latent를 어떻게 샘플링 가능한 공간으로 만들려 하는가?
- reparameterization trick은 단순 noise injection과 무엇이 다르고, 왜 gradient 관점에서 중요한가?
- posterior collapse와 mode collapse는 각각 무엇이 collapse되는 실패이며, 어떤 지표가 먼저 경고를 주는가?
- GAN loss가 그럴듯해 보여도 diversity / coverage를 따로 확인해야 하는 이유는 무엇인가?
- VAE의 smooth interpolation과 GAN의 sharp mode fitting은 각각 어떤 장점과 함정을 가지는가?

## 실행 결과 예시
아래 예시는 이 디렉터리에서 **실제로 실행되는 command/output shape**를 보여 준다.

```text
$ python 02_deep_learning/06_generative_models_vae_gan/scratch_lab.py
{
  "dataset_point_count": 8,
  "vae": {
    "input_dim": 2,
    "latent_dim": 2,
    "reconstruction_mse": 0.176561,
    "kl_term": 1.637268,
    "prior_sample_spread": 1.25591
  },
  "gan": {
    "noise_dim": 2,
    "balanced_mode_coverage": 4,
    "collapsed_mode_coverage": 1,
    "collapse_detected": true
  },
  "figure_path": "artifacts/scratch-manual/vae_gan_diagnostics.svg"
}

$ python 02_deep_learning/06_generative_models_vae_gan/framework_lab.py
{
  "device": "cpu",
  "vae": {
    "latent_dim": 2,
    "final_reconstruction_loss": 0.205331,
    "final_kl_loss": 1.427023,
    "posterior_usage_mean_abs": 0.644694
  },
  "gan": {
    "noise_dim": 2,
    "mode_coverage": 4,
    "loss_only_is_ambiguous": true
  }
}

$ python 02_deep_learning/06_generative_models_vae_gan/analysis.py
# 06 Generative Models: VAE, GAN 실행 관측
- scratch/framework metrics를 읽어 VAE latent sampling, GAN mode coverage,
  posterior collapse / mode collapse 해석을 한국어 리포트로 저장한다.
```

실행 후에는 `vae_gan_diagnostics.svg`에서 **VAE interpolation / prior samples / GAN balanced-vs-collapsed coverage**를 눈으로 비교하고, JSON metrics에서 **KL, latent usage, mode coverage, pairwise diversity** 를 바로 확인할 수 있다.

## 문서를 읽을 때 볼 포인트
- `README.md`: 무엇을 실행하고 어떤 산출물을 남기는지 먼저 본다.
- `THEORY.md`: VAE vs GAN, reparameterization trick, posterior collapse, mode collapse를 개념적으로 정리한다.
- `analysis.md`: 숫자가 바뀌어도 유지되는 generative-model 해석 프레임을 본다.
- `artifacts/analysis-manual/latest_report.md`: 이번 실행에서 실제로 나온 KL / mode coverage / collapse 관측을 읽는다.

## 다음 단위와의 연결
다음 단위 `02_deep_learning/07_training_recipes_and_debugging`에서는 바로 여기서 드러난 문제를 더 일반적인 운영 습관으로 확장한다. VAE에서는 KL weight와 latent usage, GAN에서는 generator/discriminator balance와 mode coverage를 계속 읽어야 한다. 즉 이 단위는 생성 모델의 **문제의식**을 만들고, 다음 단위는 그 문제를 다루는 **훈련/디버깅 루틴**을 제공한다.
