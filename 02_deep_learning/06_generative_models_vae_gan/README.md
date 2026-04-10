# 06 Generative Models: VAE, GAN

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 runnable/applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
이전 단위에서 autoencoder가 "입력을 어떻게 요약할 것인가"를 봤다면, 여기서는 한 걸음 더 나아가 **그 요약 공간에서 새로운 샘플을 어떻게 만들어 낼 것인가**를 본다. VAE는 확률적 latent space를 만들어 샘플링 가능한 구조를 강조하고, GAN은 generator와 discriminator의 경쟁을 통해 더 날카로운 샘플 품질을 노린다. 이 대비를 먼저 잡아 두면 later diffusion model, multimodal generation, synthetic data, image editing 같은 더 큰 생성 모델 계열을 덜 막연하게 볼 수 있다.

또한 이 단위는 "샘플 품질이 좋아 보인다"와 "latent가 구조적으로 잘 정리됐다"가 같은 말이 아니라는 점, 그리고 생성 모델은 분류 모델보다 **학습 안정성과 관찰 포인트가 더 중요하다**는 점을 처음으로 드러내는 지점이기도 하다.

## 이번 단위에서 남길 것
- outline 상태의 안내 문서 `README.md`
- VAE와 GAN의 관점을 대비한 `THEORY.md`
- 선행 개념과 빠른 자기 점검을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 실습 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - latent sampling / interpolation 요약
  - generator-vs-discriminator training snapshot
  - reconstruction-vs-sample-quality 비교 노트
  - mode collapse / posterior collapse 관찰 포인트

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 작은 autoencoder 복습에서 출발해, deterministic latent code만으로는 "아무 점이나 찍어 샘플링"하기 어렵다는 문제를 먼저 확인한다.
2. VAE 관점으로 넘어가 encoder가 `mu`, `logvar`를 내고 latent를 샘플링하는 구조를 따라가며, reconstruction term과 KL regularization이 각각 무엇을 밀어 주는지 읽는다.
3. latent interpolation을 가정해 보며, VAE가 왜 "잘 정리된 latent geometry"를 얻으려 하는지 관찰 질문을 남긴다.
4. GAN 관점으로 넘어가 generator와 discriminator의 min-max 상호작용을 읽고, adversarial signal이 왜 샘플을 더 선명하게 만들 수 있는지 직관으로 정리한다.
5. 대신 GAN은 왜 mode collapse, unstable oscillation, discriminator overpower 같은 문제를 자주 낳는지 failure pattern 중심으로 살핀다.
6. 마지막에는 VAE와 GAN을 "확률적 latent 구조화" vs "adversarial sample realism"이라는 두 축으로 비교하고, 다음 단위의 training/debugging 관점으로 연결한다.

## 이 단위에서 특히 볼 질문
- VAE의 latent sampling은 단순 노이즈 주입과 무엇이 다르고, 왜 KL term이 함께 필요할까?
- reconstruction quality가 좋다고 해서 generative quality도 좋다고 말할 수 있을까?
- GAN에서 discriminator loss가 낮다는 사실은 generator에게 항상 좋은 신호일까?
- VAE의 blurry sample과 GAN의 sharp sample 비교는 무엇을 보여 주고, 무엇을 숨길까?
- posterior collapse와 mode collapse는 이름이 비슷해 보여도 각각 어떤 실패를 뜻하는가?
- 생성 모델에서는 accuracy 대신 어떤 관찰값과 샘플 inspection이 더 중요해지는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 02_deep_learning/06_generative_models_vae_gan/scratch_lab.py
{
  "status": "sample",
  "vae": {
    "input_shape": [8, 16],
    "mu_shape": [8, 4],
    "logvar_shape": [8, 4],
    "latent_sample_shape": [8, 4],
    "reconstruction_shape": [8, 16],
    "kl_term": 0.27,
    "recon_term": 0.11
  },
  "gan": {
    "noise_shape": [8, 4],
    "generated_shape": [8, 16],
    "discriminator_logits_shape": [8, 1],
    "generator_loss": 0.91,
    "discriminator_loss": 0.63
  }
}

$ python 02_deep_learning/06_generative_models_vae_gan/framework_lab.py
{
  "status": "sample",
  "model_family": ["tiny_vae", "tiny_gan"],
  "latent_dim": 8,
  "sample_grid_shape": [16, 1, 28, 28],
  "interpolation_steps": 7,
  "alerts": ["sample_only", "watch_mode_collapse", "watch_posterior_collapse"]
}
```

핵심은 숫자 하나를 외우는 것이 아니라, **latent에서 어떤 tensor가 흘러가는지**, **VAE는 어떤 regularization을 더 보고 GAN은 어떤 경쟁 신호를 더 보는지**, **실패 징후가 로그와 샘플에서 어떻게 보이는지**를 읽는 것이다.

## 다음 단위와의 연결
다음 단위 `02_deep_learning/07_training_recipes_and_debugging`에서는 바로 여기서 드러난 문제를 더 일반적인 운영 습관으로 확장한다. VAE에서는 KL weight와 reconstruction balance, posterior collapse, latent usage를 읽어야 하고, GAN에서는 generator/discriminator 균형, divergence, mode collapse, unstable loss oscillation을 읽어야 한다. 즉 이 단위가 생성 모델의 **문제의식**을 만들고, 다음 단위가 그 문제를 다루는 **훈련/디버깅 루틴**을 제공한다.
