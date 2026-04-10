# 05 Autoencoders and Representation Learning 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 reconstruction objective, encoder/latent/decoder 역할, bottleneck intuition, denoising/compression variant를 읽는 **안정적인 해석 프레임**만 남긴다.

## 해석 프레임
- autoencoder의 reconstruction objective는 label 없이도 입력 자체를 다시 맞히게 하면서, 어떤 정보를 latent로 남겨야 하는지 압박한다.
- encoder는 입력을 latent code로 압축하고, decoder는 그 code만 보고 입력을 얼마나 복원할 수 있는지 시험한다. 따라서 latent는 단순 중간값이 아니라 정보 병목의 위치다.
- bottleneck이 충분히 넓으면 reconstruction은 쉬워지지만 representation이 지나치게 복사에 가까워질 수 있다. 너무 좁으면 핵심 구조까지 잃어 reconstruction이 나빠진다.
- denoising variant는 noisy input으로부터 clean target을 복원하게 만들어, 단순 복사보다 안정적인 구조 보존을 더 강하게 요구한다.
- compression variant는 적은 latent 차원으로 얼마나 reconstruction error를 낮출 수 있는지 보며, 저장 효율과 정보 손실의 trade-off를 드러낸다.

## 확인 질문
- reconstruction objective를 "입력을 외우는 것"과 구분해서 설명하려면 어떤 관측이 필요한가?
- encoder / latent / decoder를 각각 어떤 역할로 읽어야 bottleneck intuition이 선명해지는가?
- latent dimension을 바꿨을 때 reconstruction mse가 어떻게 달라지는지 보고, representation quality를 어떻게 추론할 수 있는가?
- denoising autoencoder와 compression-oriented autoencoder는 같은 구조를 각각 무엇에 더 민감하게 만드는가?

## 관련 이론
- [THEORY.md](./THEORY.md): reconstruction objective, bottleneck, denoising/compression variant를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
