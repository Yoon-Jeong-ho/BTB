# 06 Generative Models: VAE, GAN 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 VAE vs GAN contrast, latent sampling, adversarial intuition, posterior collapse, mode collapse를 읽는 **안정적인 해석 프레임**만 남긴다.

## 해석 프레임
- VAE는 reconstruction term과 KL term을 함께 보며, latent를 단순 압축 코드가 아니라 **샘플링 가능한 분포 좌표계**로 만들려 한다.
- reparameterization trick은 `z = mu + sigma * epsilon` 형태로 noise를 주입하면서도 gradient가 encoder로 흐르도록 돕는다.
- posterior collapse는 decoder가 너무 많은 일을 대신해 `z`를 거의 쓰지 않는 실패다. 이때 latent usage와 KL이 함께 줄어들 수 있다.
- GAN은 generator와 discriminator의 adversarial game으로 sharp sample realism을 밀어 올릴 수 있지만, loss만으로 diversity를 다 읽기 어렵다.
- mode collapse는 generator가 한두 mode만 반복 출력하는 실패다. 그래서 generative model에서는 sample quality와 함께 coverage / pairwise diversity / batch inspection을 같이 봐야 한다.

## 확인 질문
- VAE에서 reconstruction이 좋아도 KL과 latent usage를 같이 봐야 하는 이유는 무엇인가?
- reparameterization trick이 없으면 encoder가 latent sampling을 학습하기 어려운 이유를 어떻게 설명할 수 있는가?
- posterior collapse와 mode collapse는 각각 무엇이 collapse되는 현상인가?
- GAN loss가 그럴듯해 보여도 mode coverage를 따로 확인해야 하는 이유는 무엇인가?

## 관련 이론
- [THEORY.md](./THEORY.md): VAE vs GAN, reparameterization trick, posterior collapse, mode collapse를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
