# 05 Autoencoders and Representation Learning 회고

## 1. reconstruction objective를 내 말로 다시 쓰기
- reconstruction objective가 왜 label 없이도 학습 신호가 되는지 한 문장으로 적어 보자.
- "입력을 다시 맞힌다"는 말과 "그냥 복사한다"는 말을 왜 구분해야 하는지 적어 보자.

## 2. encoder / latent / decoder 역할 구분
- encoder는 입력에서 무엇을 남기고 무엇을 버린다고 느꼈는가?
- latent를 병목이라고 부르는 이유를, 이번 실습의 숫자와 함께 적어 보자.
- decoder는 어떤 점에서 "정답을 맞히는 분류기"가 아니라 "복원 시험기"처럼 보였는가?

## 3. bottleneck intuition 정리
- latent dimension이 1, 2, 3으로 바뀔 때 reconstruction mse가 어떻게 달라졌는가?
- 너무 좁은 bottleneck과 충분한 bottleneck을 각각 어떤 장단점으로 설명할 수 있는가?
- compression ratio가 좋아질수록 항상 representation도 좋아진다고 말할 수 없는 이유는 무엇인가?

## 4. denoising / compression variant 연결
- noisy input을 clean target으로 복원하게 했을 때 어떤 정보가 더 중요해졌는가?
- denoising autoencoder와 compression-oriented autoencoder가 같은 구조를 어떻게 다르게 활용하는지 비교해 보자.
- 복원 오차 말고도, 좋은 latent representation이라고 부르려면 어떤 downstream 관찰이 더 필요할까?
