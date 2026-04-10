# 07 학습 레시피와 디버깅 리플렉션

## 실행 후 바로 적어 볼 질문
1. `small_batch_baseline`과 `large_batch_constant_lr`를 비교했을 때, **같은 epoch budget** 안에서 무엇이 먼저 달라졌는가? train loss, validation loss, gap 중 무엇이 가장 읽기 쉬웠는가?
2. scratch와 framework에서 모두 `weight_decay + scheduler`가 baseline보다 약간 더 나은 validation loss를 만들었다면, 이것을 “regularization 효과”와 “late-stage step shrinkage”로 어떻게 나눠 설명할 수 있는가?
3. high learning rate probe는 어떤 epoch에서 처음 이상 신호를 냈는가? 그때의 alert를 보고, 다음 디버깅 순서를 어떻게 정하겠는가?
4. shifted-label bug probe를 보고 “모델이 약해서 못 배운 것”과 “데이터가 어긋난 것”을 어떤 근거로 구분할 수 있었는가?
5. single-batch overfit sanity check가 통과했다면 무엇이 안심되고, 실패했다면 무엇부터 의심해야 하는가?

## 다음 실험을 위한 체크리스트
- [ ] 다음 번에는 learning rate만 바꾸고 나머지는 고정한 ablation을 따로 기록했다.
- [ ] batch size를 바꿀 때 effective step count와 gradient noise 해석을 함께 적었다.
- [ ] validation metric이 흔들릴 때 데이터 버그 probe를 먼저 돌려 봤다.
- [ ] single-batch overfit / tiny-subset replay 같은 sanity check를 본 실험 전에 수행했다.
- [ ] 이번 단위의 디버깅 질문을 이후 LLM fine-tuning 또는 distributed training 로그에도 그대로 적용해 볼 수 있다.
