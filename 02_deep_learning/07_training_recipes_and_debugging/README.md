# 07 Training Recipes and Debugging

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 runnable/applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
지금까지는 perceptron, RNN, transformer, generative model처럼 **모델 가족이 무엇을 하는가**를 봤다면, 이제는 그 모델이 **실제로 학습되도록 만드는 운영 규칙**을 봐야 한다. 같은 architecture라도 learning rate, batch size, weight decay, scheduler, logging 습관이 달라지면 결과는 전혀 다르게 나온다. 이 단위는 "모델을 아는 것"에서 멈추지 않고, **실험을 수렴시키고 실패를 해석하는 기본기**를 붙여 주는 마지막 딥러닝 운영 단위다.

또한 이 감각이 있어야 이후 `05_advanced_nlp_llm`에서 instruction tuning / preference optimization을 볼 때 loss spike와 data bug를 덜 막연하게 느끼고, `06_training_systems`에서 distributed run을 볼 때도 "시스템이 느린가"와 "학습 설정이 잘못됐는가"를 분리해서 생각할 수 있다.

## 이번 단위에서 남길 것
- outline 상태의 안내 문서 `README.md`
- learning rate / batch / regularization / debugging 관점을 정리한 `THEORY.md`
- 선행 개념 체크리스트 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 실습 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - training summary metrics
  - train/validation monitoring snapshot
  - NaN/divergence triage checklist
  - sanity-check / ablation log

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 작은 supervised 실험을 하나 고정하고, seed / optimizer / scheduler / logging 항목을 함께 적어 **기본 training recipe 계약**을 만든다.
2. learning rate와 batch size를 바꿔 보며 loss curve, gradient norm, validation metric이 어떻게 달라지는지 비교한다.
3. weight decay와 scheduler를 추가해 overfit / underfit 징후가 train/validation 곡선에서 어떻게 갈리는지 본다.
4. 일부러 실패 케이스를 만든다. 예를 들어 learning rate를 과도하게 올리거나, 잘못된 label/normalization을 넣어 divergence, NaN, data bug가 각각 어떤 로그를 남기는지 관찰한다.
5. single-batch overfit, tiny-subset replay, random-label test, one-change ablation 같은 sanity check를 돌리며 "무엇이 정말 원인인가"를 좁혀 가는 습관을 만든다.
6. 마지막에는 이 루틴이 왜 later LLM fine-tuning과 distributed training에서도 그대로 재사용되는지 연결한다.

## 이 단위에서 특히 볼 질문
- learning rate는 단순히 "빠르게/느리게 학습"만 바꾸는가, 아니면 안정성 자체를 바꾸는가?
- batch size를 키웠을 때 좋아지는 것은 throughput뿐인가, 아니면 optimization noise와 generalization도 함께 바뀌는가?
- weight decay와 scheduler는 둘 다 loss를 낮추는 도구처럼 보이지만, 실제로는 어떤 다른 역할을 하는가?
- train loss와 validation loss/metric을 함께 볼 때 overfit, underfit, optimization failure를 어떻게 구분할 수 있는가?
- NaN이나 divergence가 났을 때 숫자 불안정, optimizer 설정 문제, 데이터 파이프라인 버그를 어떤 순서로 분리해 볼 것인가?
- later LLM/system work에서 왜 warmup, effective batch, logging discipline, ablation habit이 더 중요해지는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 02_deep_learning/07_training_recipes_and_debugging/framework_lab.py
{
  "status": "sample",
  "config": {
    "optimizer": "AdamW",
    "learning_rate": 0.0005,
    "batch_size": 64,
    "weight_decay": 0.01,
    "scheduler": "cosine_with_warmup"
  },
  "best_epoch": 7,
  "train_loss": [2.31, 1.84, 1.42],
  "val_loss": [2.28, 1.92, 2.11],
  "val_metric": [0.41, 0.56, 0.54],
  "grad_norm_last": 1.73,
  "alerts": ["mild_overfit", "no_nan_detected"]
}

$ python 02_deep_learning/07_training_recipes_and_debugging/analysis.py
{
  "status": "sample",
  "sanity_checks": {
    "single_batch_overfit": true,
    "tiny_subset_replay": true,
    "random_label_test": "failed_as_expected"
  },
  "debug_trace": {
    "first_bad_step": null,
    "label_range_ok": true,
    "input_nan_count": 0,
    "mixed_precision_disabled_for_repro": false
  },
  "ablation_table_shape": [4, 6],
  "notes": "expected output/sample shape only"
}
```

핵심은 숫자 하나를 외우는 것이 아니라, **어떤 설정 변화가 어떤 곡선 흔적을 남기는지**, **실패를 재현하고 좁혀 가는 로그 구조가 어떻게 생겨야 하는지**를 읽는 것이다.

## 다음 단위와의 연결
이 단위는 `02_deep_learning` 트랙의 마지막 운영 정리 단위다. 그래서 다음 연결은 같은 트랙 안의 "다음 architecture"가 아니라, **이 운영 감각을 더 큰 모델과 시스템으로 옮기는 단계**다.

- `05_advanced_nlp_llm/04_instruction_tuning_and_sft` 이후 단위들에서는 warmup, effective batch, overfit monitoring, data-quality debugging이 직접적인 품질 차이로 이어진다.
- `06_training_systems/01_torchrun_and_ddp_basics`, `07_data_parallel_grad_accumulation`, `09_profiling_monitoring_and_failure_recovery`에서는 여기서 만든 training/debugging 습관이 분산 환경용 runbook으로 확장된다.

즉 이 단위는 "좋은 모델을 고르는 법"과 "큰 시스템을 돌리는 법" 사이에서, **실험을 믿을 수 있게 만드는 최소 운영 규칙**을 마련하는 연결 고리다.
