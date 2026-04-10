# 07 학습 레시피와 디버깅 (Training Recipes and Debugging)

> Status: runnable

## 왜 이 단위를 배우는가
모델 구조를 배운 뒤에는, **그 모델이 실제로 수렴하도록 만드는 운영 규칙**을 읽을 수 있어야 한다. 같은 tiny supervised 문제라도 learning rate, batch size, weight decay, scheduler를 어떻게 잡느냐에 따라 loss 곡선은 전혀 다른 모양을 남긴다. 이 단위는 그 차이를 작은 수치와 작은 실험으로 직접 확인하게 만든다.

또한 여기서 배우는 질문들은 이후 LLM fine-tuning, instruction tuning, distributed training에서도 그대로 재사용된다. 즉 이 단위는 “모델을 이해하는 단계”에서 “실험을 믿을 수 있게 만드는 단계”로 넘어가는 연결 고리다.

## 이번 단위에서 남길 것
- scratch sweep 결과 `artifacts/scratch-manual/metrics.json`
- scratch 시각화 `artifacts/scratch-manual/recipe_comparison.svg`
- tiny PyTorch MLP 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 적는 질문 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 polynomial regression을 손으로 학습시켜 learning rate / batch size / weight decay / scheduler 차이를 비교한다.
2. 일부러 learning rate를 과하게 키워 divergence를 만들고, label을 한 칸 밀어 data bug probe도 함께 본다.
3. `framework_lab.py`에서 같은 데이터에 tiny PyTorch MLP를 얹어, recipe 차이가 프레임워크 구현에서도 반복되는지 확인한다.
4. `analysis.py`로 scratch + framework 관측을 한글 리포트로 요약하고, 안정적인 해석 문서는 `analysis.md`에 고정한다.

## 실행 결과 예시
```text
$ python 02_deep_learning/07_training_recipes_and_debugging/scratch_lab.py
{
  "figure_path": "artifacts/scratch-manual/recipe_comparison.svg",
  "sanity_checks": {
    "single_batch_overfit_final_loss": 0.000834,
    "single_batch_overfit_passed": true,
    "tiny_subset_replay_passed": true,
    "high_lr_first_bad_epoch": 5,
    "high_lr_detected": true,
    "label_bug_detected": true
  }
}

$ python 02_deep_learning/07_training_recipes_and_debugging/framework_lab.py
{
  "device": "cpu",
  "model_name": "tiny_mlp_gelu",
  "sanity_checks": {
    "single_batch_overfit_final_loss": 0.000372,
    "single_batch_overfit_passed": true,
    "high_lr_first_bad_epoch": 5,
    "high_lr_detected": true,
    "label_bug_detected": true
  }
}

$ python 02_deep_learning/07_training_recipes_and_debugging/analysis.py
# 07 학습 레시피와 디버깅 실행 관측
- scratch regularized final val loss: 0.003818
- framework regularized final val loss: 0.030674
- high learning rate alert: ['grad_explosion', 'diverged']
```
이 출력은 실제로 재현 가능한 deterministic 예시이며, 실행 후에는 JSON과 SVG가 모두 `artifacts/` 아래에 생성된다.

## 문서를 읽을 때 볼 포인트
- `README.md`: 무엇을 실행하고 어떤 산출물을 남기는지 먼저 본다.
- `THEORY.md`: learning rate / batch size / weight decay / scheduler / debugging 질문을 한 번에 정리한다.
- `analysis.md`: 숫자가 바뀌어도 유지할 해석 프레임을 본다.
- `artifacts/analysis-manual/latest_report.md`: 이번 실행에서 실제로 나온 loss, alert, sanity check를 읽는다.

## 다음 단위와의 연결
여기서 만든 습관은 이후 `05_advanced_nlp_llm`의 fine-tuning 단위와 `06_training_systems`의 distributed runbook 단위로 그대로 이어진다. 결국 중요한 것은 “좋은 모델”보다 먼저 **좋은 로그와 좋은 디버깅 질문**을 갖는 것이다.
