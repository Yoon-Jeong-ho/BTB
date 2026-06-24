# 01 VLA Vision-Language-Action Grounding

> Status: runnable · CPU-safe deterministic

## 왜 이 단위를 배우는가

`09_multimodal`에서 retrieval, captioning, VQA를 보면 이미지와 텍스트를 함께 읽는 감각은 생긴다. 하지만 VLA는 “무엇이 보이는가?”에서 끝나지 않고 **무엇을 해야 하는가?** 로 넘어간다. 이 단위는 작은 toy 상태에서 시각 관측과 언어 지시를 합쳐 `action token`과 `safety gate`를 고르는 흐름을 먼저 만든다.

## 이번 단위에서 남길 것

- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- scratch 시각화 `artifacts/scratch-manual/action_policy_matrix.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름

1. `scratch_lab.py`에서 네 개의 장면-지시 쌍을 만들고, hand-coded policy score matrix가 정답 action을 고르는지 확인한다.
2. 같은 scratch 실험에서 action accuracy와 safety gate accuracy를 분리해 본다.
3. `framework_lab.py`에서 tiny PyTorch policy head가 toy feature를 action logits와 safety logit으로 바꾸는지 확인한다.
4. `analysis.py`로 VQA와 VLA의 차이, safety gate, 실제 trajectory 실험으로 확장할 때 필요한 로그를 정리한다.

## 실행 결과 예시

```text
$ python 10_vla/01_vision_language_action_grounding/scratch_lab.py
{
  "unit": "vision_language_action_grounding",
  "scenario_count": 4,
  "policy_matrix_shape": [4, 4],
  "action_accuracy": 1.0,
  "safety_gate_accuracy": 1.0,
  "figure_path": "artifacts/scratch-manual/action_policy_matrix.svg"
}

$ python 10_vla/01_vision_language_action_grounding/framework_lab.py
{
  "unit": "vision_language_action_grounding",
  "device": "cpu",
  "logits_shape": [4, 4],
  "action_accuracy": 1.0,
  "safety_gate_accuracy": 1.0
}
```

## 다음 단계와의 연결

이 단위는 완전한 robot policy가 아니라 VLA로 가는 입구다. 다음 확장에서는 behavior cloning dataset, trajectory error, intervention count, safety violation, sim-to-real gap을 별도 산출물로 남겨야 한다.
