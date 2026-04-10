# 07 Frontier Labs

이 트랙은 `00 → 06`까지의 내용을 바탕으로 들어가는 **research / capstone / agentic sandbox** 구간이다. 앞선 트랙이 계단식 커리큘럼이었다면, 여기서는 그 재료를 조합해 `직접 재현하고, 만들고, 실패를 설계하는` 실험실 역할을 맡는다.

즉 `07`은 정답이 이미 정리된 강의형 단위라기보다, 논문 재현·모델 빌딩·agentic training/eval loop·benchmark 제작 같은 개방형 프로젝트를 안전하게 시작할 수 있게 하는 sandbox다. 이후 `08_multimodal_bridge`, `09_multimodal`로 갈 때도 이 연구 운영 습관이 그대로 이어진다.

## 단위 구성

| Unit | Status | Focus |
| --- | --- | --- |
| [01_paper_reproduction_playground](01_paper_reproduction_playground/README.md) | outlined | 논문 하나를 재현 가능한 실험 묶음으로 바꾸는 기본 playground를 만든다. |
| [02_capstone_model_building](02_capstone_model_building/README.md) | outlined | 여러 이전 단위를 묶어 하나의 end-to-end 모델 프로젝트를 설계한다. |
| [03_agentic_training_and_eval_loops](03_agentic_training_and_eval_loops/README.md) | outlined | agentic workflow가 training/eval/triage loop를 어떻게 바꾸는지 실험한다. |
| [04_benchmark_and_dataset_construction](04_benchmark_and_dataset_construction/README.md) | outlined | 새 벤치마크와 데이터셋 계약을 설계하고 quality gate를 정의한다. |
| [05_open_ended_research_tracks](05_open_ended_research_tracks/README.md) | outlined | 정답이 정해지지 않은 연구 질문을 작은 track으로 쪼개는 방법을 배운다. |

## 이 트랙에 포함되는 것

- paper reproduction, capstone build, agentic loop, benchmark/dataset construction, open-ended research question framing
- 실험 설계, ablation, evaluation protocol, artifact hygiene, self-review 관점의 연구 습관
- 이전 트랙 지식을 조합해 하나의 프로젝트로 묶는 종합 연습

## 이 트랙에서 아직 다루지 않는 것

- 기초 모델 이론 자체의 체계적 강의는 `02_deep_learning`과 `05_advanced_nlp_llm`에서 먼저 다룬다.
- 분산 병렬화 구현 세부는 `06_training_systems`에서 먼저 익힌다.
- 멀티모달 표현의 기본 연결은 `08_multimodal_bridge`에서 다룬다.
