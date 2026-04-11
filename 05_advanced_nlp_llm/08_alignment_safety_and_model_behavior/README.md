# 08 Alignment, Safety, and Model Behavior

> Status: runnable
>
> 이 단위는 CPU에서 바로 실행되는 deterministic toy alignment / safety behavior eval 실습이다. 실제 LLM을 호출하거나 학습하지 않고, capability가 높아도 배포 행동이 안전하지 않을 수 있음을 **alignment vs capability**, **refusal vs over-refusal**, **harmlessness / robustness**, **behavioral eval slice analysis**, **policy vs system-level safety** 관점에서 관찰한다.

## 왜 이 단위를 배우는가
모델이 답을 "할 수 있는가"와 제품 환경에서 "어떻게 답해야 하는가"는 같은 문제가 아니다. 높은 capability는 benign task를 잘 풀게 만들지만, 동시에 위험한 요청을 더 정교하게 수행할 잠재력도 키운다. alignment는 그 능력이 실제 사용자 상호작용에서 어떤 정책 경계와 시스템 guardrail 안에서 나타나는지를 다룬다.

이 단위는 안전성을 막연한 윤리 문구가 아니라 다음의 측정 가능한 행동 계약으로 본다.

- benign request에는 충분히 helpful하게 답한다.
- harmful request에는 refuse and redirect한다.
- borderline request에는 안전한 범위 축소나 safe alternative를 제시한다.
- paraphrase, formatting noise, jailbreak-style framing에도 behavior가 안정적이어야 한다.
- model policy가 맡을 일과 tool permission gating, moderation, audit logging 같은 system-level safety가 맡을 일을 분리한다.

## 이번 단위에서 남길 것
- `scratch_lab.py`: 손으로 만든 behavior slices에서 refusal, over-refusal, harmlessness, robustness 지표와 `artifacts/scratch-manual/alignment_behavior_slices.svg`를 생성한다.
- `framework_lab.py`: lightweight deterministic behavior-eval simulation으로 capability-only assistant와 aligned assistant를 비교하고 slice analysis metrics를 만든다.
- `analysis.py`: metrics가 없으면 명확히 실패하고, 있으면 stable `analysis.md`는 보존한 채 관측 리포트를 `artifacts/analysis-manual/latest_report.md`에 쓴다.
- `THEORY.md`, `PREREQS.md`, `reflection.md`, `lesson.yaml`: 한국어 우선 개념·선행지식·회고 질문·메타데이터를 고정한다.

## 실행 방법
프로젝트 루트(`/data_x/aa007878/projects/BTB`)에서 아래 순서로 실행한다.

```bash
python 05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/scratch_lab.py
python 05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/framework_lab.py
python 05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/analysis.py
```

생성되는 파일은 다음과 같다.

```text
05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/artifacts/
├── scratch-manual/
│   ├── metrics.json
│   └── alignment_behavior_slices.svg
├── framework-manual/
│   └── metrics.json
└── analysis-manual/
    └── latest_report.md
```

## 실행 결과 예시
아래 예시는 이 저장소의 deterministic toy script가 실제로 쓰는 핵심 필드 모양이다.

```text
$ python 05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/scratch_lab.py
{
  "setup": {
    "unit": "08_alignment_safety_and_model_behavior",
    "mode": "toy_behavior_policy_eval",
    "cpu_safe": true
  },
  "alignment_vs_capability": {
    "capability_score": 0.896667,
    "behavior_contract_score": 0.966,
    "capability_can_enable_risk": true
  },
  "behavior_slices": {
    "prompt_count": 6,
    "slice_names": ["benign", "borderline", "harmful"],
    "benign_answer_rate": 1.0,
    "harmful_refusal_rate": 1.0,
    "over_refusal_rate": 0.0,
    "safe_alternative_rate": 1.0
  },
  "figure_path": "artifacts/scratch-manual/alignment_behavior_slices.svg"
}

$ python 05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/framework_lab.py
{
  "device": "cpu",
  "simulation": "deterministic_behavior_eval_simulation",
  "dataset_size": 8,
  "aggregate_scores": {
    "capability_only_assistant": {
      "capability_score": 0.92,
      "behavior_contract_score": 0.333333
    },
    "aligned_assistant": {
      "capability_score": 0.84,
      "behavior_contract_score": 1.0
    }
  }
}

$ python 05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/analysis.py
# 08 Alignment, Safety, and Model Behavior 실행 관측
...
## 한국어 해석
- alignment vs capability는 같은 축이 아니다. capability score가 높아도 unsafe compliance가 남으면 배포 행동은 실패한다.
```

## 실습 흐름
1. benign / harmful / borderline 요청을 작게 나누고 각 요청의 expected behavior를 정한다.
2. observed behavior를 compliant, safe_refusal, safe_alternative, over_refusal, unsafe_compliance로 라벨링한다.
3. refusal rate만 보지 않고 benign answer rate, harmful refusal rate, over-refusal rate, safe alternative rate를 함께 읽는다.
4. paraphrase, noisy prompt, jailbreak-style variant에서 같은 정책 행동이 유지되는지 robustness probe를 확인한다.
5. capability-only assistant와 aligned assistant를 비교해 capability 개선이 safety 개선과 자동으로 같지 않음을 본다.
6. 마지막으로 model policy와 system guardrail의 책임을 분리한다. 예를 들어 unsafe content refusal은 model policy가 맡지만, 실제 tool permission gating과 audit logging은 system-level safety가 맡아야 한다.

## 이 단위에서 특히 볼 질문
- alignment vs capability를 분리하지 않으면 어떤 위험한 결론을 내리게 되는가?
- refusal은 언제 바람직한 harmlessness 행동이고, 언제 over-refusal로 usefulness를 해치는가?
- benign / harmful / borderline / robustness slice analysis를 왜 한 scalar 점수보다 먼저 봐야 하는가?
- robustness는 jailbreak 방어뿐 아니라 paraphrase와 formatting noise 안정성까지 왜 포함하는가?
- behavioral eval에서 높은 judge score가 실제 안전한 제품 행동을 충분히 보장하지 못하는 이유는 무엇인가?
- policy vs system-level safety를 분리할 때 model policy, tool permission gating, moderation, audit logging은 각각 어디에 놓이는가?

## 다음 단위와의 연결
이 트랙의 마지막 단위로서, 여기서 behavior contract와 safety boundary를 정리해 두면 `07_frontier_labs/04_benchmark_and_dataset_construction`에서 behavioral benchmark와 evaluation rubric을 설계할 때 훨씬 구체적인 질문을 던질 수 있다. alignment를 막연한 가치 논쟁이 아니라 측정 가능한 행동 기준과 시스템 책임 분리로 옮기는 징검다리다.
