# 06 RLHF and Reasoning RL

> Status: runnable
>
> 이 단위는 CPU에서 바로 실행되는 deterministic toy RLHF / reasoning RL 실습이다. 실제 LLM을 학습하지 않고, reward model이 preference proxy를 만들고 PPO-family policy update가 어떤 high-level 신호를 쓰며 verifier / judge / reward shaping이 어떤 실패 모드를 만들 수 있는지 관찰한다.

## 왜 이 단위를 배우는가
`05_preference_optimization_dpo_orpo_kto`는 offline preference objective로 policy log-prob margin을 직접 움직이는 감각을 준다. 하지만 실제 post-training에서는 현재 policy가 생성한 rollout을 다시 보고, reward model과 verifier / judge signal을 섞어 policy update를 반복해야 할 때가 있다. 이 단위는 RLHF를 거대한 분산 학습 레시피로 외우지 않고 다음 질문으로 쪼갠다.

- reward model은 truth engine이 아니라 어떤 annotation rubric과 judge 선호를 압축한 proxy인가?
- PPO-family update는 왜 reward만 키우지 않고 KL anchor와 regression eval을 같이 보는가?
- reasoning RL은 긴 chain을 보상하는 것이 아니라 검증 가능성, self-correction, final answer quality를 어떻게 reward shaping하는가?
- verifier와 judge signal은 어디서 도움이 되고, reward hacking / verbosity / over-refusal을 어디서 키울 수 있는가?

## 이번 단위에서 남길 것
- `scratch_lab.py`: 손으로 만든 chosen/rejected rollout batch에서 reward model score, RLHF loop view, reasoning reward shaping, `artifacts/scratch-manual/rlhf_reasoning_reward.svg`를 생성한다.
- `framework_lab.py`: tiny numeric reasoning RL simulation으로 PPO-family update, KL guardrail, verifier consistency, judge length bias를 생성한다.
- `analysis.py`: metrics가 없으면 명확히 실패하고, 있으면 stable `analysis.md`를 유지한 채 관측 리포트를 `artifacts/analysis-manual/latest_report.md`에 쓴다.
- `THEORY.md`, `PREREQS.md`, `reflection.md`, `lesson.yaml`: 한국어 우선 개념·선행지식·회고 질문·메타데이터를 고정한다.

## 실행 방법
프로젝트 루트(`/data_x/aa007878/projects/BTB`)에서 아래 순서로 실행한다.

```bash
python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/scratch_lab.py
python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/framework_lab.py
python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/analysis.py
```

생성되는 파일은 다음과 같다.

```text
05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/artifacts/
├── scratch-manual/
│   ├── metrics.json
│   └── rlhf_reasoning_reward.svg
├── framework-manual/
│   └── metrics.json
└── analysis-manual/
    └── latest_report.md
```

## 실행 결과 예시
아래 예시는 이 저장소의 deterministic toy script가 실제로 쓰는 핵심 필드 모양이다.

```text
$ python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/scratch_lab.py
{
  "reward_model_batch": {
    "prompt_count": 4,
    "candidate_count": 8,
    "chosen_rejected_pairs": 4,
    "avg_reward_chosen": 0.7932,
    "avg_reward_rejected": 0.37895,
    "reward_model_intuition": "preference proxy, not truth engine"
  },
  "rlhf_loop_view": {
    "steps": ["sample_prompts", "policy_rollouts", "score_rewards", "ppo_family_update", "regression_eval"],
    "policy_update_style": "PPO-family clipped advantage sketch, not full training",
    "kl_anchor_enabled": true
  },
  "reasoning_signal": {
    "process_reward_weight": 0.35,
    "verifier_pass_rate": 1.0,
    "judge_preference_win_rate": 1.0,
    "longer_trace_is_always_better": false
  },
  "figure_path": "artifacts/scratch-manual/rlhf_reasoning_reward.svg"
}

$ python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/framework_lab.py
{
  "device": "cpu",
  "simulation": "tiny_numeric_reasoning_rl",
  "policy_update": {
    "update_family": "PPO-family clipped advantage sketch",
    "reward_mean_before": 0.4675,
    "reward_mean_after": 0.6475,
    "advantage_mean_before": 0.015,
    "advantage_mean_after": 0.205,
    "kl_after": 0.085,
    "kl_guardrail": 0.12
  },
  "reasoning_eval": {
    "answer_accuracy_before": 0.5,
    "answer_accuracy_after": 0.68,
    "verifier_consistency_before": 0.54,
    "verifier_consistency_after": 0.74,
    "judge_length_bias_flag": true
  }
}

$ python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/analysis.py
# 06 RLHF and Reasoning RL 실행 관측
...
## 한국어 해석
- scratch 실험의 reward model은 preference proxy, not truth engine이다.
```

## 실습 흐름
1. 같은 prompt에 chosen/rejected rollout을 두고 reward model score를 계산한다.
2. reward가 진실 점수가 아니라 verifier, judge, format, safety rubric이 섞인 preference proxy임을 확인한다.
3. RLHF loop를 prompt sampling → policy rollout → reward scoring → PPO-family policy update → regression eval 순서로 읽는다.
4. reasoning RL에서는 final answer reward와 process reward를 분리하고, verifier consistency와 judge win rate가 서로 다른 신호임을 본다.
5. reward가 오르더라도 reward hacking, verbosity inflation, over-refusal, style bias가 생길 수 있으므로 failure-mode probes를 함께 읽는다.

## 이 단위에서 특히 볼 질문
- reward model은 어떤 선호를 압축하며, 왜 truth engine으로 취급하면 안 되는가?
- PPO-family policy update에서 reward mean이 오른 뒤에도 KL guardrail을 보는 이유는 무엇인가?
- verifier는 좁은 정합성 signal을, judge는 넓은 선호 signal을 주지만 각각 어떤 gaming 위험이 있는가?
- reasoning RL의 reward shaping은 긴 trace가 아니라 어떤 process behavior를 밀어야 하는가?
- reward hacking, verbosity, over-refusal이 발견되면 어떤 held-out prompts와 regression eval을 추가해야 하는가?

## 다음 단위와의 연결
이 단위가 내부 policy shaping을 다룬다면, 다음 `05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval`은 좋아 보이는 답과 실제 근거가 있는 답을 분리한다. RLHF와 reasoning RL로 model behavior를 밀어도 retrieval grounding과 evaluation은 별도 축으로 남는다.
