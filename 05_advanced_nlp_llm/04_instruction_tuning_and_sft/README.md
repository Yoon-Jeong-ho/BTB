# 04 Instruction Tuning and SFT

> Status: runnable
>
> 이 단위는 **CPU-safe, deterministic, toy instruction-tuning**만 다루는 runnable 단계다. 큰 LLM을 학습하지 않고도 instruction format, supervised fine-tuning, input-output template, system/user/assistant role framing, imitation vs helpfulness tradeoff를 직접 관찰한다.

## 왜 이 단위를 배우는가
base LM은 다음 토큰 continuation에는 강하지만, 곧바로 사용자의 지시를 읽고 assistant처럼 답하는 것은 아니다. Instruction tuning과 SFT(supervised fine-tuning)는 이미 가진 언어 능력을 **instruction-response 인터페이스** 위로 다시 정렬한다. 이 단위는 SFT를 “거대한 모델 학습”이 아니라, 작은 toy template와 loss mask를 통해 **어떤 토큰을 입력으로 보고 어떤 토큰을 정답으로 삼는가**를 읽는 훈련으로 만든다.

특히 같은 질문도 plain instruction format과 chat template로 직렬화하면 boundary, role tag, system constraint가 달라진다. SFT는 이런 포맷의 정답 응답을 잘 모방하게 만들지만, 모방이 곧 더 도움이 되는 답변을 고른다는 뜻은 아니다. 그래서 다음 단위의 preference optimization이 왜 필요한지도 함께 준비한다.

## 이번 단위에서 남길 것
- scratch template/loss-mask 관측치 `artifacts/scratch-manual/metrics.json`
- scratch SVG `artifacts/scratch-manual/sft_template_loss.svg`
- deterministic numeric SFT simulation 관측치 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 한국어 우선 학습자 회고 질문 `reflection.md`

## 실행 방법
```bash
python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/scratch_lab.py
python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/framework_lab.py
python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/analysis.py
```

생성물은 다음 위치에 남는다.
- `artifacts/scratch-manual/metrics.json`
- `artifacts/scratch-manual/sft_template_loss.svg`
- `artifacts/framework-manual/metrics.json`
- `artifacts/analysis-manual/latest_report.md`

## 실습 흐름
1. `scratch_lab.py`에서 같은 instruction example을 plain instruction-response format과 chat template로 직렬화한다.
2. 각 template의 prompt tokens, assistant response tokens, full sequence tokens를 세고, assistant-only loss mask가 무엇을 무시하는지 본다.
3. system message가 있을 때와 없을 때의 role framing score를 비교해, system/user/assistant 표식이 conditioning signal로 쓰인다는 점을 관찰한다.
4. `sft_template_loss.svg`에서 prompt tokens는 masked out되고 Assistant loss tokens가 supervised target으로 남는 그림을 확인한다.
5. `framework_lab.py`에서 4개 toy conversation batch를 deterministic numeric SFT simulation으로 만들고, labels와 loss_mask shape를 확인한다.
6. `analysis.py`로 stable `analysis.md`와 실행별 observed report를 분리해, 해석 프레임과 최신 관측값을 따로 보관한다.

## 실행 결과 예시
아래는 이 디렉터리에서 **실제로 실행되는 command/output shape**다.

```text
$ python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/scratch_lab.py
{
  "setup": {"unit": "04_instruction_tuning_and_sft"},
  "loss_masking": {
    "target_region": "assistant_response_only",
    "prompt_tokens_masked_out": 36,
    "assistant_loss_tokens": 36
  },
  "role_framing": {
    "recommended_for_role_control": "chat_template",
    "system_constraint_delta": 0.34
  },
  "figure_path": "artifacts/scratch-manual/sft_template_loss.svg"
}

$ python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/framework_lab.py
{
  "device": "cpu",
  "framework": "deterministic_numeric_sft",
  "dataset_size": 4,
  "batch_shape": {
    "input_ids": [4, 30],
    "labels": [4, 30],
    "loss_mask": [4, 30]
  },
  "next_step": {"why_sft_is_not_enough": "preference_optimization_needed"}
}

$ python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/analysis.py
# 04 Instruction Tuning and SFT 실행 관측
- instruction format, supervised fine-tuning, system/user/assistant role framing, imitation vs helpfulness tradeoff를 한국어 리포트로 저장한다.
```

## 관찰 포인트
1. **instruction format**: 같은 의미의 예시라도 plain template와 chat template는 모델이 보는 role boundary를 다르게 만든다.
2. **supervised fine-tuning**: SFT는 보통 serialized conversation 위에서 next-token loss를 쓰되, 관찰 대상은 assistant response 구간이다.
3. **input-output template**: template는 예쁜 포맷이 아니라 generation 시작점, stop marker, role boundary를 정의하는 학습 신호다.
4. **system/user/assistant framing**: system은 전체 제약, user는 현재 요청, assistant는 supervised target으로 읽는다.
5. **imitation vs helpfulness**: SFT는 좋은 답변처럼 보이는 reference를 모방하게 하지만, 여러 답변 중 무엇이 더 도움이 되는지 직접 비교하지는 않는다.

## 다음 단위와의 연결
이 단위에서 SFT를 instruction 형식에 맞춘 **supervised imitation**으로 정리하면, 다음 단위 `05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto`에서 왜 preference optimization이 필요한지 자연스럽게 이어진다. SFT는 assistant의 기본 말투와 출력 형식을 맞추고, preference optimization은 그 위에서 더 선호되는 응답 선택 기준을 추가로 학습한다.
