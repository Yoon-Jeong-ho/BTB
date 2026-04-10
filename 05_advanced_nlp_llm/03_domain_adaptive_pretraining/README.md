# 03 Domain Adaptive Pretraining

> Status: runnable
>
> 이 단위는 **CPU-safe, deterministic, toy continued-pretraining comparison**만 다루는 runnable 단계다. 큰 LLM을 학습하지 않고도 domain shift가 있는 corpus를 더 먹일 때 in-domain gain과 catastrophic forgetting risk가 어떻게 함께 움직이는지 관찰한다.

## 왜 이 단위를 배우는가
base LM이 일반 텍스트 분포를 넓게 익혔다고 해서 의료·법률·금융·사내 문서 같은 특정 도메인을 곧바로 잘 읽는 것은 아니다. 실제 현장에서는 같은 pretraining objective를 유지한 채 특정 domain corpus로 몇 step 더 학습하는 **continued pretraining**, 특히 **domain adaptive pretraining(DAPT)** 을 검토한다.

이 단위의 목표는 DAPT를 “데이터를 더 넣는 단계”로 축소하지 않는 것이다. DAPT는 **domain shift를 줄이는 specialization gain**과 **기존 일반 분포를 덜 기억하게 되는 catastrophic forgetting cost**를 동시에 관리하는 적응 설계 문제다.

## 이번 단위에서 남길 것
- scratch DAPT 비교 관측치 `artifacts/scratch-manual/metrics.json`
- scratch trade-off SVG `artifacts/scratch-manual/dapt_tradeoff.svg`
- 작은 deterministic PyTorch bigram LM continued-pretraining 관측치 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자 회고 질문 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 base model의 in-domain/general validation loss를 고정하고, pure domain schedule과 replay mixture schedule의 toy loss trajectory를 비교한다.
2. 같은 causal LM objective를 유지한다는 전제 아래 **domain share / general replay share**가 adaptation speed와 retention cost를 어떻게 바꾸는지 본다.
3. `dapt_tradeoff.svg`에서 in-domain loss는 내려가지만 general retention loss가 올라갈 수 있음을 시각적으로 확인한다.
4. `framework_lab.py`에서 작은 PyTorch bigram LM을 일반 corpus로 먼저 pretrain한 뒤, pure domain continued pretraining과 replay mixture continued pretraining을 같은 시작점에서 비교한다.
5. metrics 안의 data selection profile을 읽으며, curated small corpus가 noisy large corpus보다 DAPT 신호가 좋을 수 있는 이유를 정리한다.
6. `analysis.py`로 stable `analysis.md`와 실행별 observed report를 분리해, 해석 프레임과 최신 관측값을 따로 보관한다.

## 이번 단위에서 특히 볼 질문
- DAPT는 from-scratch pretraining이나 일반 fine-tuning과 무엇이 다르고, 왜 objective를 유지한 채 데이터 분포만 바꾼다고 말하는가?
- domain shift는 vocabulary 차이 외에 문체, 문서 형식, 정보 밀도, 최신성 차이로 어떻게 나타나는가?
- pure domain continued pretraining은 왜 빠른 specialization을 주면서 catastrophic forgetting risk도 키울 수 있는가?
- replay mixture는 general retention을 지키는 대신 adaptation speed를 얼마나 늦출 수 있는가?
- data selection에서는 왜 문서 수보다 품질, 중복, contamination, 라이선스, 최신성을 먼저 점검해야 하는가?
- stopping은 왜 in-domain validation loss 하나가 아니라 general-domain retention guardrail과 함께 정해야 하는가?

## 실행 방법
```bash
python 05_advanced_nlp_llm/03_domain_adaptive_pretraining/scratch_lab.py
python 05_advanced_nlp_llm/03_domain_adaptive_pretraining/framework_lab.py
python 05_advanced_nlp_llm/03_domain_adaptive_pretraining/analysis.py
```

## 실행 결과 예시
아래는 이 디렉터리에서 **실제로 실행되는 command/output shape**다.

```text
$ python 05_advanced_nlp_llm/03_domain_adaptive_pretraining/scratch_lab.py
{
  "setup": {
    "objective_kept_constant": "causal_lm",
    "general_regression_guardrail": 0.18
  },
  "comparison": {
    "fastest_adapter": "pure_domain",
    "safer_retention": "replay_mixture",
    "balanced_recommendation": "replay_mixture"
  },
  "figure_path": "artifacts/scratch-manual/dapt_tradeoff.svg"
}

$ python 05_advanced_nlp_llm/03_domain_adaptive_pretraining/framework_lab.py
{
  "device": "cpu",
  "objective_kept_constant": "causal_lm_bigram_next_token",
  "base_losses": {
    "general": 1.758509,
    "domain": 3.98566
  },
  "comparison": {
    "pure_domain_adapts_faster": true,
    "replay_preserves_general_better": true
  }
}

$ python 05_advanced_nlp_llm/03_domain_adaptive_pretraining/analysis.py
# 03 Domain Adaptive Pretraining 실행 관측
- pure domain과 replay mixture의 in-domain gain / general regression / stopping signal을 한국어 리포트로 저장한다.
```

실행 뒤에는 `dapt_tradeoff.svg`에서 **in-domain adaptation과 General retention 선**을 함께 보고, `metrics.json`에서 pure domain schedule이 더 빠르게 적응하지만 general regression guardrail을 더 빨리 건드리는지 확인하라.

## 해석 포인트 요약
- **domain shift**: 일반 업무 문서와 임상 기록처럼 token distribution, 문서 형식, 용어 co-occurrence가 달라지는 상황이다.
- **continued pretraining**: architecture나 objective를 바꾸는 것이 아니라 기존 base LM을 새 corpus 분포 쪽으로 더 이동시키는 과정이다.
- **catastrophic forgetting**: 새 도메인에는 좋아지지만 general-domain validation loss가 나빠지는 trade-off로 먼저 관측한다.
- **replay mixture**: domain data에 general replay를 섞어 forgetting을 늦추는 대신 adaptation 속도를 낮출 수 있다.
- **data selection / stopping**: DAPT 성공 여부는 어떤 corpus를 넣고 언제 멈추는지에 크게 좌우된다.

## 다음 단위와의 연결
이 단위에서 DAPT로 base LM의 분포 감각을 특정 도메인 쪽으로 먼저 당겨 두면, 다음 단위 `05_advanced_nlp_llm/04_instruction_tuning_and_sft`에서는 그 지식을 실제 instruction-response 행동으로 바꾸는 문제를 더 분리해서 볼 수 있다. DAPT는 “무엇을 더 잘 알게 만들 것인가”에 가깝고, SFT는 “그 지식을 어떤 형식으로 드러내게 만들 것인가”에 가깝다.
