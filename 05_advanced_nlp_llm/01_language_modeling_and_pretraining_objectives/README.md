# 01 Language Modeling and Pretraining Objectives

> Status: runnable
>
> 이 단위는 **CPU-safe, deterministic, toy objective comparison**만 다루는 runnable 단계다. 큰 모델을 학습하지 않고도 causal LM, masked LM, span corruption이 **무엇을 입력으로 받고 무엇을 정답으로 삼는지**, 그리고 **loss-mask density / context window 직관**이 어떻게 달라지는지를 직접 관찰한다.

## 왜 이 단위를 배우는가
LLM pretraining을 이해할 때 architecture만 보면 절반만 본 셈이다. 같은 transformer 계열이라도 **target framing**이 causal LM인지, masked LM인지, span corruption인지에 따라 모델이 보는 문맥과 loss가 걸리는 위치가 달라지고, 이후 더 잘하게 되는 행동도 달라진다.

이 단위는 “objective = 코드 속 옵션”이 아니라 **학습 신호 설계**라는 감각을 남기기 위해 만들어졌다. 여기서 만든 비교 틀은 다음 단위 `05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture`에서 token budget과 tokenizer/data mixture를 해석할 때 바로 이어진다.

## 이번 단위에서 남길 것
- scratch objective 비교 관측치 `artifacts/scratch-manual/metrics.json`
- scratch objective 비교 SVG `artifacts/scratch-manual/objective_comparison.svg`
- framework objective 비교 관측치 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자 회고 질문 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 같은 문장을 causal LM / masked LM / span corruption으로 각각 다시 framing한다.
2. 각 objective가 **어디에 loss를 걸고**, 그 비율(loss-mask density)이 얼마나 다른지 계산한다.
3. 같은 context window(`window_tokens=4`)를 고정해 두고도, objective가 바뀌면 어떤 토큰을 볼 수 있는지 달라짐을 비교한다.
4. `framework_lab.py`에서 작은 deterministic PyTorch tensor와 cross entropy만 사용해 scored token 수, mean loss, density ranking을 다시 확인한다.
5. `analysis.py`로 stable `analysis.md`와 실행별 observed report를 분리해, 해석 프레임과 최신 관측값을 따로 보관한다.

## 이번 단위에서 특히 볼 질문
- causal LM은 왜 **다음 토큰 prediction**으로 읽고, masked LM은 왜 **빈칸 복원**으로 읽는가?
- span corruption은 token 하나를 맞히는 문제와 무엇이 다르고, sentinel token은 왜 필요한가?
- 같은 context window여도 objective에 따라 “볼 수 있는 문맥”이 왜 다르게 느껴지는가?
- loss-mask density가 높을수록 supervision은 촘촘해지지만, 그 자체가 항상 더 좋은 objective를 뜻하지는 않는 이유는 무엇인가?

## 실행 방법
```bash
python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/scratch_lab.py
python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/framework_lab.py
python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/analysis.py
```

## 실행 결과 예시
아래는 이 디렉터리에서 **실제로 실행되는 command/output shape**다.

```text
$ python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/scratch_lab.py
{
  "context_window_tokens": 4,
  "comparisons": {
    "densest_supervision": "causal_lm",
    "sparsest_supervision": "masked_lm"
  },
  "figure_path": "artifacts/scratch-manual/objective_comparison.svg"
}

$ python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/framework_lab.py
{
  "device": "cpu",
  "vocab_size": 11,
  "density_ranking": [
    "causal_lm",
    "span_corruption",
    "masked_lm"
  ]
}

$ python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/analysis.py
# 01 Language Modeling and Pretraining Objectives 실행 관측
- target framing, loss-mask density, context window 직관을 한국어 리포트로 저장한다.
```

실행 뒤에는 `objective_comparison.svg`에서 **objective별 loss-mask density 막대**를 보고, `metrics.json`에서 causal LM / masked LM / span corruption의 scored token 수와 visible context 메모를 함께 읽어 보라.

## 해석 포인트 요약
- **causal LM**: 왼쪽 문맥만 보고 거의 모든 시점에 loss를 건다.
- **masked LM**: 일부 mask 위치에만 loss를 걸고, 그 위치는 양쪽 문맥을 함께 본다.
- **span corruption**: encoder는 손상된 입력을 보고, decoder는 sentinel을 기준으로 빠진 span을 autoregressive하게 복원한다.
- 따라서 target framing이 달라지면 **loss-mask density**, **context window에서 실제로 활용되는 문맥**, **후속 task와의 alignment 감각**이 함께 달라진다.

## 다음 단위와의 연결
여기서 objective를 비교하는 틀을 잡아 두면, 다음 단위 `05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture`에서 “어떤 tokenizer / data mixture가 어떤 objective와 잘 맞는가?”를 더 자연스럽게 묻게 된다. 이후 `05_advanced_nlp_llm/03_domain_adaptive_pretraining`에서는 objective를 고정한 채 데이터 분포만 바뀔 때 무엇이 변하는지도 더 명확히 해석할 수 있다.
