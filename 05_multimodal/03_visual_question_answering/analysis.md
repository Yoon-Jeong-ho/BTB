# 03 Visual Question Answering 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 VQA를 해석하는 안정적인 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- VQA의 첫 질문은 “overall accuracy가 높은가?”가 아니라, **어떤 answer type이 병목인가?** 다.
- yes/no, color, count는 필요한 시각 근거와 난도가 다르므로 반드시 분리해 읽어야 한다.
- count가 흔들리면 모델이 장면의 집계 정보를 제대로 쓰지 못했을 가능성이 크다. 이는 단순 오답 하나보다 더 구조적인 힌트다.
- row 단위 기록에서 `predicted_answer` 와 `error_reason` 을 함께 남기면, shortcut bias인지 grounding failure인지 빠르게 분류할 수 있다.
- tiny PyTorch multimodal classifier는 큰 VLM이 아니어도, **이미지 표현 + 질문 표현 + answer vocabulary** 구조로 VQA 핵심을 충분히 재현한다.

## 확인 질문
- 이번 run에서 가장 어려웠던 answer type은 무엇이며, 왜 그렇게 해석했는가?
- scratch failure는 질문 이해 문제였는가, 이미지 grounding 문제였는가, 아니면 prior 문제였는가?
- framework에서 count accuracy가 회복되었다면, 그 변화는 qualitative row에서 어떻게 보이는가?

## 관련 실행 로그
- 최신 실행 관측: `artifacts/analysis-manual/latest_report.md`

## 관련 이론
- [THEORY.md](./THEORY.md): answer type, shortcut bias, multimodal fusion, grounded reasoning 개념을 다시 확인한다.
