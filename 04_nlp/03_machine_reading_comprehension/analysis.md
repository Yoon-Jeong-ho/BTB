# 03 Machine Reading Comprehension 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 span extraction, partial overlap, no-answer threshold를 읽는 안정적인 해석 프레임만 남겨 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- MRC의 첫 질문은 "무슨 pretrained QA 모델을 붙일까" 보다 먼저, 질문 token이 문맥 어느 구간과 만나는지와 정답이 없을 때 멈출 기준이 있는지를 확인하는 것이다.
- exact match는 정답 span을 완전히 맞혔는지 묻는다. token F1은 경계를 조금 틀려도 핵심 단어를 얼마나 겹치게 잡았는지 보여 준다. 둘을 같이 읽어야 boundary error를 놓치지 않는다.
- scratch baseline이 잘 되면 question-context lexical alignment만으로도 풀리는 패턴이 있다는 뜻이다. 반대로 no-answer threshold가 흔들리면 질문은 읽었어도 abstention 기준이 약하다는 뜻이다.
- tiny PyTorch QA model은 질문 summary를 문맥 token에 다시 조건부로 섞어 본다. 따라서 heuristic보다 나아졌다면 단순 token overlap보다 조금 더 풍부한 질문-문맥 상호작용을 썼을 가능성이 있다.
- 오답을 읽을 때는 span이 완전히 틀렸는지, 정답 일부만 맞았는지, 애초에 답이 없는데도 억지로 답했는지를 분리해서 보는 편이 다음 실험 가설을 세우기 좋다.

## 확인 질문
- EM과 token F1이 다르게 말해 주는 boundary failure pattern은 무엇인가?
- answerable / unanswerable를 같이 볼 때 no-answer threshold는 어디서 작동하는가?
- framework 모델이 개선되었다면 그것은 질문 조건부 표현 덕분인가, 아니면 toy dataset의 surface overlap이 이미 충분했기 때문인가?

## 관련 이론
- [THEORY.md](./THEORY.md): span extraction, exact match, token F1, no-answer threshold 핵심 개념을 다시 확인한다.
