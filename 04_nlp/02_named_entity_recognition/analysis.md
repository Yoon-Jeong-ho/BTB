# 02 Named Entity Recognition 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 BIO alignment, boundary error, entity-level F1을 읽는 안정적인 해석 프레임만 남겨 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- NER의 첫 질문은 "어떤 큰 모델을 붙일까" 보다 먼저, gold label이 어떤 token 단위에 맞춰졌는지와 BIO 규칙이 깨지지 않았는지를 확인하는 것이다.
- token accuracy는 label 분포가 `O` 쪽으로 기울 때 쉽게 높아질 수 있다. 그래서 entity-level precision / recall / F1을 같이 읽어야 실제 span 복원 능력을 놓치지 않는다.
- scratch baseline이 자주 틀리는 곳은 보통 unseen surface form이나 boundary 확장 구간이다. 즉 alignment나 lexical lookup에만 기대는 방식의 한계가 드러난다.
- tiny neural sequence labeler는 앞뒤 token 문맥을 같이 본다. 따라서 같은 piece라도 문장 안 위치와 주변 token에 따라 `B-` / `I-` / `O` 결정을 더 유연하게 조정할 여지가 있다.
- 오분류를 읽을 때는 단순히 라벨이 틀렸다는 사실보다, entity가 아예 누락됐는지, span 길이가 어긋났는지, 타입만 바뀌었는지를 분리해서 보는 편이 학습 가설을 세우기 좋다.

## 확인 질문
- alignment 후 첫 piece와 뒤 piece는 각각 어떤 BIO 규칙을 따라야 하는가?
- token accuracy와 entity-level F1이 다르게 말해 주는 failure pattern은 무엇인가?
- framework 모델이 개선되었다면 그 차이는 context 이해 때문인가, 아니면 단순 vocabulary overlap 덕분인가?

## 관련 이론
- [THEORY.md](./THEORY.md): BIO tagging, label alignment, entity-level F1 핵심 개념을 다시 확인한다.
