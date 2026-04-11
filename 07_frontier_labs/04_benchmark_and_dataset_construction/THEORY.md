# 04 Benchmark and Dataset Construction 이론 노트

## 핵심 개념

### 1. benchmark는 점수판이 아니라 측정 계약이다
- benchmark를 흔히 모델 순위를 매기는 leaderboard로만 생각하지만, 실제로 더 중요한 역할은 **어떤 claim을 허용하고 어떤 claim을 보류할지 정하는 측정 계약**을 만드는 데 있다.
- 좋은 benchmark는 최소한 다음 질문에 답할 수 있어야 한다.
  - 무엇을 측정하는가?
  - 어떤 입력/출력 단위를 한 사례로 보는가?
  - 어떤 metric과 slice가 핵심인가?
  - score가 올랐을 때 무엇을 개선이라고 해석할 수 있는가?
  - score가 올라도 여전히 주장하면 안 되는 것은 무엇인가?
- 이 질문이 비어 있으면 agentic loop든 capstone이든 결국 **최적화는 빨라지고 해석은 약해지는** 상태가 된다.
- 그래서 benchmark construction의 출발점은 데이터 모으기가 아니라 task contract를 문장으로 고정하는 일이다.

### 2. dataset contract는 파일 목록이 아니라 운영 경계다
- dataset contract는 단순히 컬럼 이름 몇 개를 적는 schema 문서보다 넓다.
- 보통 다음 요소가 함께 들어가야 한다.
  - source boundary: 어떤 출처를 허용하고 어떤 출처를 제외하는가?
  - unit of record: 문서, 질의응답 쌍, 대화 turn, trajectory, preference pair 중 무엇을 한 샘플로 보는가?
  - schema: 필수 필드, optional 필드, null 처리, metadata 규칙
  - collection policy: 수집 방식, sampling 방식, filtering 기준
  - rights / license: 사용 가능 범위와 공개 가능 범위
  - versioning: 어떤 시점의 snapshot을 freeze하는가?
- 이 계약이 중요한 이유는 benchmark가 모델 성능만이 아니라 **데이터 경계와 평가 경계까지 함께 재현 가능해야** 하기 때문이다.
- 같은 task처럼 보여도 unit of record가 다르면 metric 의미가 완전히 달라질 수 있다. 예를 들어 conversation-level success와 turn-level correctness는 같은 숫자로 읽을 수 없다.

### 3. split hygiene는 random split보다 넓은 문제다
- 많은 benchmark가 train/dev/test를 나누는 순간 끝났다고 생각하지만, 실제로는 split hygiene가 더 어렵다.
- random split이 충분하지 않은 대표 상황은 다음과 같다.
  - 같은 user/source/template family가 여러 split에 섞이는 경우
  - time drift가 큰 데이터에서 과거/현재 사례가 함께 뒤섞이는 경우
  - near-duplicate나 paraphrase가 다른 split으로 넘어가는 경우
  - annotation guideline 자체가 split마다 다르게 적용되는 경우
- 그래서 split을 설계할 때는 보통 **무엇이 서로 독립이어야 하는가**를 먼저 생각한다.
  - source disjoint
  - user/group disjoint
  - time-based holdout
  - template family disjoint
  - scenario category / topic disjoint
- split hygiene의 핵심은 예쁜 비율이 아니라 **한 split에서 배운 지름길이 다른 split 점수를 부당하게 올리지 못하게 막는 것**이다.
- benchmark가 private holdout을 유지하는 이유도 여기에 있다. public benchmark만으로 반복 최적화하면 결국 public benchmark 특유의 패턴에 과적합할 수 있다.

### 4. annotation 품질은 label 수보다 rubric과 disagreement 처리에서 결정된다
- dataset construction에서 자주 생기는 착각은 annotator를 많이 붙이거나 label 수를 많이 모으면 자동으로 품질이 올라간다고 믿는 것이다.
- 실제로 더 중요한 것은 다음과 같다.
  - rubric이 구체적인가?
  - 애매한 사례에 abstain 또는 ambiguous tag를 허용하는가?
  - multi-annotator overlap이 있는가?
  - major disagreement를 어떻게 adjudicate하는가?
  - annotator calibration 예시가 축적되는가?
- disagreement는 항상 제거해야 할 noise가 아니다.
- 어떤 disagreement는 task definition이 약하다는 신호이거나, benchmark가 실제 세계의 애매함을 억지로 하나의 label로 접고 있다는 신호일 수 있다.
- 따라서 annotation QC는 agreement score 하나로 끝나지 않는다.
  - invalid label rate
  - missing metadata rate
  - annotator-by-annotator drift
  - hard slice에서의 disagreement concentration
  - adjudication turnaround와 수정 이력
- 좋은 benchmark는 label을 깨끗하게 보이게 만드는 대신, **어디서 사람이 헷갈렸는지까지 함께 남긴다.**

### 5. leakage, contamination, benchmark gaming은 서로 닮았지만 구분해야 한다
- **leakage** 는 보통 split 사이에 동일하거나 거의 동일한 정보가 흘러 들어가 평가가 쉬워지는 문제다.
- **contamination** 은 benchmark/test 정보가 training corpus나 prompt template, evaluator instruction 등으로 스며들어 점수가 과장되는 문제다.
- **benchmark gaming** 은 모델이나 시스템이 실제 capability보다 benchmark 특유의 규칙에 과적합하는 현상이다.
- 셋은 함께 나타나지만 관찰 포인트가 다르다.
  - leakage: split disjoint와 dedup audit가 핵심
  - contamination: upstream corpus / prompt / judge overlap audit가 핵심
  - gaming: slice별 failure와 private holdout 비교가 핵심
- exact overlap만 잡는 것으로 충분하지 않을 때가 많다.
  - paraphrase overlap
  - translated overlap
  - retrieval corpus를 통한 우회 접근
  - judge prompt / rubric leakage
  - synthetic data 생성 과정에서 test style이 복제되는 경우
- benchmark construction에서는 높은 점수보다 먼저 **이 점수가 얼마나 오염되지 않았는가**를 계속 질문해야 한다.

### 6. drift를 보지 않으면 benchmark는 빠르게 낡거나 왜곡된다
- benchmark drift는 데이터 분포, 사용자 요구, tool schema, annotation 기준, evaluator behavior가 시간이 지나며 바뀌는 현상이다.
- drift를 무시하면 두 가지 극단이 생긴다.
  - benchmark가 현실과 멀어져 좋은 점수도 실제 품질을 설명하지 못함
  - benchmark를 너무 자주 바꿔 과거 run과 비교가 불가능해짐
- 그래서 benchmark는 보통 두 층으로 운영하는 편이 낫다.
  - **frozen core benchmark**: 장기 비교용
  - **refresh slice / challenge set**: 새 failure mode 감시용
- 이때 중요한 것은 update 여부보다 version note다.
  - 언제 바뀌었는가?
  - 무엇이 바뀌었는가?
  - 과거 점수와 직접 비교 가능한가?
  - 새 score를 해석할 때 빠진 warning은 무엇인가?
- benchmark governance는 모델보다 덜 화려하지만, frontier 팀에서는 이 운영 감각이 score 신뢰도를 결정한다.

### 7. reporting template까지 설계해야 benchmark가 운영 가능해진다
- benchmark를 잘 만들었다고 말하려면 결과를 어떻게 보고할지도 미리 정해 두어야 한다.
- 최소 보고 항목은 보통 다음을 포함한다.
  - task contract와 known non-goals
  - dataset version / split manifest / sample counts
  - annotation rubric과 QC 통계
  - primary metric과 slice metric
  - contamination / leakage audit 결과
  - known limits와 drift watchlist
- 이 reporting template가 있어야 agentic loop나 연구 트랙이 결과를 올릴 때 **숫자와 함께 경고 문장도 같이 남길 수 있다.**
- 결국 benchmark construction은 dataset 만들기에서 끝나는 일이 아니라, **점수를 운영 가능한 evidence bundle로 만드는 일**이다.

## 직관 / 운영 프레임
- benchmark 신뢰도를 아주 거칠게 생각하면 다음 요소가 함께 곱해진다고 볼 수 있다.
  - `trust ≈ task_clarity × split_hygiene × annotation_quality × contamination_resistance`
- 어느 하나가 0에 가까우면 metric이 높아도 해석력은 급격히 떨어진다.
- 특히 contamination이나 split leakage는 metric 자체보다 작게 보일 수 있어도, 실제 claim 신뢰도는 크게 깎아 먹는다.

## 자주 헷갈리는 지점
- benchmark를 leaderboard와 동의어로 보는 실수
- dataset contract를 schema 정의만으로 충분하다고 생각하는 실수
- random split이면 leakage가 대부분 해결된다고 믿는 실수
- annotator agreement 숫자 하나로 label 품질을 다 설명할 수 있다고 생각하는 실수
- contamination check를 exact string overlap 검색으로 축소하는 실수
- benchmark가 낡았을 때 refresh만 자주 하면 해결되고 비교 가능성 문제는 사소하다고 여기는 실수
- score가 올랐으면 benchmark 설계도 자동으로 좋은 것이었다고 거꾸로 추론하는 실수

## 이 단위에서 무엇을 관찰할 것인가
- 현재 task contract가 실제 사용 시나리오와 평가하려는 claim을 충분히 연결하고 있는가?
- dataset contract 안에 source boundary, unit of record, license, version freeze가 명시돼 있는가?
- split이 random balance만 맞춘 것이 아니라 leakage가 생기기 쉬운 축을 실제로 분리하고 있는가?
- annotation disagreement가 단순 noise로 덮이지 않고 ambiguous slice 신호로 기록되는가?
- contamination audit가 exact overlap 너머의 paraphrase / template / evaluator leakage까지 보려 하는가?
- benchmark refresh 필요성과 historical comparability가 같은 문서 안에서 함께 관리되는가?
- score report에 known limits와 drift warning이 빠지지 않는가?

## Runnable 실습 용어 연결
- 이 runnable 실습에서는 **benchmark card**가 primary claim과 known non-goals를 고정한다.
- **task contract**는 input/output/unit of record와 claim boundary를 묶고, **dataset schema**는 필수 field와 optional metadata를 고정한다.
- **source/split manifest**는 source와 template family 단위 disjointness를 확인해 leakage를 줄인다.
- **annotation rubric**과 **QC**는 agreement score, major disagreement, adjudication rule을 함께 남긴다.
- **contamination**과 **drift** audit는 score 해석 전에 붙여야 하는 warning이며, **versioning**과 **report template**는 다음 연구 트랙의 비교 가능성을 보호한다.
