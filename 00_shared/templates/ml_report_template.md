# ML Study Report Template

> 이 템플릿은 결과만 적는 보고서가 아니라, **이론-실험-결과-해석-다음 가설** 을 함께 남기기 위한 형식이다.

## 1. 실험 목적

- 무엇을 검증하려는 실험인가?
- 왜 지금 이 실험을 돌렸는가?
- 연결 이론 문서: `[THEORY.md](...)`

## 2. 이론 배경 요약

- 이번 실험에서 핵심인 이론 3~5개
- 왜 그 이론이 필요한가
- 이번 실험에서 특히 중요했던 metric / split / preprocessing 이론

## 3. 실험 설정

- 데이터셋 / split
- 전처리
- 모델 후보
- 비교 기준 baseline
- 실행 환경

## 4. 메트릭 설명

- primary metric의 의미
- secondary metric의 의미
- 이 실험에서 metric을 읽을 때 주의할 점

## 5. 결과 요약

- 모델 비교표
- 최고 모델
- baseline 대비 개선폭

## 6. 결과 Figure 해석

각 figure마다 아래를 쓴다.

### figure name
- 링크: `figures/results/...`
- 이 figure가 답하는 질문
- 실제로 관찰된 패턴
- 그 패턴이 의미하는 것

## 7. 분석 Figure 해석

각 analysis figure마다 아래를 쓴다.

### figure name
- 링크: `figures/analysis/...`
- 무엇을 분석한 것인가
- 실제로 관찰된 실패 패턴
- 모델/데이터 관점에서 가능한 원인

## 8. 실패 사례 분석

- 대표 실패 사례 3개 이상
- 왜 틀렸다고 해석하는가
- 어떤 추가 feature / split / loss / threshold 조정이 도움이 될 수 있는가

## 9. 결론

- 이번 실험에서 배운 점
- 아직 남은 한계
- 다음 실험 가설
