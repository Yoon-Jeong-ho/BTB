# 04 Large Scale Tabular

## 목표

로컬에서 익힌 tabular 실험 규약을 서버 환경으로 옮기기 전에 `대형 데이터셋`, `비용`, `throughput`, `artifact 분리` 를 연습한다.

## 이번 프로젝트 기준 확정 데이터셋

- 실행 코드: `run_stage.py`

- Primary: `mstz/covertype`
- Source: Hugging Face Datasets
- Load:

```python
from datasets import load_dataset

ds = load_dataset("mstz/covertype", split="train")
df = ds.to_pandas()
```

- 이유: Hugging Face로 바로 접근 가능하고, 로컬에서 다루기엔 크고 서버 실습용으론 아직 관리 가능한 수준이다.
- Server extension: `HIGGS`

## 실습 파이프라인

1. 데이터 원본은 Git 밖에 두고 fetch/prep 스크립트만 버전 관리
2. schema와 split manifest 고정
3. baseline으로 선형/얕은 tree 모델부터 시작
4. histogram 기반 GBDT 또는 boosted tree로 확장
5. metric뿐 아니라 학습 시간, peak memory, 데이터 적재 시간을 같이 기록
6. checkpoint와 중간 feature matrix는 외부 저장소로 분리

## 결과로 남길 figure

- metric vs training time
- metric vs memory
- class distribution / score distribution

## 분석으로 남길 figure

- slice metric by class
- throughput bottleneck summary
- sampling 전략에 따른 성능 차이

## 승격 기준

- 같은 규약으로 로컬 실험과 서버 실험이 비교 가능하다.
- 어떤 artifact를 Git에 두고 어떤 artifact를 외부에 둬야 하는지 명확하다.
