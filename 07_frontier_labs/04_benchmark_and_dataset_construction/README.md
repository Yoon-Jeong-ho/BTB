# 04 Benchmark and Dataset Construction

> Status: runnable
> 이 단위는 CPU에서 바로 실행되는 deterministic benchmark/dataset construction 실습이다. 외부 dataset이나 network 없이 toy record 12개를 사용해 task contract, dataset schema, source/split manifest, annotation rubric/QC, leakage/contamination/drift audit, benchmark card, versioning, report template를 만든다.

## 왜 이 단위를 배우는가
앞 단위에서 agentic loop와 verifier gate를 세웠더라도, 그 loop가 최적화하는 benchmark와 dataset contract가 약하면 자동화된 속도로 잘못된 신호를 키우게 된다. 그래서 frontier 실험에서 benchmark construction은 리더보드 장식이 아니라 **무엇을 성공으로 읽을지와 무엇은 아직 주장할 수 없는지를 함께 고정하는 측정 계약**이다.

Dataset construction도 단순 수집 작업이 아니다. 어떤 샘플을 한 개 사례로 볼지, split을 어디서 끊을지, annotation disagreement를 어떻게 처리할지, contamination과 drift를 어떻게 감시할지까지 정해야 benchmark가 실험 운영의 기준점이 된다. 이 단위는 benchmark/dataset을 모델 바깥의 부속물이 아니라 **연구 운영 전체를 정직하게 만드는 인터페이스**로 읽게 한다.

## 이번 단위에서 남길 것
- `scratch_lab.py`: toy record를 직접 검수해 benchmark card, task contract, dataset schema, source/split manifest, annotation rubric/QC, leakage/contamination/drift audit, versioning, report template, SVG를 만든다.
- `framework_lab.py`: dataclass 기반 pipeline처럼 같은 benchmark/dataset construction 계약을 재구성한다.
- `analysis.py`: 두 metrics가 없으면 명확히 실패하고, 있으면 실행별 관측 보고서 `artifacts/analysis-manual/latest_report.md`를 만든다.
- `analysis.md`: 반복 실행해도 안정적으로 유지되는 해석 프레임이다.
- `reflection.md`: 한국어 우선 회고 질문이다.

## 실행 방법
```bash
python 07_frontier_labs/04_benchmark_and_dataset_construction/scratch_lab.py
python 07_frontier_labs/04_benchmark_and_dataset_construction/framework_lab.py
python 07_frontier_labs/04_benchmark_and_dataset_construction/analysis.py
```

생성물은 다음 위치에 남는다.
- `artifacts/scratch-manual/metrics.json`
- `artifacts/scratch-manual/benchmark_dataset_overview.svg`
- `artifacts/framework-manual/metrics.json`
- `artifacts/analysis-manual/latest_report.md`

## 실행 결과 예시
아래 숫자는 이 단위의 deterministic toy benchmark에서 실제로 재현되는 관측 형태다.

```text
$ python 07_frontier_labs/04_benchmark_and_dataset_construction/scratch_lab.py
{
  "benchmark_card": {"benchmark_id": "btb-agent-benchmark-v1"},
  "task_contract": {"unit_of_record": "agent_task_record"},
  "source_manifest": {"raw_records": 12, "accepted_records": 10},
  "split_manifest": {
    "counts": {"dev": 4, "test_private": 3, "test_public": 3},
    "source_disjoint": true,
    "template_family_disjoint": true
  },
  "annotation_qc": {"double_label_rate": 0.4, "agreement_score": 0.8},
  "leakage_contamination_drift_audit": {"contamination_flags": 2},
  "figure_path": "artifacts/scratch-manual/benchmark_dataset_overview.svg"
}

$ python 07_frontier_labs/04_benchmark_and_dataset_construction/framework_lab.py
{
  "device": "cpu",
  "simulation": "deterministic_benchmark_dataset_pipeline",
  "dataset_size": 10,
  "splits": ["dev", "test_public", "test_private"],
  "versioning": {"version": "v1.0.0", "historically_comparable_to_v0": false}
}

$ python 07_frontier_labs/04_benchmark_and_dataset_construction/analysis.py
# 04 Benchmark and Dataset Construction 실행 관측
...
```

## 관찰 포인트
1. **task contract**: benchmark가 점수판이 아니라 input/output/unit of record/claim boundary를 고정하는 측정 계약인지 확인한다.
2. **dataset schema**: 필수 field와 optional metadata, license tier, missing value policy가 재현 가능한 dataset contract를 만드는지 본다.
3. **source/split manifest**: source와 template family를 split 사이에서 분리해 leakage를 줄이고, public/private holdout의 역할을 구분한다.
4. **annotation rubric / QC**: task_success, groundedness, policy_compliance를 분리하고 double-label, agreement, adjudication rule을 함께 남긴다.
5. **leakage / contamination / drift audit**: exact overlap이 0이어도 near duplicate, judge prompt contamination, tool-schema drift warning이 score 해석을 제한할 수 있음을 본다.
6. **benchmark card / versioning / report template**: primary claim, known non-goals, frozen version, known limits를 report template에 고정해 다음 연구 트랙이 숫자와 경고를 함께 남기게 한다.

## 다음 단위와의 연결
다음 단위 `07_frontier_labs/05_open_ended_research_tracks`에서는 benchmark가 고정된 뒤, 정답이 없는 연구 질문을 어떤 stopping rule과 evidence 기준으로 쪼갤지 다룬다. 이 단위에서 benchmark contract, split hygiene, QC gate를 먼저 세워 두면 다음 연구 트랙에서는 막연히 새 아이디어를 많이 시도하는 대신 **무엇이 실제 개선이고 무엇이 benchmark 착시인지**를 더 빠르게 구분할 수 있다.
