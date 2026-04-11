# 02 Capstone Model Building 분석

## Stable interpretation

Capstone model building is a contract exercise before it is a training exercise. The stable reading for this unit is:

1. A **problem statement** must name the user-facing input/output, baseline, metric, and target delta.
2. **Non-goals** are not optional humility; they prevent scope creep from turning one capstone into several projects.
3. A **dataset contract** fixes source, split, schema fields, leakage controls, and label quality checks before model claims begin.
4. A **model contract** names a boring baseline and one or more candidates that are evaluated under the same frozen constraints.
5. An **eval contract** pairs a primary metric such as Recall@10 with secondary metrics, slice review, and qualitative failure buckets.
6. **Milestones** close decisions and required artifacts through acceptance gates, not just code tasks.
7. A **risk register** and **failure analysis** outline turn bad results into evidence for the next experiment.

## Korean-first reading

이 단위의 핵심은 "멋진 모델을 고르는 법"이 아니라, 모델을 만들기 전에 프로젝트가 어떤 언어로 끝날지 고정하는 것이다. problem statement와 non-goal이 약하면 구현 중 목표가 늘어나고, dataset/model/eval contract 중 하나가 비면 결과가 좋아도 해석할 수 없다. acceptance gate와 risk register를 먼저 쓰면, 실험이 실패했을 때도 다음 행동이 문서 안에서 바로 이어진다.

## What `analysis.py` observes

`analysis.py`는 `scratch_lab.py`가 만든 capstone contract와 `framework_lab.py`가 만든 project board를 함께 읽는다. 그리고 다음 항목을 `artifacts/analysis-manual/latest_report.md`로 요약한다.

- problem statement / non-goals
- dataset / model / eval contract
- milestone acceptance gates
- risk register
- failure-analysis outline
- report outline
- next handoff to the agentic training/eval loop

## Failure modes to notice

- **scope creep**: serving, personalization, external collection이 M0 이후 몰래 들어온다.
- **dataset leakage**: near duplicate product가 split을 넘나들어 Recall@10 개선처럼 보인다.
- **baseline weakness**: baseline이 너무 약해서 candidate improvement가 과장된다.
- **metric gaming**: Recall@10은 올랐지만 brand/OCR slice가 나빠진다.
- **report drift**: run artifact가 final report section과 연결되지 않는다.

## Next unit handoff

다음 `07_frontier_labs/03_agentic_training_and_eval_loops`에서는 이 contract를 loop 입력으로 사용한다. planner는 frozen constraints와 acceptance gates를 읽고, verifier는 artifact completeness와 protocol match를 확인하며, critic은 risk register와 failure-analysis outline을 근거로 retry / rollback / escalation을 고른다.
