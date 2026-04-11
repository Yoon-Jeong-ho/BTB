# 05 Open-Ended Research Tracks

> Status: runnable
>
> 이 단위는 외부 서비스나 GPU 없이 실행되는 **CPU-safe deterministic simulation**이다. 목적은 breakthrough story를 만드는 것이 아니라, 열린 연구를 research scope, north-star question, hypothesis registry, iteration boundary, kill criteria, evidence standard, stop / pause / escalate / archive decision, reopen condition까지 이어지는 운영 기록으로 만드는 것이다.

## 왜 이 단위를 배우는가
앞 단위들에서 우리는 논문 재현, capstone build, agentic loop, benchmark/dataset contract까지 차례로 고정했다. 그런데 frontier 연구는 그다음부터가 더 어렵다. benchmark를 세웠다고 해서 다음 행동이 자동으로 주어지는 것은 아니고, 정답이 분명하지 않은 상태에서 **어떤 질문을 이번 주에 다룰 질문으로 만들고, 어디까지를 이번 iteration boundary로 두며, 언제 멈추고 무엇을 archive할지**를 계속 결정해야 하기 때문이다.

open-ended research는 "마음껏 탐색"과 동의어가 아니다. 오히려 범위가 넓을수록 가설·증거·중단 규칙을 더 엄격하게 적어 두지 않으면, 프로젝트는 금방 scope creep와 retrospective story-telling에 빠진다. 이 단위의 목적은 열린 연구를 낭만적으로 포장하는 것이 아니라, **탐색적 목표를 다루되 반복 경계와 evidence standard를 잃지 않는 운영 감각**을 만드는 데 있다.

## 이번 단위에서 남길 것
- `scratch_lab.py` — toy frontier research track의 research scope, north-star question, out-of-scope, hypothesis registry, iteration boundary, kill criteria, evidence standard, evidence log를 `artifacts/scratch-manual/metrics.json`에 쓴다.
- `artifacts/scratch-manual/research_track_map.svg` — hypothesis별 result type과 decision flow를 눈으로 확인하는 작은 SVG map이다.
- `framework_lab.py` — negative result, inconclusive result, trust failure, success stop을 각각 archive / pause / escalate / stop 결정으로 바꾸는 research operations runbook을 만든다.
- `analysis.py` — 두 metrics 파일을 읽어 `artifacts/analysis-manual/latest_report.md`에 실행 관측 보고서를 쓴다. metrics가 없으면 먼저 실행할 명령을 알려 주며 실패한다.
- `analysis.md` — 실행해도 바뀌지 않는 stable interpretation 문서다.
- `reflection.md` — 실행 전 예측과 실행 후 연구 운영 회고 질문이다.

## 실행 방법
아래 명령은 모두 저장소 루트에서 실행한다.

```bash
python3 07_frontier_labs/05_open_ended_research_tracks/scratch_lab.py
python3 07_frontier_labs/05_open_ended_research_tracks/framework_lab.py
python3 07_frontier_labs/05_open_ended_research_tracks/analysis.py
```

생성되는 산출물은 다음 위치에 고정된다.

```text
07_frontier_labs/05_open_ended_research_tracks/artifacts/
├── scratch-manual/
│   ├── metrics.json
│   └── research_track_map.svg
├── framework-manual/
│   └── metrics.json
└── analysis-manual/
    └── latest_report.md
```

## 실행 결과 예시

```text
$ python3 07_frontier_labs/05_open_ended_research_tracks/scratch_lab.py
{
  "status": "runnable",
  "cpu_safe_simulation": true,
  "track_id": "frontier-open-ended-research-v1",
  "research_scope": {
    "research scope": "agentic long-horizon planning reliability under verifier warnings",
    "north-star question": "tool-using agent의 장기 계획 안정성을 어떻게 더 재현 가능하게 높일 수 있는가?",
    "out_of_scope": [
      "base model pretraining changes",
      "new benchmark schema redesign",
      "external service or live model calls"
    ]
  },
  "hypothesis_registry": {
    "type": "hypothesis registry",
    "hypotheses": [
      {
        "id": "H1",
        "iteration boundary": {"changed_variable": "planner_brief_format"},
        "kill criteria": ["rollback_drift_delta stays inside variance band twice"],
        "evidence standard": {"baseline_metric": "rollback_drift_rate"}
      }
    ]
  }
}

$ python3 07_frontier_labs/05_open_ended_research_tracks/framework_lab.py
{
  "status": "runnable",
  "framework": "cpu_deterministic_open_research_ops_sim",
  "decision_by_result_type": {
    "success stop": "stop",
    "negative result": "archive",
    "inconclusive result": "pause",
    "trust failure": "escalate"
  },
  "archive_contract": {
    "reopen condition": [
      "reopen only if production rollback incidents shift to a new failure slice"
    ]
  }
}

$ python3 07_frontier_labs/05_open_ended_research_tracks/analysis.py
# 05 Open-Ended Research Tracks 실행 관측
...
```

## 실습 흐름
1. `scratch_lab.py`의 research scope를 읽고 north-star question, this-iteration focus, out-of-scope, fixed constraints를 분리한다. 열린 질문일수록 먼저 다루지 않을 것을 적어야 한다.
2. hypothesis registry를 확인한다. 각 hypothesis가 claim만 가진 것이 아니라 mechanism guess, iteration boundary, kill criteria, evidence standard, reopen condition을 함께 가지는지 본다.
3. evidence log에서 negative result와 inconclusive result를 구분한다. 둘 다 "아쉬운 결과"처럼 보이지만, 하나는 archive해야 하고 다른 하나는 measurement를 고친 뒤 pause/resume해야 한다.
4. `framework_lab.py`의 decision rule을 읽고 success stop, negative result, inconclusive result, trust failure가 각각 stop / archive / pause / escalate로 갈라지는 이유를 확인한다.
5. `analysis.py` 보고서를 읽으며 research scope → hypothesis registry → evidence standard → decision note → reopen condition이 한 흐름으로 이어지는지 점검한다.
6. 마지막에는 generated artifacts를 지우더라도 `.gitkeep`만 남기면 unit이 다시 같은 결과를 만들 수 있는지 확인한다.

## 이 단위에서 특히 볼 질문
- open-ended research라고 해서 scope boundary를 느슨하게 잡아도 되는가, 아니면 오히려 더 엄격한 question decomposition이 필요한가?
- 좋은 hypothesis registry는 "재미있어 보이는 아이디어 목록"과 무엇이 다르고, 어떤 형태여야 실제로 keep / kill 판단이 가능한가?
- exploratory goal에서는 evidence standard를 어디까지 정량화해야 하며, 어디서부터는 qualitative observation이 꼭 필요한가?
- negative result와 inconclusive result를 왜 구분해야 하고, 둘을 같은 실패로 묶으면 무엇을 잃는가?
- stopping rule이 없을 때 생기는 반복 탐색은 왜 생산적인 집요함이 아니라 archive 불가능한 wandering이 되기 쉬운가?
- 연구를 stop / pause / escalate / archive하는 것이 실패가 아니라 좋은 운영 판단이 되려면 어떤 기록 discipline이 필요한가?
- reopen condition을 남기면 다음 사람이 같은 질문을 어떻게 더 싸고 정직하게 다시 열 수 있는가?

## 다음 단위와의 연결
이 단위는 `07_frontier_labs` 트랙의 마무리 역할을 한다. 앞의 paper reproduction, capstone build, agentic loop, benchmark construction이 각각 실험의 구성 요소를 세웠다면, 여기서는 그것들을 실제 frontier-style research 운영 규율로 묶는다. 즉 **정답이 아직 없는 질문도 scope, evidence, stop/archive discipline 위에서 다룰 수 있다**는 감각을 남기며 트랙을 닫는 단위다.

이후 새로운 track을 열더라도 이 단위의 산출물을 기준으로 질문을 다시 쪼개고, hypothesis registry를 만들고, negative result까지 보존하는 방식으로 이어 간다. 그래서 이 단위는 다음 번호의 lesson으로 직접 연결되기보다, `07_frontier_labs` 전체를 **재시작 가능한 연구 기록 체계**로 wrap-up하는 연결 고리다.
