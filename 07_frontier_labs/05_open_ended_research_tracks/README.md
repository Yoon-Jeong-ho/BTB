# 05 Open-Ended Research Tracks

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
앞 단위들에서 우리는 논문 재현, capstone build, agentic loop, benchmark/dataset contract까지 차례로 고정했다. 그런데 frontier 연구는 그다음부터가 더 어렵다. benchmark를 세웠다고 해서 다음 행동이 자동으로 주어지는 것은 아니고, 정답이 분명하지 않은 상태에서 **어떤 질문을 이번 주에 다룰 질문으로 만들고, 어디까지를 이번 iteration의 약속으로 두며, 언제 멈추고 무엇을 archive할지**를 계속 결정해야 하기 때문이다.

또한 open-ended research는 "마음껏 탐색"과 동의어가 아니다. 오히려 범위가 넓을수록 가설·증거·중단 규칙을 더 엄격하게 적어 두지 않으면, 프로젝트는 금방 scope creep와 retrospective story-telling에 빠진다. 이 단위의 목적은 열린 연구를 낭만적으로 포장하는 것이 아니라, **탐색적 목표를 다루되 반복 경계와 evidence standard를 잃지 않는 운영 감각**을 만드는 데 있다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- research scoping, hypothesis framing, stopping rule 운영을 정리한 `THEORY.md`
- 선행 개념과 자기 점검 질문을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - research track charter와 question decomposition 메모
  - hypothesis registry, iteration boundary, kill criterion 표
  - evidence log와 negative result / inconclusive result 분류 기록
  - stop / pause / escalate / archive decision note

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable/applied 승격 때 구현할 실습 순서다.
1. 먼저 benchmark contract와 capstone 목표를 다시 읽고, "흥미로운 질문"을 바로 실험으로 옮기지 않고 **이번 트랙에서 실제로 좁힐 research scope**로 바꾼다. 어떤 capability를 보려는지, 어떤 failure slice를 건드리는지, 이번 트랙에서 건드리지 않을 것은 무엇인지 먼저 적는다.
2. research question을 작게 쪼갠다. 메인 질문, 보조 질문, falsifiable hypothesis, 그리고 "이번 iteration에서 확인하지 않을 하위 질문"을 분리해 둔다. open-ended research에서도 질문을 쪼개지 않으면 진행이 아니라 메모 누적만 일어나기 쉽다.
3. 각 hypothesis마다 iteration boundary를 정한다. 바꿀 변수, 고정할 benchmark/protocol, 허용할 retry 횟수, 관찰하려는 evidence field, kill criterion을 함께 적어 **탐색 범위가 어디서 끝나는지**를 먼저 못 박는다.
4. evidence standard를 세운다. exploratory goal이라고 해서 느낌 좋은 qualitative 사례만 모으지 않고, baseline 대비 변화, slice별 signal, 실패 사례, negative result, inconclusive result를 같은 로그 체계 안에 남기도록 정한다.
5. stopping rule을 적용한다. 개선 신호가 variance band 안에 머무르거나, benchmark 신뢰도 자체가 흔들리거나, hypothesis가 더 이상 작게 쪼개지지 않으면 계속 밀지 말고 pause / archive / escalate 중 하나를 고른다.
6. 마지막에는 track archive를 정리한다. 살아남은 hypothesis, 죽인 hypothesis, 보류한 질문, 재현 가능한 artifact, 다음 분기에 다시 열 조건을 문장으로 남겨 `07_frontier_labs` 전체를 **열린 탐색이지만 다시 시작 가능한 연구 기록**으로 마무리한다.

## 이 단위에서 특히 볼 질문
- open-ended research라고 해서 scope boundary를 느슨하게 잡아도 되는가, 아니면 오히려 더 엄격한 question decomposition이 필요한가?
- 좋은 hypothesis는 "재미있어 보이는 아이디어"와 무엇이 다르고, 어떤 형태여야 실제로 kill / keep 판단이 가능한가?
- exploratory goal에서는 evidence standard를 어디까지 정량화해야 하며, 어디서부터는 qualitative observation이 꼭 필요한가?
- negative result와 inconclusive result를 왜 구분해야 하고, 둘을 같은 실패로 묶으면 무엇을 잃는가?
- stopping rule이 없을 때 생기는 반복 탐색은 왜 생산적인 집요함이 아니라 archive 불가능한 wandering이 되기 쉬운가?
- 연구를 중단하거나 보류하는 것이 실패가 아니라 좋은 운영 판단이 되려면 어떤 기록 discipline이 필요한가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 07_frontier_labs/05_open_ended_research_tracks/scratch_lab.py
{
  "status": "sample",
  "track_id": "frontier-open-ended-v1",
  "research_scope": {
    "north_star_question": "tool-using agent의 장기 계획 안정성을 어떻게 올릴 것인가",
    "this_track_focus": [
      "long-horizon planning failure slice",
      "rollback-after-verifier-warning behavior"
    ],
    "out_of_scope": [
      "base model pretraining changes",
      "benchmark schema redesign"
    ]
  },
  "hypotheses": [
    {
      "id": "H1",
      "claim": "critic memory를 짧은 error taxonomy로 제한하면 retry drift가 줄어든다",
      "boundary": ["same benchmark", "same tool schema", "max 3 retries"],
      "kill_if": ["verifier mismatch", "delta within variance band twice"]
    },
    {
      "id": "H2",
      "claim": "planner brief를 shorter-but-stricter format으로 바꾸면 execution variance가 줄어든다",
      "boundary": ["same dataset version", "same critic prompt family"]
    }
  ],
  "evidence_contract": {
    "required_fields": [
      "baseline_comparison",
      "failure_slice_notes",
      "negative_result_log",
      "inconclusive_reason"
    ],
    "qualitative_examples_min": 3,
    "archive_every_iteration": true
  }
}

$ python 07_frontier_labs/05_open_ended_research_tracks/analysis.py
{
  "status": "sample",
  "iteration_summary": {
    "completed": 4,
    "kept": ["H1"],
    "killed": ["H2"],
    "paused": ["H3"]
  },
  "stop_decisions": [
    {
      "hypothesis_id": "H2",
      "decision": "archive",
      "reason": "effect size stayed inside variance band across repeated runs"
    },
    {
      "hypothesis_id": "H3",
      "decision": "escalate",
      "reason": "benchmark drift suspected; requires dataset contract review"
    }
  ],
  "archive_note": {
    "reopen_when": [
      "new private holdout available",
      "planner/executor prompt family redesigned"
    ],
    "negative_results_preserved": true
  }
}
```

핵심은 멋진 breakthrough story를 만드는 것이 아니라, **어떤 질문을 이번 트랙에서 실제로 다뤘는지**, **각 hypothesis를 어떤 경계 안에서 검증했는지**, **무엇을 근거로 keep/kill/pause를 결정했는지**, **다음 사람이 다시 열 수 있을 만큼 archive가 정직한지**를 읽을 수 있어야 한다는 점이다.

## 다음 단위와의 연결
이 단위는 `07_frontier_labs` 트랙의 마무리 역할을 한다. 앞의 paper reproduction, capstone build, agentic loop, benchmark construction이 각각 실험의 구성 요소를 세웠다면, 여기서는 그것들을 실제 frontier-style research 운영 규율로 묶는다. 즉 **정답이 아직 없는 질문도 scope, evidence, stop/archive discipline 위에서 다룰 수 있다**는 감각을 남기며 트랙을 닫는 단위다.

이후에는 새로운 track을 열더라도 이 단위의 산출물을 기준으로 질문을 다시 쪼개고, hypothesis registry를 만들고, negative result까지 보존하는 방식으로 이어 가게 된다. 그래서 이 단위는 다음 번호의 lesson으로 직접 연결되기보다, `07_frontier_labs` 전체를 **재시작 가능한 연구 기록 체계**로 wrap-up하는 연결 고리다.
