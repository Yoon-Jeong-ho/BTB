# 04 Benchmark and Dataset Construction 회고

## 실행 후 바로 적어 보기
1. 이번 toy benchmark의 **task contract**는 무엇을 한 사례(unit of record)로 보고, 어떤 claim boundary를 허용하는가?
2. `dataset schema`에서 `record_id`, `source_id`, `split`, `slice_tags`, `license_tier`, `annotation`이 빠지면 어떤 재현성 문제가 생기는가?
3. `source/split manifest`가 random split보다 더 강하게 source와 template family를 분리하는 이유를 leakage 관점에서 설명해 보자.
4. `annotation rubric`의 task_success, groundedness, policy_compliance를 하나의 점수로 합치면 어떤 QC 신호가 사라지는가?
5. leakage, contamination, drift audit 중 이번 실행에서 headline score 앞에 반드시 붙여야 하는 warning은 무엇인가?

## 조금 더 깊게 생각하기
- benchmark card에 known non-goals를 적는 것은 왜 소극적 문서화가 아니라 과장 방지 장치인가?
- annotation disagreement를 noise로만 처리하지 않고 major disagreement와 adjudication rule로 남기면 어떤 연구 질문이 더 선명해지는가?
- versioning 정책에서 frozen core와 refresh slice를 분리하지 않으면 open-ended research track의 비교 가능성은 어떻게 흔들리는가?
- report template에 contamination audit와 known limits를 고정해 두면 agentic loop의 verifier가 어떤 근거로 stop/escalate 판단을 할 수 있는가?

## 다음 단위로 넘길 메모
- 다음 `05_open_ended_research_tracks`로 넘어가기 전, 이 benchmark가 실제로 측정하지 못하는 non-goal을 한 문장으로 적어 둔다.
- 새 연구 아이디어가 benchmark 자체를 바꾸는 제안인지, frozen benchmark 위에서 모델/시스템을 바꾸는 제안인지 구분해 본다.
