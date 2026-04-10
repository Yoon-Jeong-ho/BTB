from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 03 Machine Reading Comprehension 분석

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
'''


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def ensure_metrics_exist() -> None:
    missing = [path for path in (SCRATCH, FRAMEWORK) if not path.exists()]
    if not missing:
        return
    missing_list = ', '.join(str(path.relative_to(UNIT_ROOT)) for path in missing)
    raise SystemExit(
        '필수 metrics 파일이 없습니다: '
        f'{missing_list}. 먼저 scratch_lab.py와 framework_lab.py를 실행하세요.'
    )


def first_partial_match(rows: list[dict[str, object]]) -> dict[str, object]:
    for row in rows:
        exact = float(row.get('exact_match', 0.0))
        f1 = float(row.get('token_f1', 0.0))
        if exact < 1.0 and f1 > 0.0:
            return row
    return rows[0] if rows else {}


def first_answerability_issue(rows: list[dict[str, object]]) -> dict[str, object]:
    for row in rows:
        if row.get('gold_answerable') != row.get('predicted_answerable'):
            return row
    return rows[0] if rows else {}


def run() -> None:
    ensure_metrics_exist()
    scratch = load_json(SCRATCH)
    framework = load_json(FRAMEWORK)

    scratch_rows = scratch.get('prediction_rows', [])
    framework_rows = framework.get('prediction_rows', [])
    scratch_focus = first_partial_match(scratch_rows if isinstance(scratch_rows, list) else [])
    framework_focus = first_answerability_issue(framework_rows if isinstance(framework_rows, list) else [])

    observed_report = f'''# 03 Machine Reading Comprehension 실행 관측

## 관측 결과
- scratch eval exact match: `{scratch.get("eval_exact_match", 0.0)}`
- scratch eval token F1: `{scratch.get("eval_token_f1", 0.0)}`
- scratch answerable accuracy: `{scratch.get("answerable_accuracy", 0.0)}`
- scratch no-answer threshold: `{scratch.get("no_answer_threshold", 0.0)}`
- framework eval exact match: `{framework.get("eval_exact_match", 0.0)}`
- framework eval token F1: `{framework.get("eval_token_f1", 0.0)}`
- framework answerable accuracy: `{framework.get("answerable_accuracy", 0.0)}`
- framework loss history head: `{framework.get("loss_history_head", [])}`

## 한국어 해석
- scratch baseline은 threshold `{scratch.get("no_answer_threshold", 0.0)}` 를 기준으로 답할지 말지를 먼저 결정했다. 이 값이 있다는 사실 자체가 MRC에서 span extraction 못지않게 abstention이 중요하다는 뜻이다.
- scratch에서 부분적으로 겹친 대표 예시는 질문 `{scratch_focus.get("question", "예문 없음")}` 이다. gold `{scratch_focus.get("gold_answers", [])}` 대비 pred `{scratch_focus.get("predicted_answer", "")}` 이고 token F1 `{scratch_focus.get("token_f1", 0.0)}` 이 남았다면, 핵심 단어는 잡았지만 boundary가 조금 흔들렸다는 뜻으로 읽을 수 있다.
- framework 모델은 질문 `{framework_focus.get("question", "예문 없음")}` 에서 gold answerable=`{framework_focus.get("gold_answerable", "-")}`, pred answerable=`{framework_focus.get("predicted_answerable", "-")}` 를 남겼다. 이 차이는 span head보다 먼저 answerability head나 threshold 해석을 다시 보게 만든다.
- scratch와 framework의 EM / token F1을 함께 보면, 완전 일치와 partial overlap이 어디서 갈리는지 더 잘 보인다. toy QA에서는 이 차이를 읽는 습관이 pretrained QA model 해석으로 그대로 이어진다.
- 결국 이 unit의 목적은 최고 점수보다 **question-context 정렬 확인 -> EM/F1 + answerable 비교 -> no-answer failure 해석** 루틴을 몸에 익히는 것이다.

## 이론 다시 연결하기
- 안정적인 해석 프레임은 `analysis.md`에 유지한다.
- 핵심 개념 복습: [THEORY.md](./THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
