from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 02 Named Entity Recognition 분석

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


def first_boundary_mismatch(rows: list[dict[str, object]]) -> dict[str, object]:
    for row in rows:
        if row.get('gold_piece_tags') != row.get('predicted_piece_tags'):
            return row
    return rows[0] if rows else {}


def run() -> None:
    ensure_metrics_exist()
    scratch = load_json(SCRATCH)
    framework = load_json(FRAMEWORK)

    scratch_rows = scratch.get('prediction_rows', [])
    framework_rows = framework.get('prediction_rows', [])
    scratch_focus = first_boundary_mismatch(scratch_rows if isinstance(scratch_rows, list) else [])
    framework_focus = first_boundary_mismatch(framework_rows if isinstance(framework_rows, list) else [])

    observed_report = f'''# 02 Named Entity Recognition 실행 관측

## 관측 결과
- scratch token accuracy: `{scratch.get("token_accuracy", 0.0)}`
- scratch entity precision / recall / F1: `{scratch.get("entity_precision", 0.0)}` / `{scratch.get("entity_recall", 0.0)}` / `{scratch.get("entity_f1", 0.0)}`
- scratch label counts: `{scratch.get("label_counts", {})}`
- framework token accuracy: `{framework.get("token_accuracy", 0.0)}`
- framework entity precision / recall / F1: `{framework.get("entity_precision", 0.0)}` / `{framework.get("entity_recall", 0.0)}` / `{framework.get("entity_f1", 0.0)}`
- framework loss history head: `{framework.get("loss_history_head", [])}`

## 한국어 해석
- scratch baseline은 piece lookup 중심이라 `{scratch_focus.get("pieces", [])}` 같은 시퀀스에서 gold `{scratch_focus.get("gold_piece_tags", [])}` 대비 pred `{scratch_focus.get("predicted_piece_tags", [])}` 처럼 boundary를 놓칠 수 있다. 이 차이는 token lookup만으로는 unseen 조합을 복원하기 어렵다는 뜻이다.
- scratch의 entity F1 `{scratch.get("entity_f1", 0.0)}` 와 token accuracy `{scratch.get("token_accuracy", 0.0)}` 를 함께 보면, token 몇 개를 맞혔다고 해서 entity span 전체를 제대로 복원한 것은 아니라는 점을 확인할 수 있다.
- framework 모델은 `{framework_focus.get("pieces", [])}` 시퀀스에서 gold `{framework_focus.get("gold_piece_tags", [])}` 와 pred `{framework_focus.get("predicted_piece_tags", [])}` 를 비교하게 해 준다. 여기서 앞뒤 문맥을 읽어 boundary를 얼마나 안정적으로 잡는지가 핵심이다.
- framework의 entity F1 `{framework.get("entity_f1", 0.0)}` 이 scratch보다 높다면, tiny biGRU가 최소한 span 안팎 문맥을 조금 더 읽었을 가능성이 있다. 반대로 비슷하다면 toy dataset이 너무 작거나 vocabulary overlap이 대부분의 신호를 이미 설명한 것일 수 있다.
- 결국 이 unit의 목적은 최고 성능이 아니라 **alignment sanity check -> token/entity metric 분리 -> boundary error 해석** 루틴을 몸에 익히는 것이다.

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
