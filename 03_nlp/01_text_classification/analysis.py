from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 01 Text Classification 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 bag-of-words baseline과 tiny neural classifier를 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- text classification의 첫 질문은 "무슨 거대한 모델을 쓸까"보다 먼저, 문장을 어떤 token 단위와 어떤 feature 표현으로 읽을지 정하는 것이다.
- bag-of-words baseline이 잘 맞힌다면, dataset 안에 강한 lexical cue가 있다는 뜻이다. 이 신호는 해석 가능하고 빠르게 확인할 수 있다.
- tiny neural classifier는 dense embedding 평균으로 문장 표현을 만든다. 따라서 같은 label이라도 표면 token이 조금 달라진 예문에 더 유연해질 여지가 있다.
- accuracy는 전체 정답률이고, macro F1은 각 클래스를 동등 비중으로 본다. 둘을 같이 읽어야 특정 클래스 collapse를 놓치지 않는다.
- 오분류 문장을 읽을 때는 모델이 틀렸다는 사실만 보지 말고, 어떤 token cue를 과신했는지 또는 어떤 표현을 vocabulary 밖으로 흘려 보냈는지를 함께 봐야 한다.

## 확인 질문
- baseline의 top token signal은 무엇이며, 그것이 왜 분류 근거가 되는가?
- neural classifier가 baseline보다 좋아 보인다면 그것은 어떤 representation 차이를 시사하는가?
- accuracy와 macro F1을 함께 읽을 때 어떤 failure pattern이 더 잘 드러나는가?

## 관련 이론
- [THEORY.md](./THEORY.md): bag-of-words, tiny neural classifier, accuracy, macro F1 핵심 개념을 다시 확인한다.
'''


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def _ensure_metrics_exist() -> None:
    missing = [path for path in (SCRATCH, FRAMEWORK) if not path.exists()]
    if not missing:
        return

    missing_list = ', '.join(str(path.relative_to(UNIT_ROOT)) for path in missing)
    raise SystemExit(
        '필수 metrics 파일이 없습니다: '
        f'{missing_list}. 먼저 scratch_lab.py와 framework_lab.py를 실행하세요.'
    )


def run() -> None:
    _ensure_metrics_exist()
    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)

    scratch_rows = scratch.get('prediction_rows', [])
    framework_rows = framework.get('prediction_rows', [])
    first_scratch = scratch_rows[0] if isinstance(scratch_rows, list) and scratch_rows else {}
    first_framework = framework_rows[0] if isinstance(framework_rows, list) and framework_rows else {}

    observed_report = f'''# 01 Text Classification 실행 관측

## 관측 결과
- scratch eval accuracy: `{scratch.get("eval_accuracy", 0.0)}`
- scratch eval macro F1: `{scratch.get("eval_macro_f1", 0.0)}`
- scratch top positive tokens: `{scratch.get("top_positive_tokens", [])}`
- scratch top negative tokens: `{scratch.get("top_negative_tokens", [])}`
- framework eval accuracy: `{framework.get("eval_accuracy", 0.0)}`
- framework eval macro F1: `{framework.get("eval_macro_f1", 0.0)}`
- framework vocab size: `{framework.get("vocab_size", 0)}`
- framework loss history head: `{framework.get("loss_history_head", [])}`

## 한국어 해석
- scratch baseline은 `{scratch.get("top_positive_tokens", [])}` 와 `{scratch.get("top_negative_tokens", [])}` 같은 token cue를 중심으로 문장을 읽었다. 즉 이 toy dataset에서는 표면 lexical signal만으로도 어느 정도 분리가 된다.
- 첫 scratch 예문 `{first_scratch.get("text", "예문 없음")}` 에서 gold=`{first_scratch.get("gold", "-")}`, pred=`{first_scratch.get("predicted", "-")}` 로 나온 것은 baseline이 token count 합을 어떻게 의사결정 근거로 쓰는지 보여 준다.
- tiny PyTorch classifier의 첫 예문 `{first_framework.get("text", "예문 없음")}` 는 gold=`{first_framework.get("gold", "-")}`, pred=`{first_framework.get("predicted", "-")}` 였다. 확률 분포 `{first_framework.get("probabilities", {})}` 를 함께 보면, neural model이 단순 label만이 아니라 class confidence도 출력한다는 사실을 볼 수 있다.
- scratch와 framework의 accuracy / macro F1을 함께 비교하면, 전체 정답률만 볼 때 놓치기 쉬운 클래스별 균형 감각을 다시 확인할 수 있다.
- 이 toy unit에서는 두 모델 모두 매우 작기 때문에 "최고 성능"보다 **baseline을 세우고, feature 표현 차이를 해석하는 출발점**을 만드는 것이 더 중요하다.

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
