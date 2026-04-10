from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 01 Language Modeling and Pretraining Objectives 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 target framing, loss-mask density, context window intuition을 읽는 **안정적인 프레임**만 남겨 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- causal LM / masked LM / span corruption의 핵심 차이는 architecture 이름보다 **무엇을 정답으로 삼는가**에 있다.
- loss-mask density는 supervision이 얼마나 촘촘한지 보여 주지만, density 하나만으로 objective 우열을 정하면 안 된다.
- 같은 context window라도 causal LM은 왼쪽 prefix만, masked LM은 mask 주변 양쪽 문맥을, span corruption은 encoder 입력과 decoder prefix를 다르게 본다.
- span corruption은 sentinel token 덕분에 “빠진 span의 시작과 끝”을 decoder target 안에서 안정적으로 bookkeeping할 수 있다.

## 확인 질문
- target framing만 바뀌어도 model behavior intuition이 달라진다고 왜 말할 수 있는가?
- loss-mask density가 높은 causal LM과 sparse한 masked LM은 각각 어떤 학습 신호를 준다고 볼 수 있는가?
- 같은 context window=4라도 objective별 visible context를 어떻게 다르게 설명할 수 있는가?
- span corruption을 단순한 MLM 확장으로 축소하면 무엇을 놓치게 되는가?

## 관련 이론
- [THEORY.md](./THEORY.md): causal LM, masked LM, span corruption, target framing, context window intuition을 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
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

    causal = scratch['objectives']['causal_lm']
    masked = scratch['objectives']['masked_lm']
    span = scratch['objectives']['span_corruption']

    observed_report = f'''# 01 Language Modeling and Pretraining Objectives 실행 관측

## 관측 결과
- scratch context window: `{scratch.get("context_window_tokens", 0)}`
- densest supervision: `{scratch.get("comparisons", {}).get("densest_supervision", "unknown")}`
- sparsest supervision: `{scratch.get("comparisons", {}).get("sparsest_supervision", "unknown")}`
- causal LM density: `{causal.get("loss_mask_density", 0.0)}`
- masked LM density: `{masked.get("loss_mask_density", 0.0)}`
- span corruption density: `{span.get("loss_mask_density", 0.0)}`
- framework device: `{framework.get("device", "unknown")}`
- framework vocab size: `{framework.get("vocab_size", 0)}`
- density ranking: `{framework.get("density_ranking", [])}`
- causal LM mean loss: `{framework.get("objectives", {}).get("causal_lm", {}).get("mean_loss", 0.0)}`
- masked LM mean loss: `{framework.get("objectives", {}).get("masked_lm", {}).get("mean_loss", 0.0)}`
- span corruption mean loss: `{framework.get("objectives", {}).get("span_corruption", {}).get("mean_loss", 0.0)}`

## 한국어 해석
- causal LM은 `{causal.get("scored_tokens", 0)}`개의 prediction slot 전체에 loss를 걸어 density가 `{causal.get("loss_mask_density", 0.0)}`이다. toy setting에서도 supervision이 가장 촘촘하다.
- masked LM은 mask 위치 `{masked.get("mask_positions", [])}`에만 loss가 걸려 density가 `{masked.get("loss_mask_density", 0.0)}`로 가장 낮다. 대신 focus token은 양쪽 문맥을 함께 본다.
- span corruption은 decoder target `{span.get("decoder_target_tokens", [])}`를 복원하므로 density가 `{span.get("loss_mask_density", 0.0)}`다. masked LM보다 촘촘하지만 causal LM처럼 모든 원문 시점을 직접 score하지는 않는다.
- scratch 비교에서 `densest_supervision={scratch.get("comparisons", {}).get("densest_supervision", "unknown")}` 와 `sparsest_supervision={scratch.get("comparisons", {}).get("sparsest_supervision", "unknown")}` 가 유지되어, target framing 차이가 loss-mask density 차이로 바로 연결됨을 볼 수 있다.
- framework 실험의 density ranking `{framework.get("density_ranking", [])}` 도 같은 순서를 재현했다. 즉 toy tensor 수준에서도 causal LM / masked LM / span corruption의 supervision 구조 차이가 안정적으로 관찰된다.
- context window note: `{scratch.get("comparisons", {}).get("context_window_note", "")}`

## 이론 다시 연결하기
- stable 프레임은 `analysis.md`에 유지한다.
- 핵심 개념 복습: [THEORY.md](./THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
