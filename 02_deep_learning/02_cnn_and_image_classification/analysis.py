from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 02 CNN and Image Classification 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 CNN을 해석하는 **안정적인 프레임**만 남겨, 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- convolution은 작은 patch를 읽는 local rule이다. 따라서 feature map은 “이미지 전체 의미”보다 “어느 위치에서 어떤 detector가 켜졌는가”를 먼저 보여 준다.
- local receptive field는 출력 하나가 입력 전체가 아니라 작은 이웃만 본다는 뜻이다. 이미지에서는 이 제한이 오히려 inductive bias가 된다.
- parameter sharing은 같은 kernel이 여러 위치에서 재사용된다는 뜻이다. 같은 막대 패턴이 왼쪽/오른쪽 어디에 있어도 같은 detector가 반응할 수 있는 이유가 여기 있다.
- pooling은 중요한 반응을 남기고 해상도를 줄인다. 따라서 위치 정보 일부는 버리지만 class score baseline으로 넘어갈 때 더 압축된 표현을 만든다.
- 입력 channel 수와 출력 feature map 수는 서로 다른 개념이다. 전자는 데이터 관측 축, 후자는 detector 개수에 가깝다.

## 확인 질문
- local receptive field가 이미지 문제에서는 왜 도움이 되는가?
- parameter sharing이 translation-like robustness와 어떻게 연결되는가?
- pooling이 남기는 정보와 버리는 정보를 구분해서 설명할 수 있는가?
- 입력 channel과 feature map을 서로 다른 말로 설명할 수 있는가?
- 실행별 숫자를 왜 `analysis.md`가 아니라 `latest_report.md`에 남겨야 하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): convolution, pooling, channel/feature map, toy classification baseline을 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
'''

SCRATCH_REQUIRED_KEYS = (
    'dataset_shape',
    'input_channel_count',
    'output_feature_map_count',
    'local_receptive_field',
    'conv_kernel_shape',
    'feature_map_shape',
    'pooled_shape',
    'parameter_sharing_reuse_count',
    'pooling_reduction_ratio',
    'classification_accuracy',
    'figure_path',
)
FRAMEWORK_REQUIRED_KEYS = (
    'backend',
    'device',
    'dataset_shape',
    'input_channel_count',
    'output_feature_map_count',
    'conv_weight_shape',
    'feature_map_shape',
    'pooled_shape',
    'logits_shape',
    'class_names',
    'predictions',
    'accuracy',
)


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


def _ensure_required_keys(metrics: dict[str, object], *, name: str, required_keys: tuple[str, ...]) -> None:
    missing_keys = [key for key in required_keys if key not in metrics]
    if not missing_keys:
        return
    raise SystemExit(
        f'{name} metrics schema가 올바르지 않습니다: '
        f'{", ".join(missing_keys)} 키가 없습니다. 실험 스크립트를 다시 실행하거나 metrics 저장 로직을 확인하세요.'
    )


def run() -> None:
    _ensure_metrics_exist()
    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)
    _ensure_required_keys(scratch, name='scratch', required_keys=SCRATCH_REQUIRED_KEYS)
    _ensure_required_keys(framework, name='framework', required_keys=FRAMEWORK_REQUIRED_KEYS)

    observed_report = f'''# 02 CNN and Image Classification 실행 관측

## 관측 결과
- scratch dataset shape: `{scratch['dataset_shape']}`
- scratch local receptive field: `{scratch['local_receptive_field']}`
- scratch conv kernel shape: `{scratch['conv_kernel_shape']}`
- scratch feature map shape: `{scratch['feature_map_shape']}`
- scratch pooled shape: `{scratch['pooled_shape']}`
- scratch parameter sharing reuse count: `{scratch['parameter_sharing_reuse_count']}`
- scratch pooling reduction ratio: `{scratch['pooling_reduction_ratio']}`
- scratch classification accuracy: `{scratch['classification_accuracy']}`
- scratch figure: `{scratch['figure_path']}`
- framework backend: `{framework['backend']}`
- framework device: `{framework['device']}`
- framework conv weight shape: `{framework['conv_weight_shape']}`
- framework feature map shape: `{framework['feature_map_shape']}`
- framework pooled shape: `{framework['pooled_shape']}`
- framework logits shape: `{framework['logits_shape']}`
- framework accuracy: `{framework['accuracy']}`

## 한국어 해석
- scratch에서 local receptive field가 `{scratch['local_receptive_field']}` 라는 것은 detector 하나가 입력 전체가 아니라 **3×3 patch**만 본다는 뜻이다. CNN의 출발점은 “전체 의미”가 아니라 “작은 국소 패턴”이다.
- same kernel이 `{scratch['parameter_sharing_reuse_count']}`개 위치에 반복 적용된 것은 parameter sharing이 실제로 “왼쪽/오른쪽 어디에 있든 같은 세로·가로 규칙을 재사용한다”는 사실을 보여 준다.
- feature map이 `{scratch['feature_map_shape']}` 에서 pooling 뒤 `{scratch['pooled_shape']}` 로 줄어든 것은 해상도를 버리는 대신 강한 detector 반응을 더 압축해 class score baseline으로 넘긴다는 뜻이다.
- 입력 channel 수 `{scratch['input_channel_count']}` 와 출력 feature map 수 `{scratch['output_feature_map_count']}` 가 다르다는 점은, channel은 데이터 축이고 feature map은 detector 축이라는 구분을 다시 보여 준다.
- framework 관측에서 conv weight shape `{framework['conv_weight_shape']}` 와 logits shape `{framework['logits_shape']}` 가 모두 유지된 것은 scratch 직관이 PyTorch tensor shape로도 그대로 이어진다는 뜻이다.
- framework accuracy `{framework['accuracy']}` 는 이 toy 실험에서 pooled detector score만으로도 simple image classification baseline이 성립함을 보여 준다.

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
