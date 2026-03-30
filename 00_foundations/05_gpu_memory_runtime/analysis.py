from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 05 GPU Memory Runtime 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라지는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 숫자가 바뀌어도 유지되는 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- 같은 shape라도 dtype(fp32 / fp16 / bf16)가 바뀌면 메모리 budget이 바로 달라진다.
- training은 activation 보관, gradient 계산, 추가 상태 때문에 inference보다 더 무거워지는 방향으로 읽는다.
- CUDA가 없더라도 output/gradient 바이트와 runtime 차이를 proxy로 보면 training cost를 설명할 수 있다.

## 확인 질문
- batch size를 키우면 어떤 항목이 함께 증가하는가?
- training과 inference를 비교할 때 parameter만 보면 왜 부족한가?
- 이번 실행에서 관측한 구체적 숫자는 `artifacts/analysis-manual/latest_report.md`에 어떻게 정리되었는가?
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

    training_runtime = float(framework['training_runtime_ms'])
    inference_runtime = float(framework['inference_runtime_ms'])
    runtime_gap = round(training_runtime - inference_runtime, 4)
    savings_ratio = float(scratch['dtype_savings_ratio'])
    device = str(framework['device'])

    if runtime_gap >= 0:
        runtime_interpretation = (
            f'- training runtime은 inference보다 `{runtime_gap}` ms 더 걸렸다. '
            'backward와 gradient 계산이 추가되기 때문이다.'
        )
    else:
        runtime_interpretation = (
            f'- 이번 관측에서는 training runtime이 inference보다 `{abs(runtime_gap)}` ms 더 짧게 나왔다. '
            '단일 측정의 노이즈나 장치 스케줄링 차이일 수 있으므로, backward 비용은 메모리 지표와 함께 해석해야 한다.'
        )

    if device == 'cuda':
        memory_observation = (
            f"- CUDA peak allocated는 inference `{framework['inference_max_memory_allocated']}` bytes, "
            f"training `{framework['training_max_memory_allocated']}` bytes로 기록됐다.\n"
            f"- CUDA peak reserved는 training 시 `{framework['training_max_memory_reserved']}` bytes였다."
        )
    else:
        memory_observation = (
            '- CPU 환경이라 CUDA allocator 수치는 0으로 남는다. 대신 output/gradient 바이트와 runtime 차이를 proxy로 본다.\n'
            f"- gradient bytes는 `{framework['training_grad_bytes']}` bytes로 측정되어 training에서 추가 상태가 생김을 보여준다."
        )

    observed_report = f'''# 05 GPU Memory Runtime 실행 관측

## 관측 결과
- scratch batch fp32 bytes: `{scratch["batch_fp32_bytes"]}`
- scratch batch fp16 bytes: `{scratch["batch_fp16_bytes"]}`
- scratch dtype savings ratio: `{scratch["dtype_savings_ratio"]}`
- framework device: `{framework["device"]}`
- framework dtype: `{framework["dtype"]}`
- inference runtime: `{framework["inference_runtime_ms"]} ms`
- training runtime: `{framework["training_runtime_ms"]} ms`
- training grad bytes: `{framework["training_grad_bytes"]}`

## 한국어 해석
- 같은 `(32, 512, 768)` shape라도 fp16은 fp32보다 약 `{savings_ratio}`배 적은 메모리를 요구했다. 즉 dtype 선택만으로도 batch budget이 즉시 달라진다.
{runtime_interpretation}
{memory_observation}
- parameter만 세는 습관으로는 실제 runtime 비용을 설명할 수 없다. activation과 gradient를 함께 봐야 batch size와 OOM 원인을 읽을 수 있다.
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
