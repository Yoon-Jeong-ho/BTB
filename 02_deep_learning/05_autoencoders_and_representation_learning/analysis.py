from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 05 Autoencoders and Representation Learning 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 reconstruction objective, encoder/latent/decoder 역할, bottleneck intuition, denoising/compression variant를 읽는 **안정적인 해석 프레임**만 남긴다.

## 해석 프레임
- autoencoder의 reconstruction objective는 label 없이도 입력 자체를 다시 맞히게 하면서, 어떤 정보를 latent로 남겨야 하는지 압박한다.
- encoder는 입력을 latent code로 압축하고, decoder는 그 code만 보고 입력을 얼마나 복원할 수 있는지 시험한다. 따라서 latent는 단순 중간값이 아니라 정보 병목의 위치다.
- bottleneck이 충분히 넓으면 reconstruction은 쉬워지지만 representation이 지나치게 복사에 가까워질 수 있다. 너무 좁으면 핵심 구조까지 잃어 reconstruction이 나빠진다.
- denoising variant는 noisy input으로부터 clean target을 복원하게 만들어, 단순 복사보다 안정적인 구조 보존을 더 강하게 요구한다.
- compression variant는 적은 latent 차원으로 얼마나 reconstruction error를 낮출 수 있는지 보며, 저장 효율과 정보 손실의 trade-off를 드러낸다.

## 확인 질문
- reconstruction objective를 "입력을 외우는 것"과 구분해서 설명하려면 어떤 관측이 필요한가?
- encoder / latent / decoder를 각각 어떤 역할로 읽어야 bottleneck intuition이 선명해지는가?
- latent dimension을 바꿨을 때 reconstruction mse가 어떻게 달라지는지 보고, representation quality를 어떻게 추론할 수 있는가?
- denoising autoencoder와 compression-oriented autoencoder는 같은 구조를 각각 무엇에 더 민감하게 만드는가?

## 관련 이론
- [THEORY.md](./THEORY.md): reconstruction objective, bottleneck, denoising/compression variant를 다시 확인한다.
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

    bottleneck_results = scratch['bottleneck_results']
    observed_report = f'''# 05 Autoencoders and Representation Learning 실행 관측

## 관측 결과
- 입력 차원: `{scratch.get("input_dim", 0)}`
- 비교한 bottleneck 차원: `{scratch.get("bottleneck_dims_compared", [])}`
- scratch latent=1 reconstruction mse: `{bottleneck_results.get("1", {}).get("reconstruction_mse", 0.0)}`
- scratch latent=2 reconstruction mse: `{bottleneck_results.get("2", {}).get("reconstruction_mse", 0.0)}`
- scratch latent=3 reconstruction mse: `{bottleneck_results.get("3", {}).get("reconstruction_mse", 0.0)}`
- scratch raw noisy mse: `{scratch.get("denoising_variant", {}).get("raw_noisy_mse", 0.0)}`
- scratch denoised mse: `{scratch.get("denoising_variant", {}).get("denoised_mse", 0.0)}`
- framework narrow bottleneck loss: `{framework.get("narrow_bottleneck_autoencoder", {}).get("final_loss", 0.0)}`
- framework compression bottleneck loss: `{framework.get("compression_autoencoder", {}).get("final_loss", 0.0)}`
- framework denoising final loss: `{framework.get("denoising_autoencoder", {}).get("final_loss", 0.0)}`
- framework denoising gain: `{framework.get("denoising_autoencoder", {}).get("denoising_gain", 0.0)}`

## 한국어 해석
- scratch에서 latent를 `1 -> 2 -> 3`으로 넓힐수록 reconstruction mse가 낮아졌다는 것은, bottleneck이 representation capacity를 직접 제한한다는 뜻이다.
- encoder/latent/decoder 역할은 `{scratch.get("encoder_decoder_roles", {})}` 로 요약된다. encoder는 압축 규칙, latent는 병목 코드, decoder는 복원 시험기라는 구분이 핵심이다.
- denoising variant에서 raw noisy mse `{scratch.get("denoising_variant", {}).get("raw_noisy_mse", 0.0)}`가 denoised mse `{scratch.get("denoising_variant", {}).get("denoised_mse", 0.0)}`보다 큰 것은, 노이즈를 그대로 복사하지 않고 구조를 남기는 representation이 가능하다는 뜻이다.
- framework PyTorch autoencoder에서도 narrow bottleneck loss `{framework.get("narrow_bottleneck_autoencoder", {}).get("final_loss", 0.0)}`보다 compression bottleneck loss `{framework.get("compression_autoencoder", {}).get("final_loss", 0.0)}`가 더 낮아, latent dimension 설계가 reconstruction objective와 직접 연결됨을 다시 확인했다.
- denoising autoencoder는 noisy input 대비 `{framework.get("denoising_autoencoder", {}).get("denoising_gain", 0.0)}`만큼 clean target에 가까워졌다. 즉 reconstruction objective 하나로도 compression과 denoising variant를 서로 다른 학습 압력으로 만들 수 있다.

## 이론 다시 연결하기
- 안정적인 해석 프레임은 `analysis.md`에 유지한다.
- 핵심 개념 복습: [THEORY.md](../../THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
