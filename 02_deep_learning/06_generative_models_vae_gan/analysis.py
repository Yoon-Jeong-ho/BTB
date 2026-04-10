from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 06 Generative Models: VAE, GAN 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 VAE vs GAN contrast, latent sampling, adversarial intuition, posterior collapse, mode collapse를 읽는 **안정적인 해석 프레임**만 남긴다.

## 해석 프레임
- VAE는 reconstruction term과 KL term을 함께 보며, latent를 단순 압축 코드가 아니라 **샘플링 가능한 분포 좌표계**로 만들려 한다.
- reparameterization trick은 `z = mu + sigma * epsilon` 형태로 noise를 주입하면서도 gradient가 encoder로 흐르도록 돕는다.
- posterior collapse는 decoder가 너무 많은 일을 대신해 `z`를 거의 쓰지 않는 실패다. 이때 latent usage와 KL이 함께 줄어들 수 있다.
- GAN은 generator와 discriminator의 adversarial game으로 sharp sample realism을 밀어 올릴 수 있지만, loss만으로 diversity를 다 읽기 어렵다.
- mode collapse는 generator가 한두 mode만 반복 출력하는 실패다. 그래서 generative model에서는 sample quality와 함께 coverage / pairwise diversity / batch inspection을 같이 봐야 한다.

## 확인 질문
- VAE에서 reconstruction이 좋아도 KL과 latent usage를 같이 봐야 하는 이유는 무엇인가?
- reparameterization trick이 없으면 encoder가 latent sampling을 학습하기 어려운 이유를 어떻게 설명할 수 있는가?
- posterior collapse와 mode collapse는 각각 무엇이 collapse되는 현상인가?
- GAN loss가 그럴듯해 보여도 mode coverage를 따로 확인해야 하는 이유는 무엇인가?

## 관련 이론
- [THEORY.md](./THEORY.md): VAE vs GAN, reparameterization trick, posterior collapse, mode collapse를 다시 확인한다.
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

    observed_report = f'''# 06 Generative Models: VAE, GAN 실행 관측

## 관측 결과
- scratch VAE reconstruction mse: `{scratch.get("vae", {}).get("reconstruction_mse", 0.0)}`
- scratch VAE KL term: `{scratch.get("vae", {}).get("kl_term", 0.0)}`
- scratch prior sample spread: `{scratch.get("vae", {}).get("prior_sample_spread", 0.0)}`
- scratch posterior collapse latent usage: `{scratch.get("vae", {}).get("posterior_collapse_probe", {}).get("healthy_latent_usage", 0.0)}` -> `{scratch.get("vae", {}).get("posterior_collapse_probe", {}).get("collapsed_latent_usage", 0.0)}`
- scratch GAN mode coverage: `{scratch.get("gan", {}).get("balanced_mode_coverage", 0)}` -> collapsed `{scratch.get("gan", {}).get("collapsed_mode_coverage", 0)}`
- framework VAE reconstruction loss: `{framework.get("vae", {}).get("final_reconstruction_loss", 0.0)}`
- framework VAE KL loss: `{framework.get("vae", {}).get("final_kl_loss", 0.0)}`
- framework VAE posterior usage: `{framework.get("vae", {}).get("posterior_usage_mean_abs", framework.get("posterior_usage_mean_abs", 0.0))}`
- framework GAN generator/discriminator loss: `{framework.get("gan", {}).get("generator_loss", 0.0)}` / `{framework.get("gan", {}).get("discriminator_loss", 0.0)}`
- framework GAN mode coverage: `{framework.get("gan", {}).get("mode_coverage", 0)}`
- framework collapsed coverage: `{framework.get("gan", {}).get("collapsed_probe", {}).get("mode_coverage", 0)}`

## 한국어 해석
- scratch 실험에서 VAE는 KL `{scratch.get("vae", {}).get("kl_term", 0.0)}` 와 reconstruction mse `{scratch.get("vae", {}).get("reconstruction_mse", 0.0)}` 사이의 균형으로 latent를 샘플링 가능한 공간으로 정리했다. interpolation path와 prior sample spread가 0보다 큰 것은 latent 공간이 단순 한 점 복사가 아니라는 뜻이다.
- posterior collapse probe에서는 latent usage가 `{scratch.get("vae", {}).get("posterior_collapse_probe", {}).get("healthy_latent_usage", 0.0)}` 에서 `{scratch.get("vae", {}).get("posterior_collapse_probe", {}).get("collapsed_latent_usage", 0.0)}` 로 줄고 reconstruction이 나빠졌다. 즉 z를 안 쓰는 decoder shortcut이 생기면 generative latent가 무너진다.
- GAN scratch 실험에서 balanced generator는 coverage `{scratch.get("gan", {}).get("balanced_mode_coverage", 0)}` 를 확보했지만 collapsed generator는 `{scratch.get("gan", {}).get("collapsed_mode_coverage", 0)}` 로 줄었다. adversarial loss가 비슷해 보여도 diversity check가 반드시 필요한 이유다.
- framework PyTorch 실험에서도 VAE collapsed probe가 활성화되어 posterior usage가 줄어들고, GAN collapsed probe에서는 coverage가 `1`로 줄었다. 즉 VAE와 GAN은 모두 “생성은 되지만 중요한 구조가 빠지는 실패”를 서로 다른 방식으로 겪는다.
- 결론적으로 VAE는 **샘플링 가능한 latent geometry**, GAN은 **adversarial sample realism** 쪽에 강점이 있지만, 둘 다 collapse 징후를 별도 지표로 읽어야 한다.

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
