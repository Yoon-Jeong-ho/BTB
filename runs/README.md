# Runs

이 폴더는 로컬 또는 서버에서 생성되는 비정제 실험 산출물 위치다.

기본적으로 Git에서 제외한다. 권장 구조는 아래와 같다.

```text
runs/<track>/<stage>/<run_id>/
├── config.yaml
├── metrics.json
├── summary.md
├── logs/
├── figures/
│   ├── results/
│   └── analysis/
├── predictions/
└── checkpoints/
```

검토할 가치가 있는 결과만 `reports/` 또는 `artifacts/promoted/` 로 승격한다.
