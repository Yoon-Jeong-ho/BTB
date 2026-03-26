# Reports

이 폴더에는 Git으로 공유할 가치가 있는 실험 결과만 둔다.

권장 구조는 아래와 같다.

```text
reports/<track>/<stage>/<run_id>/
├── summary.md
├── metrics.json
└── figures/
    ├── results/
    └── analysis/
```

## 무엇을 올릴까

- 다시 봐야 하는 대표 figure
- 비교에 필요한 핵심 metrics
- 다음 실험 결정을 돕는 요약

## 무엇을 올리지 말까

- 원시 로그 전체
- 중간 checkpoint 다수
- 재생성 가능한 임시 파일
