# Scripts

이 폴더는 향후 학습/평가 스크립트를 넣기 위한 자리다.

권장 인터페이스는 아래와 같다.

```text
python scripts/train.py --track 02_nlp --stage 01_text_classification --config path/to/config.yaml
python scripts/eval.py --run-dir runs/02_nlp/01_text_classification/<run_id>
python scripts/promote.py --run-dir runs/... --report-dir reports/...
```

아직 실제 스크립트는 만들지 않았고, 지금 단계에서는 문서와 폴더 구조를 먼저 고정한다.
