# Scripts

이 폴더는 학습/평가/검증 스크립트를 모으는 공간이다.

```text
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode scratch
python scripts/run_lesson.py --unit 00_foundations/05_gpu_memory_runtime --mode framework
python scripts/build_lesson_report.py --unit 00_foundations/01_tensor_shapes
python scripts/check_curriculum_links.py
python scripts/train.py --track 03_nlp --stage 01_text_classification --config path/to/config.yaml
python scripts/eval.py --run-dir runs/03_nlp/01_text_classification/<run_id>
python scripts/eval.py --run-dir runs/05_multimodal/01_image_text_retrieval/<run_id>
```

## Task 6 automation scaffold

- `run_lesson.py`: `lesson.yaml`을 읽고 `scratch_lab.py` 또는 `framework_lab.py`를 실행한다.
- `build_lesson_report.py`: unit의 `artifacts/summary.md` 스캐폴드를 만든다.
- `check_curriculum_links.py`: 루트 README, docs, foundations/bridge/track 문서의 로컬 markdown 링크를 점검한다.
- `_lesson_metadata.py`: 현재 BTB의 제한된 `lesson.yaml` 스키마(top-level scalar/list)만 파싱하는 무의존성 로더다.
