# Scripts

이 폴더는 학습/평가/검증 스크립트를 모으는 공간이다. 예시 경로도 현재 `00→10` 커리큘럼 루트를 기준으로 적는다.

```text
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode scratch
python scripts/run_lesson.py --unit 00_foundations/05_gpu_memory_runtime --mode framework
python scripts/build_lesson_report.py --unit 00_foundations/01_tensor_shapes
python scripts/check_curriculum_links.py
python scripts/train.py --track 04_nlp --stage 01_text_classification --config path/to/config.yaml
python scripts/eval.py --run-dir runs/04_nlp/01_text_classification/<run_id>
python scripts/eval.py --run-dir runs/09_multimodal/01_image_text_retrieval/<run_id>
python scripts/run_lesson.py --unit 10_vla/01_vision_language_action_grounding --mode scratch
```

현재 커리큘럼 기준으로 NLP 브리지는 `03_nlp_bridge`, applied NLP는 `04_nlp`, multimodal 브리지는 `08_multimodal_bridge`, multimodal 실습은 `09_multimodal`, VLA 입구는 `10_vla` 아래에 놓인다.

새 scaffold track/unit는 아직 `planned` 상태일 수 있으므로, 실행 전에는 먼저 `docs/curriculum_status.json`에서 runnable 여부를 확인하고, 각 track README에 status note/table이 있으면 보조 맥락으로 함께 참고하는 것을 권장한다.

## Task 6 automation scaffold

- `run_lesson.py`: `lesson.yaml`을 읽고 `scratch_lab.py` 또는 `framework_lab.py`를 실행한다.
- `build_lesson_report.py`: unit의 `artifacts/summary.md` 스캐폴드를 만든다.
- `check_curriculum_links.py`: 루트 README, docs, foundations/bridge/track/VLA 문서의 로컬 markdown 링크를 점검한다.
- `build_web_catalog.py`: manifest와 lesson metadata를 정적 웹사이트용 `web/catalog.json`으로 변환한다.
- `_lesson_metadata.py`: 현재 BTB의 제한된 `lesson.yaml` 스키마(top-level scalar/list)만 파싱하는 무의존성 로더다.

- `playwright_site_qa.js`: Playwright로 정적 웹사이트의 데스크톱/모바일 reader, 코드 설명, viewport overflow를 점검한다. `npm run qa:web`으로 실행한다.
