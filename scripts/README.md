# Scripts

이 폴더는 학습/평가/검증 스크립트를 모으는 공간이다. 예시 경로도 현재 `00→10` 커리큘럼 루트를 기준으로 적는다.

```text
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode scratch
python scripts/run_lesson.py --unit 00_foundations/05_gpu_memory_runtime --mode framework --device cpu
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode analysis --device cpu
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode all --device cpu
python scripts/run_lesson.py --unit 01_ml/01_tabular_classification --mode all --device cpu
python scripts/build_lesson_report.py --unit 00_foundations/01_tensor_shapes
python scripts/audit_curriculum.py --strict
python scripts/check_curriculum_links.py
python 01_ml/01_tabular_classification/run_stage.py --gpu 0
python scripts/run_lesson.py --unit 10_vla/01_vision_language_action_grounding --mode framework --device auto
```

`auto`/`cuda`는 공유 장비의 idle 상태를 `nvidia-smi`로 확인한 뒤, 필요하면 `CUDA_VISIBLE_DEVICES=<idle-index>`로 노출 장치를 제한해 사용한다. 일반 학습은 기본값 `cpu`로 시작한다.

현재 커리큘럼 기준으로 NLP 브리지는 `03_nlp_bridge`, applied NLP는 `04_nlp`, multimodal 브리지는 `08_multimodal_bridge`, multimodal 실습은 `09_multimodal`, VLA 입구는 `10_vla` 아래에 놓인다.

새 scaffold track/unit는 아직 `planned` 상태일 수 있으므로, 실행 전에는 먼저 `docs/curriculum_status.json`에서 runnable 여부를 확인하고, 각 track README에 status note/table이 있으면 보조 맥락으로 함께 참고하는 것을 권장한다.

## Task 6 automation scaffold

- `run_lesson.py`: `lesson.yaml`을 읽고 `scratch`, `framework`, `analysis`를 개별 실행하거나 `all`로 순서대로 실행한다. ML real-data unit처럼 `run_stage.py`만 있는 경우 `--mode stage` 또는 `--mode all`이 stage entrypoint를 실행한다. 기본은 공유 GPU를 임의로 잡지 않는 `cpu`이며, `--device auto|cpu|cuda` 선택은 `BTB_DEVICE`로 실습 코드에 전달된다.
- `build_lesson_report.py`: unit의 확인 가능한 필수 산출물을 검사하고 metric 값·artifact 링크·분석 질문이 포함된 `artifacts/summary.md`를 만든다. 자동 확인할 수 없는 자유 형식 선언도 report에 명시한다.
- `audit_curriculum.py`: manifest의 모든 단원을 canonical metadata parser로 읽고 필수 metadata, 실행 entrypoint, fidelity/compute coverage를 검사한다. CI나 제출 전에는 `--strict`를 사용한다.
- `check_curriculum_links.py`: 루트 README, docs, foundations/bridge/track/VLA 문서의 로컬 markdown 링크를 점검한다.
- `build_web_catalog.py`: manifest와 lesson metadata를 정적 웹사이트용 `web/catalog.json`으로 변환한다.
- `_lesson_metadata.py`: 현재 BTB의 제한된 `lesson.yaml` 스키마(top-level scalar/list와 한 단계 문자열 mapping)를 파싱하는 무의존성 canonical 로더다.

- `playwright_site_qa.js`: Playwright로 정적 웹사이트의 데스크톱/모바일 reader, 코드 설명, viewport overflow를 점검한다. `npm run qa:web`으로 실행한다.
