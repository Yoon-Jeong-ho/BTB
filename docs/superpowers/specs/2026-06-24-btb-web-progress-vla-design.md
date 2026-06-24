# BTB Web Progress + VLA Bridge Design

## Goal
BTB 학습자가 문서 트리를 직접 뒤지는 대신 웹사이트에서 전체 커리큘럼을 보고, 단원별 진행 체크를 사용자별 로컬 캐시에만 누적하며, 무기초에서 LLM/RL/멀티모달/VLA까지 이어지는 학습 경로를 확인하게 만든다.

## Approach
- 정적 웹 앱(`web/`)을 추가한다. 별도 서버/DB 없이 GitHub Pages 또는 `python -m http.server -d web`로 볼 수 있다.
- 커리큘럼 데이터는 `docs/curriculum_status.json`, 각 track README, 각 `lesson.yaml`에서 생성한 `web/catalog.json`을 사용한다.
- 진행률은 브라우저 `localStorage`에만 저장한다. 저장 키는 `btb.study.progress.v1`이고, Git에 올라가는 파일에는 사용자 진행 상태를 쓰지 않는다.
- 사용자 프로필은 로컬 display name/profile id 단위로 나누어 같은 브라우저 안에서도 학습자별 진행률을 분리한다.
- 기존 커리큘럼의 마지막이 multimodal이므로, VLA 진입을 위해 `10_vla/01_vision_language_action_grounding` runnable unit을 추가한다.
- 남는 GPU/conda 사용은 강제 학습이 아니라 안전한 환경 점검/실험 실행 계획으로 구성한다. 현재 장비에서 GPU 4-7이 유휴로 보이며, PyTorch 2.8 CUDA 환경이 있다.

## Components
1. `scripts/build_web_catalog.py`
   - curriculum manifest와 lesson metadata를 읽어 정적 JSON을 생성한다.
   - lesson.yaml이 없는 01_ml 단원도 README 기반 fallback으로 포함한다.
2. `web/index.html`, `web/styles.css`, `web/app.js`, `web/catalog.json`
   - 트랙/단원 탐색, 검색, 미완료 필터, 다음 학습 추천, 체크리스트, 메모, export/import/reset 제공.
   - 진행 저장은 `localStorage`만 사용하고 네트워크 전송 코드를 두지 않는다.
3. `10_vla/01_vision_language_action_grounding`
   - vision-language-action의 최소 개념: 시각 상태 + 언어 지시 → action token / safety gate.
   - scratch/framework/analysis 산출물을 갖춘 CPU-safe runnable unit으로 둔다.
4. `docs/04_gpu_conda_experiment_plan.md`
   - 현재 conda/GPU 상태 기반 권장 실행 명령과 유휴 GPU 활용 규칙을 문서화한다.

## Data Flow
`docs/curriculum_status.json` → `scripts/build_web_catalog.py` → `web/catalog.json` → browser render.
User progress: browser UI → `localStorage[btb.study.progress.v1]`; no repo writes, no GitHub sharing.

## Error Handling
- `catalog.json` 로드 실패 시 로컬 서버 실행 안내를 보여 준다.
- 손상된 progress JSON은 백업 키로 옮기고 빈 상태로 복구한다.
- Import JSON은 schema version과 users map을 확인한 뒤 병합한다.

## Testing
- 웹 카탈로그 생성기가 manifest와 on-disk 단원을 빠짐없이 반영하는지 테스트한다.
- `web/catalog.json`이 생성기 출력과 동기화되어 있는지 테스트한다.
- `app.js`가 localStorage 기반이며 진행률 네트워크 전송 API를 포함하지 않는지 테스트한다.
- VLA unit의 required files, docs, scratch/framework/analysis 산출물 계약을 테스트한다.
- 기존 curriculum topology/status/link tests를 업데이트한다.
