# BTB 학습 지도

`web/`는 BTB 커리큘럼을 브라우저에서 따라가기 위한 학습 지도다. 파일 이름을 그대로 눌러 새 탭으로 이동하는 대신, 사이트 안에서 “단원 안내”, “핵심 이론”, “준비 확인”, 실습 코드를 한 흐름으로 읽고 체크할 수 있게 한다.

```bash
python scripts/build_web_catalog.py
python scripts/study_server.py --port 8000 --device auto
# http://localhost:8000/web/ 열기
```

> 주의: 저장소 루트에서 실행해야 사이트 안에서 단원 문서와 실습 코드가 함께 열린다. `web/` 폴더 안에서 서버를 띄우면 커리큘럼 문서 fetch 경로가 깨질 수 있다.

## Python 버튼 실행까지 쓰기

위의 `study_server.py`가 기본 실행 경로다. 일반 `python -m http.server 8000`은 정적 파일만 보여 주므로 브라우저 버튼으로 Python을 실행할 수 없다. 이미 정적 서버로 열었다면 그 터미널에서 `Ctrl+C`로 멈춘 뒤, 저장소 루트에서 아래 실행 서버로 다시 띄운다.

```bash
python scripts/study_server.py --port 8000 --device auto
# http://localhost:8000/web/ 열기
```

읽기 전용으로 문서만 확인할 때만 아래 정적 서버를 fallback으로 쓴다.

```bash
python -m http.server 8000
# http://localhost:8000/web/ 열기
```

`study_server.py`는 임의 명령을 실행하지 않고, 저장소 안의 `scratch_lab.py`, `framework_lab.py`, `analysis.py`, `run_stage.py`만 허용 목록으로 실행한다. 실행 버튼은 코드 블록 아래에 있고, 실행 결과·종료 코드·선택된 실행 환경은 그 아래 결과 패널에 표시된다. 실행이 끝나면 이번 실행에서 새로 만들어지거나 갱신된 JSON/SVG/표 산출물을 “산출물 뷰어”에서 바로 열어 보고, 전체 파일을 돌리기 전에는 “선택 함수 미리보기”로 함수의 입력·호출·저장 단서를 먼저 훑을 수 있다.

conda 환경이나 GPU 선택을 명시하고 싶다면 아래처럼 실행한다.

```bash
# conda 환경 이름으로 실행
python scripts/study_server.py --port 8000 --conda-env btb --device auto

# conda prefix 경로로 실행
python scripts/study_server.py --port 8000 --conda-prefix /path/to/env --device auto

# GPU를 쓰지 않고 CPU로 고정
python scripts/study_server.py --port 8000 --device cpu

# 특정 GPU를 강제로 사용
python scripts/study_server.py --port 8000 --device cuda --gpu-index 0
```

`--device auto`는 `nvidia-smi`로 idle GPU를 찾고, 조건에 맞는 GPU가 없거나 `nvidia-smi`가 없으면 CPU로 fallback한다.

## 진행률 저장 방식

- 단원 체크, 상태, 메모는 브라우저 `localStorage`의 `btb.study.progress.v1` 키에만 저장된다.
- 이 값은 GitHub에 공유되지 않고, Git 커밋에도 포함되지 않는다.
- 같은 브라우저 안에서 사용자별 프로필을 만들어 진행 사항을 따로 누적할 수 있다.
- 다른 브라우저/기기에서 이어 보려면 내보내기/가져오기 JSON을 수동으로 사용한다.

## 데이터 갱신

커리큘럼 문서나 `docs/curriculum_status.json`을 수정한 뒤에는 아래 명령으로 `web/catalog.json`을 다시 만든다.

```bash
python scripts/build_web_catalog.py --output web/catalog.json
```

## 보는 방식

- 왼쪽에서 트랙과 단원을 고르고, 오른쪽 넓은 reader에서 단원 안내·핵심 이론·준비 확인·실습 코드를 바로 읽는다.
- 사용자 프로필 옆의 학습 경로에서 `전체 1-pass`, `LLM/RLHF 빠른 경로`, `Multimodal/VLA 경로`, `Systems 심화 경로` 중 하나를 고르면 해당 경로 기준 진행률과 다음 단원 추천이 표시된다. 빠른 경로는 트랙 전체가 아니라 필수 unit만 압축해 보여준다.
- 문서 안의 로컬 `.md` 링크도 가능한 한 사이트 안에서 이어서 열리므로, 원본 문서 파일이 그대로 깨져 보이는 흐름을 피한다.
- `study_server.py --device auto`로 열면 Python 코드 아래의 버튼으로 실행하고 결과를 바로 아래에서 확인한다.
- 실행 후에는 원문 로그뿐 아니라 실행 관찰 카드에서 봐야 할 숫자, 이번 실행의 산출물 위치, 다음 질문을 먼저 확인한다.
- 산출물 뷰어에서 이번 실행이 갱신한 지표 JSON, SVG 그림, CSV/텍스트 표를 바로 확인하고 분석 질문과 연결한다.
- 선택 함수 미리보기는 코드를 임의 실행하지 않고 AST로 함수 구조, 호출 이름, 중간 변수, 산출물 단서를 보여준다.
- 단원별 자가 점검과 미니 퀴즈를 통해 “목표 설명 → 실행 관찰 → 분석 질문 답변”까지 끝났는지 스스로 확인한다. 짧은 서술형은 자동 정답 처리하지 않고 예시 기준과 비교한다.
- 틀린 퀴즈와 메모는 오답노트에 저장되며, 사용자 프로필마다 현재 브라우저에만 남는다.
- 체크리스트와 읽음 표시는 사용자별 localStorage에만 누적된다.

자세한 UX 발전 방향과 참고 사이트는 [docs/09_web_learning_experience_roadmap.md](../docs/09_web_learning_experience_roadmap.md)에 정리했다.

## Playwright QA

브라우저에서 실제로 보기 편한지 확인하려면 Playwright를 사용한다. 최초 1회는 `npm install`과 `npx playwright install chromium`을 실행하고, 이후에는 아래 명령으로 데스크톱/모바일 smoke QA와 스크린샷 캡처를 반복한다.

```bash
npm run qa:web
```

스크린샷은 기본적으로 `/tmp/btb-playwright-site-qa`에 저장된다. `BTB_QA_OUT=원하는/경로 npm run qa:web`처럼 출력 경로를 바꿀 수 있다.
