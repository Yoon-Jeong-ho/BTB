# BTB Study Website

`web/`는 BTB 커리큘럼을 브라우저에서 보기 위한 정적 사이트다. 단원 README, THEORY, PREREQS, scratch/framework/analysis 코드를 README 파일을 새 탭으로 직접 여는 방식이 아니라 사이트 안에서 렌더링해 읽고 체크할 수 있게 한다.

```bash
python scripts/build_web_catalog.py
python -m http.server 8000
# http://localhost:8000 또는 http://localhost:8000/web/ 열기
```

> 주의: 저장소 루트에서 실행해야 사이트 안에서 단원 문서와 실습 코드가 함께 열린다. `web/` 폴더 안에서 서버를 띄우면 커리큘럼 문서 fetch 경로가 깨질 수 있다.

## Python 버튼 실행까지 쓰기

일반 `python -m http.server 8000`은 정적 파일만 보여 주므로 브라우저 버튼으로 Python을 실행할 수 없다. 이미 정적 서버로 열었다면 그 터미널에서 `Ctrl+C`로 멈춘 뒤, 저장소 루트에서 아래 실행 서버로 다시 띄운다.

```bash
python scripts/study_server.py --port 8000 --device auto
# http://localhost:8000/web/ 열기
```

`study_server.py`는 임의 명령을 실행하지 않고, 저장소 안의 `scratch_lab.py`, `framework_lab.py`, `analysis.py`, `run_stage.py`만 허용 목록으로 실행한다. 실행 결과의 표준 출력/오류 출력, 종료 코드, 선택된 실행 환경은 코드 블록 위의 결과 패널에 표시된다.

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

- 왼쪽 좁은 영역에서 track과 unit을 고르고, 오른쪽 넓은 reader에서 README/THEORY/PREREQS/실습 코드를 바로 읽는다.
- 문서 안의 로컬 `.md` 링크도 가능한 한 사이트 안에서 이어서 열리므로, raw README 파일이 깨져 보이는 흐름을 피한다.
- `study_server.py --device auto`로 열면 Python 코드 탭에서 버튼 하나로 실행하고 결과를 바로 아래에서 확인한다.
- 체크리스트와 자료 완료 표시는 사용자별 localStorage에만 누적된다.

## Playwright QA

브라우저에서 실제로 보기 편한지 확인하려면 Playwright를 사용한다. 최초 1회는 `npm install`과 `npx playwright install chromium`을 실행하고, 이후에는 아래 명령으로 데스크톱/모바일 smoke QA와 스크린샷 캡처를 반복한다.

```bash
npm run qa:web
```

스크린샷은 기본적으로 `/tmp/btb-playwright-site-qa`에 저장된다. `BTB_QA_OUT=원하는/경로 npm run qa:web`처럼 출력 경로를 바꿀 수 있다.
