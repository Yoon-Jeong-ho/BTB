# BTB Study Website

`web/`는 BTB 커리큘럼을 브라우저에서 보기 위한 정적 사이트다. 단원 README, THEORY, PREREQS, scratch/framework/analysis 코드를 README 파일을 새 탭으로 직접 여는 방식이 아니라 사이트 안에서 렌더링해 읽고 체크할 수 있게 한다.

```bash
python scripts/build_web_catalog.py
python -m http.server 8000
# http://localhost:8000 또는 http://localhost:8000/web/ 열기
```

> 주의: 저장소 루트에서 실행해야 사이트 안에서 단원 문서와 실습 코드가 함께 열린다. `web/` 폴더 안에서 서버를 띄우면 커리큘럼 문서 fetch 경로가 깨질 수 있다.

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
- 체크리스트와 현재 자료 체크는 사용자별 localStorage에만 누적된다.

## Playwright QA

브라우저에서 실제로 보기 편한지 확인하려면 Playwright를 사용한다. 최초 1회는 `npm install`과 `npx playwright install chromium`을 실행하고, 이후에는 아래 명령으로 데스크톱/모바일 smoke QA와 스크린샷 캡처를 반복한다.

```bash
npm run qa:web
```

스크린샷은 기본적으로 `/tmp/btb-playwright-site-qa`에 저장된다. `BTB_QA_OUT=원하는/경로 npm run qa:web`처럼 출력 경로를 바꿀 수 있다.
