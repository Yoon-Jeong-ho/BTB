# BTB Study Website

`web/`는 BTB 커리큘럼을 브라우저에서 보기 위한 정적 사이트다.

```bash
python scripts/build_web_catalog.py
python -m http.server 8000
# http://localhost:8000/web/ 열기
```

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
