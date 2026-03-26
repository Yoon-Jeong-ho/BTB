# Artifacts

모델 가중치와 관련 메타데이터를 관리하는 위치다.

## 폴더 역할

- `staging/`: 서버에서 임시로 모으는 후보 가중치. Git ignore.
- `promoted/`: 작은 모델 또는 정말 공유 가치가 있는 가중치.
- `MODEL_REGISTRY.md`: Git 또는 Hugging Face에 올라간 모델 기록.

## 운영 원칙

1. 실험 중간 checkpoint는 커밋하지 않는다.
2. 작은 가중치만 `promoted/` 아래에 둔다.
3. `promoted/` 의 가중치는 Git LFS를 사용한다.
4. 큰 가중치는 Hugging Face Hub에 올리고, 링크와 commit hash를 `MODEL_REGISTRY.md` 에 남긴다.

## Hugging Face Hub 권장 흐름

1. 로컬/서버에서 모델 카드와 핵심 metrics 정리
2. Hugging Face Hub에 업로드
3. 이 저장소의 `MODEL_REGISTRY.md` 에
   - HF repo
   - 사용 데이터셋
   - 핵심 metric
   - 대응 Git commit
   를 기록
