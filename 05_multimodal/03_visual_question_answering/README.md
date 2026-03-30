# 03 Visual Question Answering

## 목표

시각 정보와 언어 정보를 함께 읽어 `정답률`, `질문 유형별 성능`, `추론 실패 패턴` 을 분석한다.

## 추천 데이터셋

- `VQA v2`: 범용 시각 질의응답
- `ScienceQA`: 설명과 reasoning 중심
- `NLVR2`: 시각적 논리 판단

## 실습 파이프라인

1. question type / answer type 분포 확인
2. frozen VLM prompting 또는 zero-shot baseline
3. PEFT 방식의 task adaptation
4. answer type별 accuracy와 calibration 비교
5. 왜 틀렸는지 image grounding / question understanding / commonsense 부족으로 분류

## 결과로 남길 figure

- answer type breakdown
- overall accuracy chart
- qualitative QA panel

## 분석으로 남길 figure

- question type별 failure summary
- reasoning chain mismatch 사례
- text-only shortcut 여부를 보는 ablation plot

## 승격 기준

- 숫자뿐 아니라 실패 원인을 유형화할 수 있다.
- 모델이 이미지를 정말 보는지, 텍스트 shortcut에 기대는지 분석이 남아 있다.
