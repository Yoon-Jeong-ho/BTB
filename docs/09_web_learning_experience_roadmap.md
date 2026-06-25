# 09 Web Learning Experience Roadmap

BTB 웹사이트를 단순 문서 뷰어가 아니라 “읽기 → 실행 → 관찰 → 해석 → 자가 점검 → 다음 단원” 루프로 발전시키기 위한 기준이다.

## 참고한 학습 사이트 패턴

- Google Machine Learning Crash Course는 모듈을 자기완결적으로 구성하되, 초심자에게는 순서대로 진행하라고 안내한다. 또한 텍스트, 시각 위젯, 영상, 퀴즈, 선택형 프로그래밍 실습을 함께 둔다.
  - https://developers.google.com/machine-learning/crash-course
  - https://support.google.com/machinelearningeducation/answer/7652516?hl=en
- fast.ai Practical Deep Learning for Coders는 초반부터 완성 예제를 보여주고, 각 lesson에 hands-on notebook 실습을 붙여 “코드를 직접 돌리는 것”을 학습의 핵심으로 둔다.
  - https://course.fast.ai/
  - https://course.fast.ai/Lessons/lesson1.html
- Hugging Face LLM Course는 Transformer 기본, 모델 사용/파인튜닝/공유, datasets/tokenizers, classic NLP와 LLM, demos, advanced LLM topics처럼 선행 지식이 필요한 순서로 확장한다. Python과 introductory deep learning 선행을 명시한다.
  - https://huggingface.co/learn/llm-course/en/chapter1/1

## BTB에 적용할 원칙

1. **문서 이동보다 학습 루프를 앞세운다.** README/THEORY/코드를 파일 링크로 보내지 말고, 단원 화면 안에서 읽고 바로 실행하게 한다.
2. **경로 선택을 명시한다.** 무기초 전체 1-pass, LLM/RLHF 빠른 경로, Multimodal/VLA 경로, Systems 심화 경로를 따로 보여 주고 각 경로의 다음 단원을 추천한다.
3. **실행 로그는 해석 카드로 바꾼다.** raw stdout만 보여주지 말고, 봐야 할 숫자, 산출물 위치, 다음 질문을 먼저 제시한다.
4. **자가 점검은 로컬 진행률로만 남긴다.** GitHub에 공유하지 않고 브라우저 localStorage에 사용자별 체크·메모·경로 상태를 누적한다.
5. **이론/코드/실전 비율을 단원 안에서 닫는다.** 각 단원은 목표, 선행 개념, 핵심 용어, 실행 코드, metric/artifact, 분석 질문이 한 화면에서 연결되어야 한다.

## 이번 반영분

- 사용자 프로필 옆에 학습 경로 선택을 추가하고, 선택 경로 기준 진행률과 다음 단원 추천을 표시한다.
- Python 실행 후 “실행 관찰 카드”를 표시해 runner/device, 봐야 할 숫자, artifact 힌트, 후속 질문을 바로 보여준다.
- 단원 가이드에 자가 점검 체크리스트를 추가해 목표 설명, 실행 관찰, 분석 질문 답변까지 확인한다.
- Playwright QA에 경로 선택, 자가 점검, 실행 관찰 카드 검사를 추가해 UI 회귀를 막는다.

## 다음 발전 우선순위

1. **미니 퀴즈/오답 노트**: Google MLCC처럼 단원별 2~3문항을 사이트 안에서 풀고, 오답 이유를 localStorage 메모로 남긴다.
2. **artifact viewer**: 실행 결과 JSON/SVG/표를 raw log 밑에 별도 카드로 렌더링해 숫자 변화와 그림을 바로 확인한다.
3. **코드 셀형 실행**: fast.ai/Jupyter 흐름처럼 전체 파일 실행 외에 핵심 함수 또는 작은 입력 예제를 부분 실행할 수 있게 한다.
4. **경로 prerequisite gate**: Hugging Face LLM Course처럼 LLM/RLHF/Multimodal/VLA 진입 전에 Python, DL, Transformer 선행 점검을 보여준다.
5. **개념 연결 그래프**: key terms를 다음 단원과 연결해 “왜 이 단원을 지금 배우는지”를 시각적으로 보여준다.
