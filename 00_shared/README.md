# 00 Shared

공통 템플릿과 실험 규약을 두는 공간이다.

## 공통 Unit Contract

기본 contract는 `README/THEORY/PREREQS/scratch/framework/analysis/reflection` 흐름이다.
각 학습 단위는 가능하면 아래 파일을 함께 둔다.

- `README.md`: 왜 이 단위를 배우는지와 실습 진입점
- `THEORY.md`: 핵심 개념, 수식, 직관
- `PREREQS.md`: 선행 개념과 다시 볼 링크
- `scratch_lab.py`: 바닥부터 확인하는 작은 실험
- `framework_lab.py`: 프레임워크 기반 재현 실험
- `analysis.md`: 결과 해설과 실패 사례
- `reflection.md`: 학습자 관점 회고와 다음 질문

## 템플릿 목록

- `foundation_readme_template.md`
- `foundation_theory_template.md`
- `foundation_analysis_template.md`
- `foundation_reflection_template.md`
- `ml_theory_template.md`
- `ml_report_template.md`
- `run_summary_template.md`
- `model_card_template.md`

## 사용 원칙

- 템플릿은 복붙 종착점이 아니라 초안 시작점으로 쓴다.
- 표나 수치를 넣으면 반드시 해설 문장을 붙인다.
- runtime 관측이 있는 단위는 GPU/CPU 메모리와 시간 차이를 함께 남긴다.
