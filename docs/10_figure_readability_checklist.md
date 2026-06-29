# Figure Readability Checklist

학습용 SVG는 “코드를 돌렸더니 그림이 생겼다”가 아니라, 그림만 봐도 실험의 결론을 읽을 수 있어야 한다. 새 `scratch_lab.py`가 SVG를 만들 때는 아래 항목을 기본 계약으로 둔다.

## 필수 기준

1. **그림이 답하는 질문을 제목/부제에 쓴다.** 예: 같은 learning rate에서 왜 raw feature가 터지는가?
2. **축이 있는 그림은 축 이름과 tick 값을 둔다.** 단순 막대도 단위(MB, loss, gradient 등)를 표시한다.
3. **코드용 축약어보다 학습자용 라벨을 우선한다.** `normalized+l2`보다 `Z-score normalized + L2`처럼 쓴다.
4. **스케일이 크게 다르면 확대 패널이나 작은 다중 패널로 나눈다.** 한 축에 눌려 보이지 않는 선은 학습 자료로 실패한 그림이다.
5. **핵심 숫자를 그림 안에 직접 적는다.** final loss, 평균 gradient, saving ratio처럼 해석을 결정하는 값은 범례만으로 남기지 않는다.
6. **범례보다 직접 라벨을 선호한다.** 눈이 선↔범례를 왕복하지 않아도 되게 한다.
7. **비교가 목적이면 작은 요약 카드/표를 붙인다.** 특히 tiny demo에서 “loss는 조금 나빠졌지만 weight norm은 줄었다” 같은 nuance가 중요하다.
8. **3초 안에 결론이 보이지 않으면 다시 설계한다.** 정확한 SVG보다 읽히는 SVG가 우선이다.

## 이번 점검에서 확인한 대표 위험

- `00_foundations/04_regularization_and_normalization`: raw loss가 너무 커서 normalized 곡선이 바닥에 눌렸다. → 같은 스케일 패널 + normalized 확대 패널 + 요약 카드로 수정.
- `00_foundations/03_gradients_and_backpropagation`: 축/틱/이동 방향 설명이 부족했다. → 축, tick, before/after callout을 추가.
- `06_training_systems/01_torchrun_and_ddp_basics`: rank별 gradient 막대가 all-reduce 평균과 연결되지 않았다. → 평균선과 rank/node/local rank 라벨을 추가.
- `06_training_systems/03_deepspeed_zero`: ZeRO stage 막대가 MB 단위와 절감 비율을 바로 보여 주지 않았다. → y축 단위, value label, Stage 3 saving callout을 추가.
