# 04 Large Scale Tabular

이 단계는 **대규모 표형 분류를 어떻게 읽고, 어떻게 비교하고, 무엇을 의심해야 하는지**를 익히는 공부 노트다.
작은 데이터에서는 점수표 하나만으로도 모델을 고를 수 있지만, 큰 데이터에서는 그렇지 않다. 여기서는 품질, class 균형성, 학습 비용, 추론 비용을 함께 읽어야 한다.

## 먼저 잡아야 할 질문

- 왜 accuracy만 보면 class별 실패를 놓치기 쉬운가?
- macro-F1과 macro-recall은 어떤 질문에 답하는가?
- 왜 large-scale tabular에서는 품질과 비용을 같이 기록해야 하는가?
- 왜 GPU boosting이 strong baseline이 되는가?
- 어떤 confusion pair가 다음 실험 가설의 출발점이 되는가?

## 이 stage에서 배우는 용어

### large-scale tabular
샘플 수가 커서 한 번의 실험이 비싸지는 표형 데이터 문제를 뜻한다. 점수만 높으면 끝이 아니라, **얼마나 빨리 반복 실험할 수 있는가**까지 함께 본다.

### multiclass classification
정답이 두 개가 아니라 여러 개의 class 중 하나인 분류 문제다. 이 stage에서는 `cover_type` 7개 class를 구분한다.

### macro-F1 / macro-recall
각 class의 성능을 먼저 계산한 뒤 평균낸 지표다. 다수 class가 전체 점수를 덮어버리는 문제를 줄여 준다.

### cost-quality trade-off
더 좋은 점수를 얻는 데 드는 시간/메모리 비용을 함께 보는 관점이다. 대규모 실험에서는 이 trade-off를 무시하면 실험 회전율이 무너진다.

## 왜 Covertype를 쓰는가

- 표형 feature 수가 많고 데이터 규모가 커서 실험 비용을 체감하기 좋다.
- multiclass이므로 accuracy 하나만으로는 부족하다는 사실이 잘 드러난다.
- tree 계열, linear 계열, GPU MLP를 같은 규약으로 비교하기 좋다.
- 다음 단계 서버 실험(HIGGS 등)으로 넘어가기 전 기준점을 만들 수 있다.

## 어떤 모델을 비교하는가

- `sgd_linear`: 아주 빠른 약한 baseline. 선형 경계가 어디까지 버티는지 본다.
- `shallow_tree`: 얕은 비선형 baseline. 복잡도를 많이 쓰지 않고도 어느 정도 class 경계를 잡는지 본다.
- `hist_gbdt`: scikit-learn strong baseline. 큰 tabular에서 자주 강한 출발점이다.
- `xgboost_gpu`: GPU boosting strong baseline. 품질-비용 균형을 끌어올릴 수 있는지 본다.
- `gpu_mlp`: 신경망 비교군. GPU를 쓴다고 tabular에서 항상 이기지는 않는다는 점을 확인한다.

## 무엇을 주의 깊게 봐야 하나

1. accuracy가 높아도 macro-F1이 충분히 높은가?
2. class별 recall 편차가 큰가?
3. 어느 confusion pair가 특히 큰가?
4. fit time / predict time / peak RSS가 감당 가능한가?
5. 더 큰 데이터를 다룰 때도 같은 구조가 유지될 수 있는가?

## 코드 구조

이 stage는 이제 thin wrapper가 아니라 stage 내부에서 바로 읽히는 코드 구조를 사용한다.

- `dataset.py`: Covertype 로딩과 전처리 보조
- `experiment.py`: 모델 비교, 비용-성능 계산, class-wise figure 생성
- `report.py`: 최신 artifact 경로 보조 함수
- `run_stage.py`: stage 단독 실행 진입점

## artifact 구조

실험 산출물은 이제 이 stage 안에 같이 놓는다.

- 최신 artifact: [`artifacts/20260327-164831_covertype_large-scale-suite_s42/README.md`](artifacts/20260327-164831_covertype_large-scale-suite_s42/README.md)
- 요약: [`artifacts/20260327-164831_covertype_large-scale-suite_s42/summary.md`](artifacts/20260327-164831_covertype_large-scale-suite_s42/summary.md)
- 결과 figure: `artifacts/20260327-164831_covertype_large-scale-suite_s42/figures/results/`
- 분석 figure: `artifacts/20260327-164831_covertype_large-scale-suite_s42/figures/analysis/`

## 읽는 순서

1. [THEORY.md](THEORY.md)로 accuracy/macro metric/cost-quality trade-off 배경을 잡는다.
2. [artifacts/20260327-164831_covertype_large-scale-suite_s42/README.md](artifacts/20260327-164831_covertype_large-scale-suite_s42/README.md)로 실제 실험 설계와 결과 해석을 본다.
3. summary와 figure를 보며 어떤 class boundary가 무너졌는지 확인한다.
4. 다음 실험 가설을 스스로 문장으로 적어 본다.
