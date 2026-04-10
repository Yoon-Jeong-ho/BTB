# 01 Text Classification 실행 관측

## 관측 결과
- scratch eval accuracy: `1.0`
- scratch eval macro F1: `1.0`
- scratch top positive tokens: `['검색이', '만족한다', '명확해서', '복습하기', '빨라져서']`
- scratch top negative tokens: `['광고가', '낮다', '너무', '눌리고', '늦고']`
- framework eval accuracy: `1.0`
- framework eval macro F1: `1.0`
- framework vocab size: `44`
- framework loss history head: `[0.762755, 0.580499, 0.421497, 0.266388, 0.136398]`

## 한국어 해석
- scratch baseline은 `['검색이', '만족한다', '명확해서', '복습하기', '빨라져서']` 와 `['광고가', '낮다', '너무', '눌리고', '늦고']` 같은 token cue를 중심으로 문장을 읽었다. 즉 이 toy dataset에서는 표면 lexical signal만으로도 어느 정도 분리가 된다.
- 첫 scratch 예문 `업데이트가 안정적이고 사용이 편하다` 에서 gold=`positive`, pred=`positive` 로 나온 것은 baseline이 token count 합을 어떻게 의사결정 근거로 쓰는지 보여 준다.
- tiny PyTorch classifier의 첫 예문 `업데이트가 안정적이고 사용이 편하다` 는 gold=`positive`, pred=`positive` 였다. 확률 분포 `{'negative': 0.214353, 'positive': 0.785647}` 를 함께 보면, neural model이 단순 label만이 아니라 class confidence도 출력한다는 사실을 볼 수 있다.
- scratch와 framework의 accuracy / macro F1을 함께 비교하면, 전체 정답률만 볼 때 놓치기 쉬운 클래스별 균형 감각을 다시 확인할 수 있다.
- 이 toy unit에서는 두 모델 모두 매우 작기 때문에 "최고 성능"보다 **baseline을 세우고, feature 표현 차이를 해석하는 출발점**을 만드는 것이 더 중요하다.

## 이론 다시 연결하기
- 안정적인 해석 프레임은 `analysis.md`에 유지한다.
- 핵심 개념 복습: [THEORY.md](./THEORY.md)
