# 03 Domain Adaptive Pretraining 회고

- 학습자 입장에서 **domain shift**를 vocabulary 차이 하나로만 보지 않으려면, 문체·문서 형식·정보 밀도 중 무엇을 먼저 관찰해야 하는가?
- 같은 causal LM objective를 유지한 채 continued pretraining을 한다는 말이, fine-tuning이나 instruction tuning과 어떻게 다른가?
- pure domain schedule이 in-domain gain을 빠르게 만들면서도 **catastrophic forgetting** 위험을 키우는 이유를 내 말로 설명해 보라.
- replay mixture를 넣으면 general retention은 좋아질 수 있지만 adaptation 속도가 느려질 수 있다. 내 프로젝트라면 어느 정도의 general replay share를 먼저 시도하겠는가?
- data selection에서 문서 수가 많은 noisy corpus와 문서 수가 적은 curated corpus가 있을 때, 어떤 품질·중복·contamination 기준으로 선택하겠는가?
- stopping을 마지막 step이나 최저 training loss로 정하지 않고, in-domain validation과 general-domain retention을 함께 보는 이유는 무엇인가?
- DAPT를 끝낸 뒤 SFT로 넘어갈 때, “무엇을 더 알게 만들었는가”와 “어떤 assistant behavior로 드러낼 것인가”를 어떻게 분리해 기록할 것인가?
