# 02 Image Captioning 선행 개념

## 꼭 알고 오면 좋은 것
- retrieval와 captioning이 둘 다 image-text 문제이지만, ranking과 generation이라는 서로 다른 목표를 가진다는 점
- teacher forcing이 학습 시 정답 이전 토큰을 넣어 주는 방식이라는 점
- greedy decoding이 각 시점에서 가장 높은 토큰 하나만 고르는 단순한 추론 전략이라는 점

## 빠른 자기 점검
- 같은 이미지라도 reference caption이 여러 개일 수 있어서 exact match 하나만으로는 부족하다는 점을 설명할 수 있는가?
- BLEU-1처럼 unigram 겹침을 세는 지표가 왜 hallucination을 완전히 잡아내지 못하는지 말할 수 있는가?
- decoder loss가 낮아도 실제 생성 문장이 틀릴 수 있다는 말을 teacher forcing 관점에서 설명할 수 있는가?
