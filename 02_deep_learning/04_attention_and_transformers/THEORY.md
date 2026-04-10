# 04 Attention and Transformers 이론 노트

## 핵심 개념

### 1. attention은 sequence mixing이다
- self-attention은 각 토큰이 같은 시퀀스 안의 다른 토큰을 얼마나 참고할지 weight로 정하고, 그 weight로 value를 섞어 새로운 표현을 만든다.
- 그래서 attention output은 "토큰 하나를 고른 결과"가 아니라 **여러 토큰 표현의 가중합**이다.
- `03_nlp_bridge/02_attention_and_transformer_block`에서 attention weight와 mask를 봤다면, 여기서는 그 현상을 모델 패밀리 차원에서 다시 읽는다.

### 2. multi-head는 서로 다른 관점의 mixing을 병렬로 둔다
- head를 여러 개 두면 같은 시퀀스를 여러 projection subspace에서 동시에 읽을 수 있다.
- 어떤 head는 가까운 토큰 관계에, 다른 head는 긴 거리 dependency나 구문 단서에 더 크게 반응할 수 있다.
- 직관적으로는 "한 번에 하나의 섞기 규칙"이 아니라 **여러 섞기 규칙을 병렬로 둔 뒤 다시 합치는 구조**라고 볼 수 있다.

### 3. encoder block과 decoder block은 정보 접근 규칙이 다르다
- **encoder block**은 보통 bidirectional self-attention을 사용해 입력 시퀀스 전체를 서로 참고하게 만든다.
- **decoder block**은 causal mask가 있는 self-attention으로 미래 토큰을 보지 못하게 막는다.
- seq2seq 구조에서는 decoder가 encoder hidden states를 읽는 **cross-attention** 단계가 추가될 수 있다.
- 그래서 둘 다 transformer block 계열이지만, "누구를 볼 수 있는가"라는 규칙이 다르다.

### 4. transformer는 recurrent bottleneck을 완화한다
- RNN / LSTM / GRU는 hidden state를 시간축으로 순서대로 갱신해야 해서 병렬화가 어렵다.
- transformer self-attention은 한 layer 안에서 모든 위치 쌍의 상호작용을 한 번에 계산할 수 있어, 학습 시 병렬 처리에 유리하다.
- 또한 멀리 떨어진 두 토큰 사이 정보 경로가 recurrent chain보다 짧아지기 쉽다.
- 대신 attention matrix 비용이 시퀀스 길이에 따라 크게 늘 수 있으므로, bottleneck이 완전히 사라지는 것이 아니라 **형태가 바뀐다**.

### 5. 이 단위는 model family track과 NLP bridge를 잇는 다리다
- NLP bridge에서는 attention block 내부 동작을 읽었다면, 여기서는 transformer가 왜 하나의 거대한 모델 패밀리가 되었는지 본다.
- 이 관점이 있으면 encoder-only / decoder-only / encoder-decoder 분류가 BERT / GPT / T5 같은 이름보다 먼저 머리에 들어온다.
- 즉, 세부 메커니즘 이해를 **모델 계열 분류틀**로 올려 주는 단위다.

## 자주 헷갈리는 지점
- attention weight가 크면 "그 토큰 하나만 선택됐다"고 오해하는 경우
- multi-head를 단지 파라미터 수 증가로만 보고, 서로 다른 mixing 관점이라는 점을 놓치는 경우
- encoder와 decoder 모두 self-attention을 쓰므로 사실상 같은 블록이라고 생각하는 경우
- transformer가 recurrent bottleneck을 줄여 준다는 말을 듣고, 계산량 문제까지 모두 해결한다고 오해하는 경우
- NLP 태스크에서 attention을 봤다는 이유로, 모델 패밀리 차원의 encoder/decoder 분류까지 이미 이해했다고 착각하는 경우

## 이 단위에서 관찰할 것
- attention row 하나가 어떤 token들로 분산되는지, 그래서 output이 왜 혼합 표현인지 확인한다.
- head를 나눴을 때 head별로 다른 토큰에 weight가 몰릴 수 있다는 점을 관찰한다.
- encoder block에서는 전체 문맥 접근이 가능하고, decoder block에서는 causal mask 때문에 미래 위치가 막힌다는 점을 비교한다.
- RNN류는 time step 수만큼 순차 의존이 남지만, transformer는 layer 단위 병렬 mixing이 가능하다는 점을 표나 shape 관찰로 정리한다.
- 이후 NLP 트랙에서 만날 모델들을 encoder-only / decoder-only / encoder-decoder 중 어디에 놓을지 미리 분류해 본다.
