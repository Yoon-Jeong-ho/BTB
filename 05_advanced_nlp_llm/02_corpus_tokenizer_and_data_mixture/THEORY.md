# 02 Corpus, Tokenizer, and Data Mixture 이론 노트

## 핵심 개념

### 1. pretraining 성능은 objective만이 아니라 data pipeline으로도 결정된다
- causal LM이든 masked/span corruption이든, 실제로 모델이 보는 학습 신호는 결국 corpus와 tokenizer를 통과한 뒤의 token stream이다.
- 그래서 "같은 objective"라도 어떤 문서를 넣었는지, 문서를 어떤 단위로 쪼갰는지, 어떤 slice를 더 자주 보게 했는지에 따라 모델 행동이 달라진다.
- 큰 그림에서는 다음 세 질문이 함께 움직인다.
  - **무엇을 넣을 것인가?** → corpus quality / domain / language coverage
  - **어떻게 자를 것인가?** → tokenizer design / vocabulary / sequence length
  - **얼마나 섞을 것인가?** → data mixture / token budget / weighting

### 2. corpus quality는 단순히 "문서가 많다"가 아니다
- 좋은 corpus는 보통 다음 조건을 동시에 본다.
  - 출처와 라이선스가 분명한가
  - boilerplate, spam, OCR noise, templated SEO text가 과도하지 않은가
  - 언어/도메인 라벨이 어느 정도 신뢰 가능한가
  - 문서 단위가 지나치게 짧거나 깨져 있지 않은가
  - evaluation set과 겹치지 않는가
- 규모가 커도 품질이 낮으면 모델은 **노이즈를 학습하는 데 token budget을 낭비** 할 수 있다.
- 반대로 너무 깨끗한 소규모 corpus만 쓰면 분포 coverage가 약해져 다양한 표현을 못 배울 수 있다.
- 결국 corpus curation은 "무조건 많이"와 "무조건 깨끗하게" 사이의 균형을 찾는 문제다.

### 3. tokenizer는 비용과 표현력을 함께 바꾸는 설계다
- tokenizer는 문자열을 모델이 다루는 discrete unit으로 바꾸는 규칙이다.
- vocabulary가 크면 자주 쓰는 단어나 subword를 한 번에 묶어 **sequence length를 줄일 수** 있지만, 희귀 조각의 통계가 sparse해지고 vocab 관리가 무거워질 수 있다.
- vocabulary가 작으면 coverage는 넓어지지만, 특히 한국어·전문용어·코드 식별자처럼 형태 변화가 큰 영역에서 token이 과도하게 잘게 쪼개질 수 있다.
- 중요한 trade-off는 보통 아래와 같다.
  - **compression**: 평균 문서당 token 수가 얼마나 줄어드는가?
  - **coverage**: 희귀 단어/도메인 용어를 지나치게 `[UNK]` 또는 과도한 분해 없이 담는가?
  - **boundary interpretability**: subword 경계가 사람이 해석 가능한 단위를 어느 정도 보존하는가?
  - **multilingual fairness**: 특정 언어만 유난히 많은 token을 쓰게 만들지 않는가?

### 4. deduplication과 contamination check는 다른 문제다
- **deduplication** 은 주로 training corpus 내부에서 exact duplicate / near-duplicate / template duplicate를 줄이는 문제다.
- **contamination check** 는 benchmark, validation, held-out test, downstream evaluation 문서가 training data에 스며드는지 보는 문제다.
- 둘은 겹치지만 목표가 다르다.
  - dedup: 중복 때문에 모델이 같은 신호를 과하게 반복 학습하지 않게 함
  - contamination: 평가가 과장되지 않게 함
- exact string match만으로는 충분하지 않을 때가 많다.
  - boilerplate를 포함한 near-duplicate
  - benchmark prompt/paraphrase leakage
  - train/valid/test split 간 거의 같은 문서
  - 번역/요약 형태로 우회된 contamination
- 반대로 dedup을 너무 강하게 하면 자주 등장해야 하는 합법적 패턴(예: 법률 서식, 표준 문구, 코드 boilerplate 일부)까지 날릴 수 있다.

### 5. data mixture는 "문서 수 섞기"보다 "학습 신호 배분"에 가깝다
- mixture는 여러 corpus slice를 어떤 비율로 sampling할지 정하는 규칙이다.
- 여기서 자주 놓치는 점은 **문서 비율과 token 비율이 다르다** 는 것이다.
  - 긴 문서가 많은 slice는 같은 문서 수라도 더 많은 token budget을 차지한다.
  - tokenizer가 언어별로 다르게 압축하면 같은 문자 수라도 token 수가 다르다.
- 그래서 mixture는 보통 다음 축을 함께 봐야 한다.
  - 문서/샘플 비율
  - token 비율
  - 품질 가중치
  - domain coverage
  - downstream 목표와의 정합성
- 작은 고품질 도메인 데이터를 oversample할지, 큰 일반 데이터에 묻어 둘지는 이후 모델 사용 목적과 연결된다.

### 6. multilingual mixture는 "언어를 추가한다"보다 더 어렵다
- 다국어 corpus에서는 고자원 언어가 규모와 tokenizer 효율 면에서 동시에 유리한 경우가 많다.
- 예를 들어 shared tokenizer를 쓰면 어떤 언어는 한 문장을 적은 token으로 표현하고, 어떤 언어는 훨씬 더 길게 쪼개질 수 있다.
- 이 경우 같은 문서 수를 sampling해도 실제 학습에서는 특정 언어가 더 많은/적은 token budget을 차지한다.
- multilingual mixture를 설계할 때는 보통 아래를 함께 본다.
  - 언어별 문서 수와 token 수
  - tokenizer의 언어별 fragmentation 정도
  - 언어별 품질 차이와 노이즈 수준
  - cross-lingual transfer 기대치와 interference 위험
- 즉 "다국어를 넣었다"보다 중요한 질문은 **어떤 언어가 실제로 얼마나 학습되고 있는가** 다.

### 7. corpus와 tokenizer는 서로 독립이 아니라 상호작용한다
- domain-specific corpus가 많아지면 그 도메인 용어를 더 잘 보존하는 tokenizer가 필요할 수 있다.
- 반대로 tokenizer가 특정 언어/도메인을 지나치게 잘게 쪼개면, mixture ratio를 그대로 둬도 실제 gradient 신호는 한쪽으로 기울 수 있다.
- 그래서 corpus selection, tokenizer training, mixture weighting은 보통 따로 최적화하기보다 **같이 관찰하며 조정하는 루프** 로 보는 편이 안전하다.

## 수식 / 직관
- 한 slice `d`에 배정되는 대략적인 token budget을 직관적으로 쓰면 다음처럼 볼 수 있다.
  - `budget_d ≈ total_steps × tokens_per_step × mixture_share_d`
- 하지만 실제 유효 신호는 품질과 중복 정도 때문에 더 줄어들 수 있다.
  - `effective_signal_d ≈ budget_d × quality_factor_d × (1 - duplicate_rate_d)`
- tokenizer가 문장 길이를 늘리면 같은 context window 안에 담기는 정보량도 달라진다.
  - 같은 문서라도 `avg_tokens_per_doc`가 커지면 더 자주 truncate되거나, 같은 step 수에서 덜 다양한 문맥을 보게 된다.

## Common Confusion
- corpus가 크기만 하면 품질 문제를 모두 덮을 수 있다고 생각하는 실수
- dedup과 contamination check를 같은 작업으로 뭉뚱그리는 실수
- tokenizer 성능을 단순 압축률 하나로만 평가하는 실수
- multilingual shared tokenizer를 쓰면 자동으로 언어 간 공정성이 맞춰진다고 생각하는 실수
- mixture를 문서 수 기준으로만 읽고 token budget 차이를 놓치는 실수
- benchmark contamination을 exact string overlap만으로 충분히 찾을 수 있다고 생각하는 실수

## 이 단위에서 무엇을 관찰할 것인가
- corpus slice별로 언어/도메인/길이/노이즈/중복률이 어떻게 다르게 나타나는가?
- tokenizer를 바꾸면 평균 sequence length, 희귀 용어 분해 정도, 언어별 fragmentation이 어떻게 달라지는가?
- dedup 이전/이후에 실제로 줄어든 것은 문서 수인가, token 수인가, 아니면 거의 같은 문서의 반복 노출인가?
- benchmark overlap이나 split leakage는 exact duplicate보다 더 넓게 잡아야 하는가?
- mixture 비율을 문서 기준, token 기준, 품질 가중치 기준으로 읽었을 때 어떤 slice가 실제로 더 많이 학습되는가?
- domain adaptive pretraining으로 넘어가기 전에, 현재 base corpus 설계가 어느 도메인을 과소대표/과대표하고 있는가?
