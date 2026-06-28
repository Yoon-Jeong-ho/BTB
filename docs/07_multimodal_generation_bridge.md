# 07 Multimodal Generation Bridge

이 문서는 `08_multimodal_bridge/01_contrastive_alignment`의 shared embedding retrieval 감각에서 `09_multimodal`의 captioning, VQA, multimodal reasoning으로 넘어가기 위한 다리다. retrieval은 “같은 장면을 가까운 벡터로 놓는 일”이고, generation/VQA는 “이미지 정보를 token 생성 또는 답변 선택 과정에 계속 주입하는 일”이다.

## Shared embedding만으로 부족한 이유

Contrastive retrieval은 이미지 벡터와 텍스트 벡터의 거리만 잘 맞추면 된다. 하지만 captioning과 VQA는 다음을 요구한다.

- 이미지의 여러 부분을 순서대로 참고한다.
- 질문 token과 이미지 region이 상호작용한다.
- 답변이나 caption token을 생성하면서 이전 token과 시각 정보를 함께 본다.
- 실패가 retrieval failure가 아니라 grounding failure일 수 있다.

## Cross-attention vs shared embedding retrieval

- **shared embedding retrieval**: 이미지 전체와 문장 전체를 각각 하나의 벡터로 요약한 뒤 similarity를 계산한다.
- **cross-attention**: text token이 image patch/region feature를 바라보며 필요한 정보를 골라 온다.

예를 들어 “빨간 컵의 왼쪽에 무엇이 있나?”라는 질문은 전체 이미지-문장 similarity보다, `빨간 컵`, `왼쪽`, `무엇` token이 어떤 region을 보는지가 중요하다.

## 이미지 토큰은 텍스트 토큰과 같은 뜻인가?

VLM을 공부할 때 가장 헷갈리는 지점은 “이미지는 연속값인데 왜 토큰이라고 부르는가?”이다. 핵심은 **토큰이라는 말이 항상 discrete vocabulary ID를 뜻하지는 않는다**는 점이다.

| 구분 | 텍스트 토큰 | 이미지/비전 토큰 |
| --- | --- | --- |
| 출발점 | 문자열을 tokenizer가 자른 조각 | 이미지를 패치 또는 region으로 나눈 조각 |
| 입력 형태 | 정수 ID (`314`, `50256` 등) | 픽셀 패치, region feature, 또는 이미 인코딩된 feature |
| 임베딩 방식 | embedding table lookup | patch flatten/conv/linear projection 또는 vision encoder 출력 |
| 값의 성격 | finite vocabulary 중 하나 | 보통 연속 벡터, 즉 “soft token” |
| 순서 정보 | 1D positional encoding | 2D 위치 정보 또는 patch 좌표 정보 |

따라서 “이미지 토큰”은 보통 **이미지의 한 조각이 transformer sequence의 한 위치로 들어갈 수 있게 바뀐 벡터**를 뜻한다. 텍스트처럼 “고양이=정수 ID 하나”가 아니라, 예를 들어 `14x14x3` 또는 `16x16x3` 픽셀 패치를 펼친 뒤 선형 변환해서 `d_model` 차원의 벡터로 만든다. 그 벡터가 self-attention/cross-attention에서 다른 패치나 텍스트 토큰과 문맥 연산을 한다.

여기서 “토큰”이라는 이름은 **어휘 사전의 항목**이라기보다 **attention이 처리하는 sequence element**라는 뜻에 가깝다. Google의 ViT 설명도 이미지를 같은 크기의 패치로 나누고 이것을 language model에서 물려받은 용어인 token이라고 부른다고 설명한다. PaliGemma도 이미지를 SigLIP encoder가 “soft token”으로 바꾼 뒤, multimodal projector를 통해 언어 모델 입력 공간으로 보낸다.

### soft token과 discrete image token을 구분하기

- **soft image token**: vision encoder나 linear projection이 만든 연속 벡터다. 대부분의 VLM 이해 모델에서 말하는 “image token”은 여기에 가깝다.
- **discrete image token**: VQ-VAE나 image tokenizer가 codebook index로 양자화한 토큰이다. 이미지 생성 모델이나 unified next-token prediction 계열에서 자주 보인다.

초보자 입장에서는 다음처럼 기억하면 된다.

> 텍스트 토큰은 “단어 조각 ID → embedding”이고, 이미지 토큰은 “이미지 조각 → embedding”이다. 둘 다 LLM/Transformer에 들어갈 때는 sequence의 한 칸을 차지하는 벡터가 된다.

### 토크나이저 ID가 있는 “비전 토큰”은 무엇인가?

실제 모델 문서에서는 “비전 토큰”이라는 말이 세 층위로 섞여 나온다.

1. **텍스트 tokenizer vocabulary에 있는 control token**
   - 예: Qwen2-VL의 chat template에는 `<|vision_start|><|image_pad|><|vision_end|>` 같은 문자열이 들어간다.
   - 이 문자열들은 tokenizer vocabulary 안에 있는 special token이라서 실제 token ID를 가진다.
   - 하지만 `<|image_pad|>` 하나가 “이미지 전체의 모든 패치 정보”를 담는다는 뜻은 아니다. processor가 이 placeholder 위치를 보고, 별도로 계산한 image feature sequence를 그 자리에 맞춰 넣는다.

2. **모델 내부에서 LLM context 자리를 차지하는 visual token**
   - Qwen2-VL은 arbitrary resolution 이미지를 dynamic number of visual tokens로 매핑한다고 설명한다.
   - Hugging Face model card도 visual token 수를 기본 `4~16384` 범위로 보고, `min_pixels`, `max_pixels`로 예를 들어 `256~1280` token 범위를 맞출 수 있다고 안내한다.
   - 이 수는 “서로 다른 vocabulary ID가 1280개 생긴다”가 아니라, 이미지 feature가 LLM 앞에서 **1280개의 sequence position**을 차지한다는 뜻에 가깝다.

3. **API 사용량/과금에서 세는 image token**
   - Gemini API 문서는 이미지도 tokenized input으로 계산하며, 작은 이미지는 258 tokens, 큰 이미지는 `768x768` tile마다 258 tokens로 계산한다고 설명한다.
   - 이 258은 사용량·context budget을 설명하기 위한 단위이지, 사용자가 확인할 수 있는 “258개의 텍스트 vocabulary ID”라는 뜻은 아니다.

따라서 사용자가 “Qwen이나 Gemini에서 이미지 토큰이 텍스트보다 많이 먹는다”고 들은 것은 맞다. 다만 그 의미는 보통 **tokenizer가 서로 다른 이미지 단어 ID를 잔뜩 만든다**가 아니라, **이미지에서 나온 visual feature들이 LLM context 안에서 텍스트 토큰처럼 여러 자리를 차지한다**는 뜻이다.

```text
Qwen식 입력을 단순화하면:

텍스트 tokenizer:
<|vision_start|> <|image_pad|> <|vision_end|> Describe this image
        │              │
        │              └─ processor/model이 이 위치에 N개의 visual embeddings를 대응시킴
        └─ 실제 vocabulary ID가 있는 control token

모델 내부:
[vision token 1] [vision token 2] ... [vision token N] [Describe] [this] [image]
```

여기서 `<|image_pad|>`는 실제 vocabulary ID가 있는 “손잡이(handle)”이고, `vision token 1..N`은 이미지 해상도와 processor 설정에 따라 늘어나는 내부 visual token sequence라고 이해하면 가장 덜 헷갈린다.

## Google Gemma 4 12B: “이미지 토큰 없음”이 아니라 “vision encoder 없음”

2026년 Google은 Gemma 4 12B를 **unified, encoder-free multimodal model**로 소개했다. 공식 설명에 따르면 기존 multimodal model은 별도 vision/audio encoder가 입력을 먼저 번역한 뒤 LLM에 넘기는 경우가 많지만, Gemma 4 12B는 vision encoder를 가벼운 embedding module로 대체한다.

중요한 해석은 다음과 같다.

1. 이미지 자체가 마법처럼 LLM에 바로 들어가는 것은 아니다.
2. 이미지는 여전히 패치 단위로 잘리고, 각 패치는 LLM hidden dimension으로 투영된다.
3. 다만 수십 layer짜리 vision transformer encoder가 먼저 문맥화하지 않고, **single matrix multiplication + positional embedding + normalization** 정도의 가벼운 단계 뒤 LLM backbone이 시각 처리를 맡는다.
4. 그래서 “image-token-free”라기보다 정확히는 **encoder-free** 또는 **separate vision encoder-free**에 가깝다.

즉 기존 PaliGemma식 흐름이:

```text
image pixels → patch embedding → vision encoder에서 문맥화된 soft image tokens
             → projector → LLM decoder
```

라면 Gemma 4 12B식 encoder-free 흐름은:

```text
image pixels → raw patch projection + 위치 정보 + 정규화
             → LLM backbone이 직접 시각 문맥화
```

로 이해하면 된다. 이 변화는 latency와 memory footprint를 줄일 수 있지만, vision encoder가 하던 공간적 귀납편향과 초기 시각 특징 추출을 LLM backbone이 더 많이 떠안는 trade-off가 있다.

## 이 개념을 VQA와 captioning에서 어떻게 읽을까?

- 이미지 토큰 수가 많을수록 세밀한 정보를 많이 보존하지만 attention 비용이 커진다.
- 전역 이미지 벡터 하나만 쓰면 retrieval은 가능해도 위치·개수·관계 질문에서 정보가 뭉개질 수 있다.
- VQA의 count/color/location failure는 “LLM이 똑똑하지 않아서”뿐 아니라 **어떤 visual token이 어떤 해상도와 위치 정보로 들어갔는가**의 문제일 수 있다.
- encoder-free 모델은 “vision encoder가 아예 필요 없다”는 결론이라기보다, 충분한 학습과 큰 decoder backbone이 있을 때 일부 시각 처리를 decoder 쪽으로 옮길 수 있다는 실험적 방향으로 봐야 한다.

## Encoder-decoder multimodal generation

Captioning에서는 보통 다음 흐름을 생각한다.

1. vision encoder가 image feature를 만든다.
2. text decoder가 지금까지 만든 caption token을 본다.
3. decoder cross-attention이 image feature를 참조한다.
4. 다음 caption token을 생성한다.
5. stop token까지 반복한다.

따라서 caption 품질은 language fluency만이 아니라 image grounding과 decoding policy에 의해 같이 결정된다.

## VQA fusion

VQA는 captioning보다 짧은 답을 만들 수 있지만, 질문과 이미지의 결합이 더 중요하다.

- 질문이 색, 위치, 개수, 행위 중 무엇을 묻는지 분류한다.
- image feature에서 해당 evidence를 찾는다.
- answer vocabulary 또는 decoder가 답을 낸다.
- answer-type별 정확도를 따로 본다.

숫자 하나의 accuracy만 보면 color question은 잘 맞히지만 counting question은 계속 틀리는 문제를 놓친다.

## Grounding failure vs retrieval failure

- **retrieval failure**: 맞는 image-text pair를 가까이 두지 못한다.
- **grounding failure**: 이미지는 봤지만 질문/문장 속 특정 대상, 위치, 관계를 잘못 연결한다.
- **generation failure**: 시각 정보는 맞게 봤지만 decoder가 반복, hallucination, generic answer로 무너진다.

`09_multimodal`에서는 이 세 failure를 같은 것으로 취급하지 말고, figure와 예시 표에서 분리해 적는다.

## `09_multimodal`에 들어가기 전 체크리스트

- cross-attention이 왜 retrieval similarity보다 더 세밀한 연결인지 설명할 수 있다.
- 이미지 토큰이 discrete vocabulary ID가 아니라, 많은 VLM에서는 연속적인 soft token이라는 점을 설명할 수 있다.
- PaliGemma식 vision encoder 흐름과 Gemma 4 12B식 encoder-free 흐름의 차이를 구분할 수 있다.
- caption hallucination과 retrieval mismatch를 구분할 수 있다.
- VQA answer-type breakdown이 왜 필요한지 말할 수 있다.
- qualitative panel에서 이미지, 질문/캡션, 모델 출력, 실패 원인을 함께 볼 수 있다.

## 최소 실험 아이디어

- 같은 이미지에 retrieval caption, generated caption, VQA question을 하나씩 붙인다.
- 실패를 `retrieval`, `grounding`, `generation` 중 하나로 라벨링한다.
- held-out 또는 adversarial 예시 2~4개를 만들어 “학습 데이터에서는 맞지만 새 조합에서는 틀리는” 경우를 기록한다.

## 참고 출처

- Google Research Blog, [Scaling Vision with Sparse Mixture of Experts](https://research.google/blog/scaling-vision-with-sparse-mixture-of-experts/): ViT가 이미지를 patch token sequence로 보는 기본 직관.
- Google Developers Blog, [Gemma explained: PaliGemma architecture](https://developers.googleblog.com/ko/gemma-explained-paligemma-architecture/): PaliGemma의 SigLIP vision tower, soft token, projector 흐름.
- Google Blog, [Introducing Gemma 4 12B](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/): vision/audio encoder를 제거한 unified encoder-free 구조 소개.
- Google Developers Blog, [Gemma 4 12B: The Developer Guide](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/): raw `48x48` patch projection, 위치 lookup, LLM backbone 직접 처리 설명.
