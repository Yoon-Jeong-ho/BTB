# BTB 커리큘럼 재설계 디자인 스펙

## 1. 배경과 문제 정의

BTB 저장소의 현재 방향은 `기초 ML -> NLP -> Multimodal` 순서로 실험을 축적하는 데에는 도움이 되지만, 학습자 관점에서는 다음 문제가 남아 있다.

1. `01_ml`은 비교적 구체적인 코드/이론/결과 구조를 갖고 있지만 `02_nlp`, `03_multimodal`은 아직 "태스크 목록"에 가까워 학습 계단이 약하다.
2. 활성화 함수(activation), 그래디언트(gradient), 역전파(backpropagation), GPU 메모리, 토크나이저(tokenizer), 임베딩(embedding), 어텐션(attention), 멀티모달 정렬(alignment) 같은 핵심 기초가 독립된 학습 축으로 드러나지 않는다.
3. 현재 문서 구조는 읽는 사람을 단계적으로 끌고 가기보다는, 이미 알고 있는 사람이 실험 산출물을 훑는 데 더 적합하다.
4. 일부 문서는 문체와 구조가 기계적으로 느껴질 수 있으며, "왜 이것을 배우는지"와 "결과가 왜 이렇게 나왔는지"를 설명하는 연결 문장이 부족하다.
5. 향후 자동화를 붙이더라도, 먼저 학습 단위 자체가 사람에게 읽히는 구조여야 한다.

이 설계의 목적은 BTB를 단순 실험 저장소가 아니라 **기초를 메우면서 연구 실습으로 올라가는 한글 중심 학습 프로그램**으로 재구성하는 것이다.

## 2. 목표

### 핵심 목표
- 내부 원리 이해를 중심으로 한 학습 계단을 만든다.
- 작은 구현, 실제 프레임워크 실습, 결과 분석, 회고가 한 단위 안에서 닫히도록 만든다.
- 사용자가 읽으면서 공부할 수 있고, 다른 사람도 그대로 따라올 수 있는 문서형 결과물을 남긴다.
- 실험 자동화는 지원하되, 사람이 읽을 수 있는 설명과 분석을 훼손하지 않는다.

### 성공 기준
- 저장소 최상위 구조만 봐도 학습 순서가 명확해야 한다.
- 각 학습 단위는 같은 계약(contract)을 따라야 한다.
- `00_foundations`와 bridge 구간이 명시적으로 존재해야 한다.
- 한국어 문서만 읽어도 각 단위의 목적, 선행 개념, 실습, 결과 해석을 이해할 수 있어야 한다.
- 첫 1~2개의 gold-standard 단위가 이후 전체 확장의 기준 템플릿 역할을 해야 한다.

## 3. 비목표(Non-goals)

- 기존 내용을 한 번에 모두 갈아엎지 않는다.
- 초기 단계에서 전체 자동화까지 완성하지 않는다.
- 파일명, 함수명, CLI 인자까지 한국어로 바꾸지 않는다.
- 모든 단위를 동일한 깊이로 즉시 완성하려 하지 않는다.

## 4. 설계 원칙

1. **기초 우선**: 상위 태스크보다 내부 작동 원리와 시스템 감각을 먼저 세운다.
2. **계단식 구조**: Foundation -> Bridge -> Applied Track 순서를 인덱싱으로 드러낸다.
3. **혼합형 학습**: scratch 구현과 PyTorch/Hugging Face 기반 실습을 모두 포함한다.
4. **문서 우선**: README/THEORY/분석/회고를 통해 코드보다 먼저 학습 흐름이 보이게 한다.
5. **증거 우선**: 자동 생성 보고서도 숫자, 그림, 실패 사례를 근거로 서술한다.
6. **한글 우선**: 학습자 대상 문서는 기본적으로 한국어로 작성하되, 핵심 영문 기술 용어는 병기한다.
7. **점진적 마이그레이션**: 읽기 경험 개선 -> 구조 재편 -> 표준 유닛 구축 -> 자동화 확장 순으로 간다.

## 5. 목표 정보 구조(Architecture)

### 5.1 최상위 디렉터리 구조

```text
BTB/
├── 00_foundations/
├── 01_ml/
├── 02_nlp_bridge/
├── 03_nlp/
├── 04_multimodal_bridge/
├── 05_multimodal/
├── artifacts/
├── data/
├── docs/
├── reports/
├── runs/
└── scripts/
```

### 5.2 왜 bridge도 top-level 인덱스를 가져야 하는가

- GitHub와 파일 탐색기에서 순서가 안정적으로 보인다.
- bridge가 "부록"이 아니라 필수 학습 구간이라는 점이 드러난다.
- 사용자가 `01_ml`에서 `03_nlp`로 바로 점프하지 않고, 그 사이의 개념 다리를 자연스럽게 거치게 된다.

### 5.3 각 top-level의 역할

- `00_foundations/`: 공통 기초. 수학 감각, 텐서, 활성화 함수, 손실 함수, 그래디언트, 역전파, 최적화, 일반화, GPU/runtime, 텍스트/멀티모달 기초 표현을 다룬다.
- `01_ml/`: 표형 데이터 기반의 첫 응용 트랙. 평가 지표, 에러 분석, 해석, 실험 discipline을 익힌다.
- `02_nlp_bridge/`: 토크나이징, 임베딩, 시퀀스 모델링, 어텐션, transformer block을 실습 중심으로 연결한다.
- `03_nlp/`: 텍스트 분류, NER, MRC를 수행하며 scratch mini-lab + framework lab + 실패 분석을 갖춘다.
- `04_multimodal_bridge/`: contrastive learning, alignment, cross-attention, retrieval vs generation, compute cost를 다룬다.
- `05_multimodal/`: retrieval, captioning, VQA를 실습하고 grounding/shortcut/failure analysis를 남긴다.

## 6. 6~8주 Core Phase 설계

Core phase는 "얇은 읽기 목록"이 아니라, 매주 학습 결과물이 남는 **고밀도 학습 스프린트**로 설계한다.

### 6.1 권장 기간
- 기본: 8주
- Week 8은 선택형 integration mini-project로 운영 가능

### 6.2 주간 리듬
각 주차는 아래 리듬을 기본으로 한다.

- 이론 lesson 2~3개
- 실습 2개
  - scratch mini-lab 1개
  - framework/system lab 1개
- 분석 산출물 1개
- 짧은 회고 1개

### 6.3 주차 구성

1. **Week 1 — 수학 감각 + 텐서 직관**
   - 벡터/행렬, 내적, broadcasting, shape 읽기, batch 차원, 수치 안정성 기초
2. **Week 2 — Forward pass, activation, loss**
   - 선형층, 비선형성, logits, softmax, cross-entropy, regression loss
3. **Week 3 — Gradients, backpropagation, optimization**
   - chain rule, autograd 검증, SGD vs Adam, learning rate 직관, exploding/vanishing gradients
4. **Week 4 — Regularization, normalization, training dynamics**
   - overfitting, dropout, weight decay, batch/layer norm, early stopping, calibration
5. **Week 5 — GPU, memory, runtime mechanics**
   - 무엇이 VRAM을 차지하는가, training vs inference 메모리, mixed precision, gradient accumulation, batch-size tradeoff
6. **Week 6 — Text representation bridge**
   - tokenization, subword, embedding lookup, positional information, attention intuition, transformer block anatomy
7. **Week 7 — Multimodal representation bridge**
   - image encoder basics, contrastive alignment, retrieval vs generation, cross-attention, grounding failure
8. **Week 8 — Integration mini-project (optional but recommended)**
   - 작은 end-to-end 프로젝트와 결과/해석/회고 작성

### 6.4 B 수준 기초 보강 장치
각 lesson에는 다음 보조 장치를 둔다.

- `prereq refresh box`: 흐린 수학/시스템 개념 재학습 링크
- `micro drill`: 5~15분 분량의 짧은 확인 문제
- `common confusion note`: 흔한 오해를 바로잡는 짧은 노트
- `system walkthrough`: CPU RAM / GPU VRAM / dataloader / checkpoint 흐름 설명

## 7. 학습 단위(Unit) 계약

각 unit은 사람이 읽어도 이해되고, 자동화가 붙어도 품질이 유지되도록 같은 구조를 따른다.

### 7.1 기본 산출물
- 이론 노트
- 작은 구현(scratch)
- 실제 프레임워크 실습
- 결과 분석
- 실패 분석
- 짧은 회고

### 7.2 권장 파일 구조

```text
<unit>/
├── README.md
├── THEORY.md
├── PREREQS.md            # 또는 README/THEORY 안의 prereq 박스
├── lesson.yaml
├── scratch_lab.py        # 또는 notebook
├── framework_lab.py
├── analysis.py
├── analysis.md
├── reflection.md
└── artifacts/
```

### 7.3 파일별 역할
- `README.md`: 이 unit가 무엇을 가르치며, 왜 중요한지, 무엇을 남겨야 하는지 설명하는 입구 문서
- `THEORY.md`: 핵심 개념 설명, 공식, 그림, common confusion 정리
- `PREREQS.md`: 필요한 선행 개념을 빠르게 복습하는 문서
- `scratch_lab.py`: 내부 메커니즘을 직접 확인하는 작은 구현
- `framework_lab.py`: PyTorch / Hugging Face / 시스템 도구 기반 실습
- `analysis.py`: 표/그림/요약 수치 생성
- `analysis.md`: 결과가 왜 나왔는지 설명하는 해석 문서
- `reflection.md`: 여전히 애매한 점, 다시 볼 개념, 다음 unit 연결점을 정리하는 학습자 메모

## 8. 자동화 설계(Data Flow + Tooling Contract)

### 8.1 lesson.yaml 계약
각 unit은 최소한 아래 메타데이터를 가져야 한다.

- objective
- prereqs
- key_terms
- datasets / models / tools
- required_outputs
- expected_figures
- analysis_questions
- runtime_observation_hooks
- report_style

### 8.2 실행 데이터 흐름(Data Flow)

```text
lesson.yaml
  -> run command
  -> scratch/framework 실습 실행
  -> metrics / figures / runtime stats 저장
  -> analysis scaffold 생성
  -> summary / analysis / reflection 초안 생성
  -> 사람이 읽고 다듬음
```

### 8.3 자동화가 반드시 보장해야 하는 것
- 단일 명령으로 실습을 재현할 수 있어야 한다.
- figure, metrics, runtime 관측치가 누락되면 명시적으로 실패해야 한다.
- 결과 보고서 초안은 "왜 이런 결과가 나왔는가"를 쓰도록 질문형 프롬프트를 포함해야 한다.
- 결과 문서는 관련 이론 섹션으로 역링크(backlink)를 제공해야 한다.

### 8.4 GPU / 시스템 관측 훅
선택적으로 아래를 기록할 수 있어야 한다.
- batch size
- dtype(fp32/fp16/bf16)
- train vs inference memory
- `torch.cuda.max_memory_allocated()` / `reserved()`
- throughput(samples/sec)
- `nvidia-smi` 스냅샷 또는 동등한 시스템 관측치

## 9. 문체와 언어 정책

### 9.1 기본 정책
- README, THEORY, PREREQS, 분석 문서, 회고 문서는 **한글 우선**으로 작성한다.
- 기술 용어는 필요할 때 한글 + 영어 병기를 사용한다.
  - 예: 역전파(backpropagation), 어텐션(attention), 가중치 감쇠(weight decay)
- 코드, 함수명, 파일명, config key, CLI 인자는 영어를 유지한다.

### 9.2 문체 정책
- 짧고 명시적인 문장을 사용한다.
- 일반론과 공허한 칭찬 문장을 줄인다.
- 근거가 있는 주장만 쓴다.
- "좋아졌다" 대신 "macro F1이 0.71에서 0.77로 올랐다"처럼 쓴다.
- 자동 생성 문서는 evidence-first 구조를 가져야 하며, generic AI 요약을 피해야 한다.

## 10. 마이그레이션 계획

### Phase 0 — Navigation / Policy 정리
- 루트 README, 프로그램 맵, 언어/문체 규칙을 먼저 정리한다.
- 목표: 지금 레포를 읽는 순간부터 덜 혼란스럽게 만든다.

### Phase 1 — 새 뼈대 생성
- `00_foundations/`, `02_nlp_bridge/`, `04_multimodal_bridge/` 생성
- 각 디렉터리에 한글 입구 README 작성

### Phase 2 — 기존 트랙 재인덱싱
- `02_nlp/` -> `03_nlp/`
- `03_multimodal/` -> `05_multimodal/`
- 링크와 참조를 전면 수정

### Phase 3 — Gold-standard unit 1~2개 완성
- 우선순위:
  1. `00_foundations/01_tensor_shapes`
  2. `00_foundations/05_gpu_memory_runtime`
- 이 둘을 이후 모든 단위의 품질 기준 템플릿으로 삼는다.

### Phase 4 — Unit contract 확장
- NLP, 멀티모달 전 구간에 README/THEORY/PREREQS/scratch/framework/analysis/reflection 구조를 확장한다.

### Phase 5 — 자동화 도입
- 구조가 안정화된 후 `lesson.yaml`, 단일 실행 엔트리, report scaffold, runtime hooks를 붙인다.

## 11. 첫 Gold-standard 단위 권장안

### 11.1 00_foundations/01_tensor_shapes
목적:
- 텐서 shape, broadcasting, batch 차원, matmul 흐름을 몸으로 익힌다.

필수 포함 요소:
- shape 읽기 이론 노트
- NumPy/PyTorch scratch mini-lab
- shape mismatch 실패 사례 분석
- 다음 단위(backprop / attention)로 이어지는 연결 설명

### 11.2 00_foundations/05_gpu_memory_runtime
목적:
- 무엇이 GPU 메모리를 차지하는지, training과 inference가 왜 다른지 이해한다.

필수 포함 요소:
- VRAM 구성요소 설명
- shape/dtype 기반 메모리 계산 mini-lab
- 실제 GPU 관측 실험
- batch size / dtype / grad accumulation 변화에 대한 해석

이 두 단위를 먼저 만드는 이유는 사용자가 현재 가장 부족하다고 느끼는 부분을 가장 빠르게 보완할 수 있기 때문이다.

## 12. 위험요소와 대응(Error Handling / Recovery)

### 위험 1: 구조 개편이 커서 진행이 흐려짐
대응:
- 먼저 문서와 gold-standard unit만 만든다.
- 전체 확장은 표준이 검증된 뒤 진행한다.

### 위험 2: 자동화가 문서를 기계적으로 만듦
대응:
- 자동화는 초안만 만들고, 분석/회고는 사람이 검토하고 다듬도록 설계한다.
- evidence-first, short-sentence 규칙을 template에 포함한다.

### 위험 3: GPU가 없는 환경에서 system lesson이 막힘
대응:
- CPU fallback 설명과 이론 모드를 제공한다.
- GPU 관측치가 없을 경우, 메모리 추정과 예시 스냅샷 기반 분석 문서를 사용한다.

### 위험 4: 링크/번호 이동으로 참조가 깨짐
대응:
- Phase 2에서 link check와 index 검증을 별도 수행한다.

## 13. 검증 계획(Testing / Verification)

이 설계가 구현 단계에서 만족해야 할 검증 기준은 아래와 같다.

1. top-level README와 program map이 새 번호 체계를 정확히 반영한다.
2. gold-standard unit 2개가 동일한 unit contract를 만족한다.
3. README/THEORY/analysis/reflection가 모두 한글 중심 문체로 작성된다.
4. unit 실행 명령이 metrics/figures/runtime stats/report scaffold를 누락 없이 생성한다.
5. 링크 검증 스크립트 또는 수동 검토로 깨진 참조가 없음을 확인한다.
6. GPU가 없는 경우에도 최소한 이론 + scratch + 대체 분석 경로로 unit를 완주할 수 있어야 한다.

## 14. 최종 권고

BTB는 전면 재작성보다, **인덱싱된 foundation-heavy ladder를 먼저 세우고**, 그 위에 **문서형 gold-standard unit를 1~2개 완성한 뒤**, 이후 NLP/멀티모달로 확장하는 방식이 가장 안전하고 효과적이다.

이 설계는 다음을 동시에 만족시키는 것을 목표로 한다.
- 사용자의 부족한 기초를 직접 메우는 학습 계단
- 읽으면서 배울 수 있는 한글 중심 문서 구조
- 실험, 분석, 회고가 닫힌 실습 단위
- 향후 자동화 가능한 메타데이터와 실행 인터페이스
