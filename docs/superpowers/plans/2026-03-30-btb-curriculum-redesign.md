# BTB Curriculum Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** BTB를 인덱싱된 한글 중심 학습 사다리로 재구성하고, 두 개의 gold-standard foundations unit와 최소 자동화 스캐폴드를 추가해 이후 NLP/멀티모달 확장의 기준선을 만든다.

**Architecture:** 먼저 루트 탐색 경험과 top-level 구조를 고정한 뒤, bridge가 포함된 번호 체계를 적용한다. 그 다음 공통 unit contract와 shared templates를 만들고, `00_foundations`의 대표 unit 두 개(`01_tensor_shapes`, `05_gpu_memory_runtime`)를 gold-standard로 구현한다. 마지막으로 `lesson.yaml` 기반 실행 스캐폴드와 링크 검증 스크립트를 추가해 문서형 학습 흐름과 자동화를 연결한다.

**Tech Stack:** Markdown, Python 3, `unittest`, shell/git, NumPy, PyTorch, 기존 BTB 디렉터리 규약

---

## File Structure Map

### Create
- `00_foundations/README.md`
- `02_nlp_bridge/README.md`
- `04_multimodal_bridge/README.md`
- `00_shared/templates/foundation_readme_template.md`
- `00_shared/templates/foundation_theory_template.md`
- `00_shared/templates/foundation_analysis_template.md`
- `00_shared/templates/foundation_reflection_template.md`
- `00_foundations/01_tensor_shapes/{README.md,THEORY.md,PREREQS.md,lesson.yaml,scratch_lab.py,framework_lab.py,analysis.py,analysis.md,reflection.md,artifacts/.gitkeep}`
- `00_foundations/05_gpu_memory_runtime/{README.md,THEORY.md,PREREQS.md,lesson.yaml,scratch_lab.py,framework_lab.py,analysis.py,analysis.md,reflection.md,artifacts/.gitkeep}`
- `scripts/_lesson_metadata.py`
- `scripts/run_lesson.py`
- `scripts/build_lesson_report.py`
- `scripts/check_curriculum_links.py`
- `tests/test_curriculum_topology.py`
- `tests/test_reindexed_tracks.py`
- `tests/test_shared_templates.py`
- `tests/test_foundations_unit_contract.py`
- `tests/test_lesson_runner_contract.py`
- `tests/test_curriculum_links.py`

### Modify
- `.gitignore`
- `README.md`
- `docs/00_program_map.md`
- `docs/01_experiment_playbook.md`
- `00_shared/README.md`
- `scripts/README.md`

### Move
- `02_nlp/` → `03_nlp/`
- `03_multimodal/` → `05_multimodal/`

---

### Task 1: Lock the top-level curriculum ladder and root navigation

**Files:**
- Create: `00_foundations/README.md`, `02_nlp_bridge/README.md`, `04_multimodal_bridge/README.md`, `tests/test_curriculum_topology.py`
- Modify: `.gitignore`, `README.md`, `docs/00_program_map.md`
- Test: `tests/test_curriculum_topology.py`

- [ ] **Step 1: Write the failing topology test**

```python
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestCurriculumTopology(unittest.TestCase):
    def test_root_readme_mentions_new_ladder(self) -> None:
        text = (ROOT / 'README.md').read_text(encoding='utf-8')
        for rel in [
            '00_foundations',
            '01_ml',
            '02_nlp_bridge',
            '03_nlp',
            '04_multimodal_bridge',
            '05_multimodal',
        ]:
            self.assertIn(rel, text)

    def test_program_map_mentions_foundations_and_bridges(self) -> None:
        text = (ROOT / 'docs' / '00_program_map.md').read_text(encoding='utf-8')
        self.assertIn('00_foundations', text)
        self.assertIn('02_nlp_bridge', text)
        self.assertIn('04_multimodal_bridge', text)
        self.assertIn('한글', text)

    def test_new_entry_dirs_have_readmes(self) -> None:
        for rel in ['00_foundations', '02_nlp_bridge', '04_multimodal_bridge']:
            self.assertTrue((ROOT / rel / 'README.md').exists(), f'missing {rel}/README.md')

    def test_superpowers_artifacts_are_ignored(self) -> None:
        text = (ROOT / '.gitignore').read_text(encoding='utf-8')
        self.assertIn('.superpowers/', text)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run the test to verify the current repo fails**

Run: `python -m unittest tests/test_curriculum_topology.py -v`  
Expected: FAIL because the new directories and root-doc references do not exist yet.

- [ ] **Step 3: Create the directories and rewrite the root docs**

```bash
mkdir -p 00_foundations 02_nlp_bridge 04_multimodal_bridge
python - <<'PY'
from pathlib import Path

# .gitignore
path = Path('.gitignore')
text = path.read_text(encoding='utf-8')
if '.superpowers/' not in text:
    text = text.rstrip() + '\n\n.superpowers/\n'
path.write_text(text, encoding='utf-8')

# README.md
Path('README.md').write_text('''# BTB

## NLP 바보에서 박사

이 저장소는 `00_foundations -> 01_ml -> 02_nlp_bridge -> 03_nlp -> 04_multimodal_bridge -> 05_multimodal` 순서로 올라가며, 한글 문서를 읽으면서 실습과 분석까지 따라갈 수 있게 만드는 학습 저장소다.

## 학습 순서

1. [00_foundations/README.md](00_foundations/README.md)
2. [01_ml/README.md](01_ml/README.md)
3. [02_nlp_bridge/README.md](02_nlp_bridge/README.md)
4. [03_nlp/README.md](03_nlp/README.md)
5. [04_multimodal_bridge/README.md](04_multimodal_bridge/README.md)
6. [05_multimodal/README.md](05_multimodal/README.md)

## 문서 원칙

- 학습자 대상 문서는 한글 우선으로 작성한다.
- 코드와 파일명은 영어를 유지한다.
- 각 단위는 이론, 실습, 분석, 회고를 함께 남긴다.
''', encoding='utf-8')

# docs/00_program_map.md
Path('docs/00_program_map.md').write_text('''# 00 Program Map

## 목표

BTB는 `00_foundations -> 01_ml -> 02_nlp_bridge -> 03_nlp -> 04_multimodal_bridge -> 05_multimodal` 순서로 올라가는 한글 중심 학습 사다리다.

## 계층 구조

1. `00_foundations`: 텐서, activation, loss, gradient, backpropagation, optimizer, GPU/runtime, tokenizer, attention, multimodal basics
2. `01_ml`: metrics, error analysis, interpretation, experiment discipline
3. `02_nlp_bridge`: tokenization, embedding, sequence modeling, attention, transformer block
4. `03_nlp`: text classification, NER, MRC
5. `04_multimodal_bridge`: contrastive learning, alignment, retrieval vs generation, cross-attention
6. `05_multimodal`: retrieval, captioning, VQA

## 언어 정책

- README, THEORY, PREREQS, 분석/회고 문서는 한글 우선
- 핵심 기술 용어는 영어를 병기
''', encoding='utf-8')

Path('00_foundations/README.md').write_text('''# 00 Foundations

이 트랙은 상위 태스크로 가기 전에 필요한 공통 기초를 메우는 구간이다.

## 다루는 축

- tensor shape / broadcasting
- activation / loss / logits
- gradient / backpropagation / optimization
- GPU memory / runtime mechanics
- tokenization / embedding / attention
- multimodal representation basics
''', encoding='utf-8')

Path('02_nlp_bridge/README.md').write_text('''# 02 NLP Bridge

이 구간은 `01_ml`과 `03_nlp` 사이의 개념 다리다.

## 목적

- tokenization과 subword 분해를 이해한다.
- embedding과 positional information을 이해한다.
- attention과 transformer block을 NLP 실습 전에 감각적으로 익힌다.
''', encoding='utf-8')

Path('04_multimodal_bridge/README.md').write_text('''# 04 Multimodal Bridge

이 구간은 `03_nlp`와 `05_multimodal` 사이의 개념 다리다.

## 목적

- contrastive alignment를 이해한다.
- retrieval와 generation의 차이를 이해한다.
- cross-attention과 grounding failure를 멀티모달 실습 전에 익힌다.
''', encoding='utf-8')
PY
```

- [ ] **Step 4: Re-run the topology test**

Run: `python -m unittest tests/test_curriculum_topology.py -v`  
Expected: PASS with 4 tests.

- [ ] **Step 5: Commit the new root ladder**

```bash
git add .gitignore README.md docs/00_program_map.md 00_foundations/README.md 02_nlp_bridge/README.md 04_multimodal_bridge/README.md tests/test_curriculum_topology.py
git commit -F - <<'EOF'
Expose the new study ladder before moving any track content

Readers need to see the numbered curriculum shape immediately,
so this change adds the top-level foundation and bridge entry
points, updates the root navigation docs, and ignores temporary
brainstorm browser artifacts.

Constraint: The repo must stay readable while migration happens in phases
Rejected: Rename every track first | would leave root navigation broken during the transition
Confidence: high
Scope-risk: moderate
Directive: Keep the published top-level numbering stable once downstream docs start linking to it
Tested: python -m unittest tests/test_curriculum_topology.py -v
Not-tested: Cross-track links that depend on later directory renames
EOF
```

---

### Task 2: Renumber the existing NLP and multimodal tracks

**Files:**
- Create: `tests/test_reindexed_tracks.py`
- Modify: `README.md`, `docs/00_program_map.md`, `scripts/README.md`
- Move: `02_nlp/` → `03_nlp/`, `03_multimodal/` → `05_multimodal/`
- Test: `tests/test_reindexed_tracks.py`

- [ ] **Step 1: Write the failing reindexing test**

```python
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestReindexedTracks(unittest.TestCase):
    def test_new_track_dirs_exist(self) -> None:
        self.assertTrue((ROOT / '03_nlp').exists())
        self.assertTrue((ROOT / '05_multimodal').exists())

    def test_old_track_dirs_are_gone(self) -> None:
        self.assertFalse((ROOT / '02_nlp').exists())
        self.assertFalse((ROOT / '03_multimodal').exists())

    def test_stage_readmes_survive_move(self) -> None:
        for rel in [
            '03_nlp/01_text_classification/README.md',
            '03_nlp/02_named_entity_recognition/README.md',
            '03_nlp/03_machine_reading_comprehension/README.md',
            '05_multimodal/01_image_text_retrieval/README.md',
            '05_multimodal/02_image_captioning/README.md',
            '05_multimodal/03_visual_question_answering/README.md',
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run the reindexing test and verify it fails**

Run: `python -m unittest tests/test_reindexed_tracks.py -v`  
Expected: FAIL because the directories still live under the old names.

- [ ] **Step 3: Move the directories and update root examples**

```bash
git mv 02_nlp 03_nlp
git mv 03_multimodal 05_multimodal
python - <<'PY'
from pathlib import Path

replacements = {
    '02_nlp/': '03_nlp/',
    '03_multimodal/': '05_multimodal/',
}
for rel in ['README.md', 'docs/00_program_map.md', 'scripts/README.md']:
    path = Path(rel)
    text = path.read_text(encoding='utf-8')
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding='utf-8')
PY
```

Use this `scripts/README.md` body:

````markdown
# Scripts

이 폴더는 학습/평가/검증 스크립트를 모으는 공간이다.

```text
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode scratch
python scripts/run_lesson.py --unit 00_foundations/05_gpu_memory_runtime --mode framework
python scripts/check_curriculum_links.py
python scripts/train.py --track 03_nlp --stage 01_text_classification --config path/to/config.yaml
python scripts/eval.py --run-dir runs/03_nlp/01_text_classification/<run_id>
```
````

- [ ] **Step 4: Run the reindexing test and the ML regression test**

Run: `python -m unittest tests/test_reindexed_tracks.py tests/test_01_ml_report_contract.py -v`  
Expected: PASS. The moved directories exist and ML remains untouched.

- [ ] **Step 5: Commit the renumbering move**

```bash
git add README.md docs/00_program_map.md scripts/README.md tests/test_reindexed_tracks.py 03_nlp 05_multimodal
git commit -F - <<'EOF'
Preserve the study sequence when expanding beyond ML

The curriculum now needs explicit bridge layers, so the NLP and
multimodal tracks move to numbered positions that leave room for
those prerequisites without changing their internal lesson names.

Constraint: Existing stage content should survive the move unchanged in this pass
Rejected: Nest bridge content inside each applied track | weakens the top-level learning ladder
Confidence: high
Scope-risk: moderate
Directive: Update future docs to the new 03_nlp / 05_multimodal paths only
Tested: python -m unittest tests/test_reindexed_tracks.py tests/test_01_ml_report_contract.py -v
Not-tested: Markdown links outside the touched root docs
EOF
```

---

### Task 3: Define the shared unit contract and templates

**Files:**
- Create: `00_shared/templates/foundation_readme_template.md`, `00_shared/templates/foundation_theory_template.md`, `00_shared/templates/foundation_analysis_template.md`, `00_shared/templates/foundation_reflection_template.md`, `tests/test_shared_templates.py`
- Modify: `00_shared/README.md`, `docs/01_experiment_playbook.md`
- Test: `tests/test_shared_templates.py`

- [ ] **Step 1: Write the failing template test**

```python
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = ROOT / '00_shared' / 'templates'


class TestSharedTemplates(unittest.TestCase):
    def test_foundation_templates_exist(self) -> None:
        for name in [
            'foundation_readme_template.md',
            'foundation_theory_template.md',
            'foundation_analysis_template.md',
            'foundation_reflection_template.md',
        ]:
            self.assertTrue((TEMPLATE_ROOT / name).exists(), name)

    def test_shared_readme_mentions_unit_contract(self) -> None:
        text = (ROOT / '00_shared' / 'README.md').read_text(encoding='utf-8')
        self.assertIn('README/THEORY/PREREQS/scratch/framework/analysis/reflection', text)

    def test_playbook_mentions_lesson_yaml(self) -> None:
        text = (ROOT / 'docs' / '01_experiment_playbook.md').read_text(encoding='utf-8')
        self.assertIn('lesson.yaml', text)
        self.assertIn('reflection.md', text)
        self.assertIn('runtime', text)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run the template test to confirm the contract is missing**

Run: `python -m unittest tests/test_shared_templates.py -v`  
Expected: FAIL because the foundation templates and updated playbook text do not exist.

- [ ] **Step 3: Add the templates and update the shared docs**

Use this `00_shared/README.md` body:

```markdown
# 00 Shared

공통 템플릿과 실험 규약을 두는 공간이다.

## 공통 Unit Contract

모든 학습 단위는 가능하면 아래 흐름을 유지한다.

- `README.md`
- `THEORY.md`
- `PREREQS.md`
- `scratch_lab.py`
- `framework_lab.py`
- `analysis.md`
- `reflection.md`

## 템플릿 목록

- `foundation_readme_template.md`
- `foundation_theory_template.md`
- `foundation_analysis_template.md`
- `foundation_reflection_template.md`
```

Use these template bodies:

```markdown
# {{unit_title}}

## 왜 이 단위를 배우는가

## 이번 단위에서 남길 것

## 선행 개념

## 다음 단위와의 연결
```

```markdown
# {{unit_title}} 이론 노트

## 핵심 개념

## 수식 / 직관

## Common Confusion
```

```markdown
# {{unit_title}} 분석

## 관측 결과

## 해석

## 실패 사례
```

```markdown
# {{unit_title}} 회고

- 이번에 이해한 것:
- 아직 애매한 것:
- 다음에 다시 볼 것:
```
```

Update `docs/01_experiment_playbook.md` with this section:

````markdown
## 2. 모든 실험/학습 단위가 남겨야 하는 파일

```text
<unit>/
├── lesson.yaml
├── scratch_lab.py
├── framework_lab.py
├── analysis.md
├── reflection.md
└── artifacts/
```

- `lesson.yaml`: 목표, 선행 개념, 출력 계약, 분석 질문
- `analysis.md`: 결과 해설 문서
- `reflection.md`: 학습자 관점 회고
- runtime 관련 실습은 GPU/CPU 관측치를 함께 남긴다.
````

- [ ] **Step 4: Run the template test again**

Run: `python -m unittest tests/test_shared_templates.py -v`  
Expected: PASS with 3 tests.

- [ ] **Step 5: Commit the shared contract**

```bash
git add 00_shared/README.md 00_shared/templates/foundation_readme_template.md 00_shared/templates/foundation_theory_template.md 00_shared/templates/foundation_analysis_template.md 00_shared/templates/foundation_reflection_template.md docs/01_experiment_playbook.md tests/test_shared_templates.py
git commit -F - <<'EOF'
Define a document-heavy lesson contract before cloning new units

The redesign depends on repeatable, readable study units, so this
change formalizes the shared contract and adds foundation templates
before any gold-standard lesson is authored.

Constraint: New units must stay readable to humans before automation is added
Rejected: Reuse the ML-only templates as-is | they do not cover prereqs, reflection, or system lessons
Confidence: high
Scope-risk: narrow
Directive: Add new study units by extending these templates rather than inventing ad hoc layouts
Tested: python -m unittest tests/test_shared_templates.py -v
Not-tested: Real-world readability of the first authored units
EOF
```

---

### Task 4: Build `00_foundations/01_tensor_shapes`

**Files:**
- Create: `00_foundations/01_tensor_shapes/{README.md,THEORY.md,PREREQS.md,lesson.yaml,scratch_lab.py,framework_lab.py,analysis.py,analysis.md,reflection.md,artifacts/.gitkeep}`
- Create: `tests/test_foundations_unit_contract.py`
- Test: `tests/test_foundations_unit_contract.py`

- [ ] **Step 1: Write the failing foundations contract test**

```python
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONS = ROOT / '00_foundations'
REQUIRED = [
    'README.md',
    'THEORY.md',
    'PREREQS.md',
    'lesson.yaml',
    'scratch_lab.py',
    'framework_lab.py',
    'analysis.py',
    'analysis.md',
    'reflection.md',
    'artifacts',
]


class TestFoundationsUnitContract(unittest.TestCase):
    def test_tensor_shapes_unit_has_required_files(self) -> None:
        unit = FOUNDATIONS / '01_tensor_shapes'
        for rel in REQUIRED:
            self.assertTrue((unit / rel).exists(), rel)

    def test_tensor_shapes_metadata_mentions_outputs(self) -> None:
        text = (FOUNDATIONS / '01_tensor_shapes' / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('required_outputs:', text)
        self.assertIn('analysis_questions:', text)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run the contract test and verify it fails**

Run: `python -m unittest tests/test_foundations_unit_contract.py -v`  
Expected: FAIL because `00_foundations/01_tensor_shapes` does not exist yet.

- [ ] **Step 3: Create the unit with Korean docs and minimal runnable labs**

Use this `lesson.yaml`:

```yaml
objective: 텐서 shape, broadcasting, batch 차원, matmul 흐름을 몸으로 익힌다.
prereqs:
  - 벡터와 행렬 표기
  - 행렬 곱의 조건
key_terms:
  - tensor
  - shape
  - broadcasting
required_outputs:
  - scratch metrics json
  - framework metrics json
  - analysis markdown
analysis_questions:
  - 어떤 연산에서 shape mismatch가 났는가?
  - broadcasting이 편하지만 위험한 이유는 무엇인가?
```

Use this `scratch_lab.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'


def run() -> None:
    a = np.arange(6).reshape(2, 3)
    b = np.arange(12).reshape(3, 4)
    c = a @ b
    mismatch_error = ''
    try:
        _ = a + np.ones((4,))
    except ValueError as exc:
        mismatch_error = str(exc)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(
            {
                'a_shape': list(a.shape),
                'b_shape': list(b.shape),
                'matmul_shape': list(c.shape),
                'mismatch_error': mismatch_error,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )


if __name__ == '__main__':
    run()
```

Use this `framework_lab.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import torch

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'


def run() -> None:
    batch = torch.randn(4, 8)
    layer = torch.nn.Linear(8, 3)
    logits = layer(batch)
    probs = torch.softmax(logits, dim=-1)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(
            {
                'batch_shape': list(batch.shape),
                'logits_shape': list(logits.shape),
                'probs_shape': list(probs.shape),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )


if __name__ == '__main__':
    run()
```

Use this `analysis.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'


def run() -> None:
    scratch = json.loads(SCRATCH.read_text(encoding='utf-8'))
    framework = json.loads(FRAMEWORK.read_text(encoding='utf-8'))
    Path(UNIT_ROOT / 'analysis.md').write_text(
        f'''# 01 Tensor Shapes 분석

- scratch matmul shape: `{scratch["matmul_shape"]}`
- scratch mismatch error: `{scratch["mismatch_error"]}`
- framework logits shape: `{framework["logits_shape"]}`

## 해석

- `2 x 3`과 `3 x 4`의 행렬 곱 결과가 `2 x 4`가 되는 것을 직접 확인했다.
- broadcasting은 편리하지만 축 정렬이 맞지 않으면 즉시 shape mismatch가 난다.
''',
        encoding='utf-8',
    )


if __name__ == '__main__':
    run()
```

Use these docs:

```markdown
# 01 Tensor Shapes

## 왜 이 단위를 배우는가
텐서 shape를 읽지 못하면 backpropagation, attention, GPU 메모리 분석까지 모두 흐려진다.
```

```markdown
# 01 Tensor Shapes 이론 노트

## 핵심 개념
- shape는 각 축의 길이
- batch 차원은 여러 샘플을 동시에 처리하는 축
- broadcasting은 길이가 1인 축을 확장해 연산하는 규칙
```

```markdown
# 01 Tensor Shapes 선행 개념

- 벡터와 행렬 표기
- 행렬 곱의 조건
```

```markdown
# 01 Tensor Shapes 회고

- 이번에 이해한 것:
- 아직 애매한 것:
```
```

- [ ] **Step 4: Run the unit test and the unit scripts**

Run:
- `python -m unittest tests/test_foundations_unit_contract.py -v`
- `python 00_foundations/01_tensor_shapes/scratch_lab.py`
- `python 00_foundations/01_tensor_shapes/framework_lab.py`
- `python 00_foundations/01_tensor_shapes/analysis.py`

Expected:
- PASS
- `artifacts/scratch-manual/metrics.json` created
- `artifacts/framework-manual/metrics.json` created
- `analysis.md` filled with Korean interpretation

- [ ] **Step 5: Commit the first gold-standard unit**

```bash
git add 00_foundations/01_tensor_shapes tests/test_foundations_unit_contract.py
git commit -F - <<'EOF'
Teach tensor shapes before deeper model mechanics

The redesign starts with the smallest concept that every later unit
needs: reading tensor shapes, spotting broadcasting, and mapping
scratch intuition to framework tensors.

Constraint: The first foundation unit must run without GPU access
Rejected: Start with backpropagation | shape intuition is the more universal prerequisite
Confidence: high
Scope-risk: narrow
Directive: Preserve this unit as a readability benchmark when authoring later foundation lessons
Tested: python -m unittest tests/test_foundations_unit_contract.py -v; python 00_foundations/01_tensor_shapes/scratch_lab.py; python 00_foundations/01_tensor_shapes/framework_lab.py; python 00_foundations/01_tensor_shapes/analysis.py
Not-tested: Automated link validation for the new unit docs
EOF
```

---

### Task 5: Build `00_foundations/05_gpu_memory_runtime`

**Files:**
- Create: `00_foundations/05_gpu_memory_runtime/{README.md,THEORY.md,PREREQS.md,lesson.yaml,scratch_lab.py,framework_lab.py,analysis.py,analysis.md,reflection.md,artifacts/.gitkeep}`
- Modify: `tests/test_foundations_unit_contract.py`
- Test: `tests/test_foundations_unit_contract.py`

- [ ] **Step 1: Extend the foundations contract test**

```python
    def test_gpu_memory_unit_has_required_files(self) -> None:
        unit = FOUNDATIONS / '05_gpu_memory_runtime'
        for rel in REQUIRED:
            self.assertTrue((unit / rel).exists(), rel)

    def test_gpu_metadata_mentions_runtime_hooks(self) -> None:
        text = (FOUNDATIONS / '05_gpu_memory_runtime' / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('runtime_observation_hooks:', text)
        self.assertIn('max_memory_allocated', text)
```

- [ ] **Step 2: Run the test and verify the new assertions fail**

Run: `python -m unittest tests/test_foundations_unit_contract.py -v`  
Expected: FAIL because the GPU/runtime unit does not exist yet.

- [ ] **Step 3: Create the GPU/runtime unit with CPU fallback**

Use this `lesson.yaml`:

```yaml
objective: 무엇이 GPU 메모리를 차지하는지, training과 inference가 왜 다른지 이해한다.
prereqs:
  - tensor shape 감각
  - dtype 기본 개념
key_terms:
  - vram
  - parameter
  - activation
  - mixed precision
runtime_observation_hooks:
  - dtype
  - device
  - max_memory_allocated
  - max_memory_reserved
analysis_questions:
  - 어떤 항목이 batch size와 함께 증가했는가?
  - training과 inference의 차이는 무엇인가?
```

Use this `scratch_lab.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
DTYPE_BYTES = {'fp32': 4, 'fp16': 2, 'bf16': 2}


def tensor_bytes(shape: tuple[int, ...], dtype: str) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total * DTYPE_BYTES[dtype]


def run() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(
            {
                'batch_fp32_bytes': tensor_bytes((32, 512, 768), 'fp32'),
                'batch_fp16_bytes': tensor_bytes((32, 512, 768), 'fp16'),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )


if __name__ == '__main__':
    run()
```

Use this `framework_lab.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import torch

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'


def run() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
    ).to(device)
    batch = torch.randn(16, 1024, device=device)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(batch)
    inference_allocated = int(torch.cuda.max_memory_allocated()) if device == 'cuda' else 0

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    loss = model(batch).pow(2).mean()
    loss.backward()
    training_allocated = int(torch.cuda.max_memory_allocated()) if device == 'cuda' else 0

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(
            {
                'device': device,
                'dtype': str(batch.dtype),
                'inference_allocated': inference_allocated,
                'training_allocated': training_allocated,
                'max_memory_reserved': int(torch.cuda.max_memory_reserved()) if device == 'cuda' else 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )


if __name__ == '__main__':
    run()
```

Use this `analysis.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'


def run() -> None:
    scratch = json.loads(SCRATCH.read_text(encoding='utf-8'))
    framework = json.loads(FRAMEWORK.read_text(encoding='utf-8'))
    Path(UNIT_ROOT / 'analysis.md').write_text(
        f'''# 05 GPU Memory Runtime 분석

- fp32 bytes: `{scratch["batch_fp32_bytes"]}`
- fp16 bytes: `{scratch["batch_fp16_bytes"]}`
- device: `{framework["device"]}`
- inference allocated: `{framework["inference_allocated"]}`
- training allocated: `{framework["training_allocated"]}`

## 해석

- 같은 shape라도 dtype이 바뀌면 메모리 사용량이 즉시 달라진다.
- training은 backward 때문에 inference보다 비싸다.
''',
        encoding='utf-8',
    )


if __name__ == '__main__':
    run()
```

Use these docs:

```markdown
# 05 GPU Memory Runtime

## 왜 이 단위를 배우는가
GPU 메모리 감각이 없으면 batch size, mixed precision, gradient accumulation, OOM 원인을 설명할 수 없다.
```

```markdown
# 05 GPU Memory Runtime 이론 노트

## VRAM을 차지하는 것
- parameter
- activation
- gradient
- optimizer state
```

```markdown
# 05 GPU Memory Runtime 선행 개념

- tensor shape
- dtype
- forward / backward 구분
```

```markdown
# 05 GPU Memory Runtime 회고

- 이번에 이해한 것:
- 아직 애매한 것:
```
```

- [ ] **Step 4: Run the updated contract test and both unit scripts**

Run:
- `python -m unittest tests/test_foundations_unit_contract.py -v`
- `python 00_foundations/05_gpu_memory_runtime/scratch_lab.py`
- `python 00_foundations/05_gpu_memory_runtime/framework_lab.py`
- `python 00_foundations/05_gpu_memory_runtime/analysis.py`

Expected:
- PASS
- CPU-only 환경에서도 metrics 파일 생성
- `analysis.md`가 training vs inference 차이를 설명

- [ ] **Step 5: Commit the GPU/runtime unit**

```bash
git add 00_foundations/05_gpu_memory_runtime tests/test_foundations_unit_contract.py
git commit -F - <<'EOF'
Make GPU memory behavior concrete before transformer-scale work

The curriculum needs a unit that explains what actually occupies
VRAM and why training costs more than inference, so this change adds
an evidence-first system lesson with CPU fallback.

Constraint: The lesson must remain useful on machines without CUDA
Rejected: Defer runtime mechanics until multimodal work | leaves a core learner pain point unresolved
Confidence: high
Scope-risk: narrow
Directive: Keep runtime analysis grounded in observed metrics, not folklore about GPUs
Tested: python -m unittest tests/test_foundations_unit_contract.py -v; python 00_foundations/05_gpu_memory_runtime/scratch_lab.py; python 00_foundations/05_gpu_memory_runtime/framework_lab.py; python 00_foundations/05_gpu_memory_runtime/analysis.py
Not-tested: CUDA-specific max-memory values on multiple GPU models
EOF
```

---

### Task 6: Add a lesson runner, report scaffold, and markdown link checker

**Files:**
- Create: `scripts/_lesson_metadata.py`, `scripts/run_lesson.py`, `scripts/build_lesson_report.py`, `scripts/check_curriculum_links.py`, `tests/test_lesson_runner_contract.py`, `tests/test_curriculum_links.py`
- Modify: `scripts/README.md`
- Test: `tests/test_lesson_runner_contract.py`, `tests/test_curriculum_links.py`

- [ ] **Step 1: Write the failing automation tests**

```python
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestLessonRunnerContract(unittest.TestCase):
    def test_runner_executes_tensor_shapes_scratch(self) -> None:
        result = subprocess.run(
            [sys.executable, 'scripts/run_lesson.py', '--unit', '00_foundations/01_tensor_shapes', '--mode', 'scratch'],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('scratch-manual', result.stdout)


if __name__ == '__main__':
    unittest.main()
```

```python
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestCurriculumLinks(unittest.TestCase):
    def test_link_checker_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, 'scripts/check_curriculum_links.py'],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the scripts are missing**

Run: `python -m unittest tests/test_lesson_runner_contract.py tests/test_curriculum_links.py -v`  
Expected: FAIL because the scripts do not exist yet.

- [ ] **Step 3: Implement the minimal runner and checker**

Use this `scripts/_lesson_metadata.py`:

```python
from __future__ import annotations

from pathlib import Path


def load_lesson_metadata(path: str | Path) -> dict[str, object]:
    data: dict[str, object] = {}
    current_key: str | None = None
    for raw_line in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith('#'):
            continue
        if line.startswith('  - ') and current_key is not None:
            data.setdefault(current_key, [])
            assert isinstance(data[current_key], list)
            data[current_key].append(line[4:])
            continue
        key, _, value = line.partition(':')
        current_key = key.strip()
        value = value.strip()
        data[current_key] = value if value else []
    return data
```

Use this `scripts/run_lesson.py`:

```python
from __future__ import annotations

import argparse
import runpy
from pathlib import Path

from _lesson_metadata import load_lesson_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', required=True)
    parser.add_argument('--mode', choices=['scratch', 'framework'], required=True)
    args = parser.parse_args()
    unit = Path(args.unit)
    metadata = load_lesson_metadata(unit / 'lesson.yaml')
    target = unit / ('scratch_lab.py' if args.mode == 'scratch' else 'framework_lab.py')
    runpy.run_path(str(target), run_name='__main__')
    print(f"unit={unit} mode={args.mode} objective={metadata['objective']} output={unit / 'artifacts'}")


if __name__ == '__main__':
    main()
```

Use this `scripts/build_lesson_report.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    unit = Path(args.unit)
    scratch = unit / 'artifacts' / 'scratch-manual' / 'metrics.json'
    framework = unit / 'artifacts' / 'framework-manual' / 'metrics.json'
    summary = unit / 'artifacts' / 'summary.md'
    summary.write_text(
        '\n'.join(
            [
                f'# {unit.name} 요약',
                '',
                f'- scratch keys: {sorted(json.loads(scratch.read_text(encoding=\"utf-8\")).keys()) if scratch.exists() else []}',
                f'- framework keys: {sorted(json.loads(framework.read_text(encoding=\"utf-8\")).keys()) if framework.exists() else []}',
                '- 다음 질문: 분석 문서에서 왜 이런 결과가 나왔는지 설명하기',
            ]
        ),
        encoding='utf-8',
    )
    print(summary)


if __name__ == '__main__':
    main()
```

Use this `scripts/check_curriculum_links.py`:

```python
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r'\[[^\]]+\]\(([^)]+)\)')


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for rel in ['README.md', 'docs', '00_foundations', '01_ml', '02_nlp_bridge', '03_nlp', '04_multimodal_bridge', '05_multimodal']:
        root = ROOT / rel
        if root.is_file():
            files.append(root)
        elif root.exists():
            files.extend(sorted(root.rglob('*.md')))
    return files


def main() -> int:
    missing: list[str] = []
    for md in iter_markdown_files():
        text = md.read_text(encoding='utf-8')
        for link in LINK_RE.findall(text):
            if link.startswith('http') or link.startswith('#'):
                continue
            if not (md.parent / link).resolve().exists():
                missing.append(f'{md.relative_to(ROOT)} -> {link}')
    if missing:
        print('\n'.join(missing), file=sys.stderr)
        return 1
    print('OK: curriculum markdown links resolve')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 4: Run the automation tests and verification commands**

Run:
- `python -m unittest tests/test_lesson_runner_contract.py tests/test_curriculum_links.py -v`
- `python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode scratch`
- `python scripts/run_lesson.py --unit 00_foundations/05_gpu_memory_runtime --mode framework`
- `python scripts/build_lesson_report.py --unit 00_foundations/01_tensor_shapes`
- `python scripts/check_curriculum_links.py`

Expected:
- PASS
- lesson runner prints the chosen unit and output path
- `artifacts/summary.md` created
- link checker prints `OK: curriculum markdown links resolve`

- [ ] **Step 5: Commit the automation scaffold**

```bash
git add scripts/README.md scripts/_lesson_metadata.py scripts/run_lesson.py scripts/build_lesson_report.py scripts/check_curriculum_links.py tests/test_lesson_runner_contract.py tests/test_curriculum_links.py
git commit -F - <<'EOF'
Automate lesson execution without sacrificing readable study outputs

The first foundation units are now structured enough to support a
small runner, summary scaffold, and markdown link checker, giving the
curriculum a repeatable execution path without adding heavy tooling.

Constraint: No new dependency should be required just to parse lesson metadata
Rejected: Add a YAML dependency immediately | unnecessary for the constrained initial metadata schema
Confidence: medium
Scope-risk: moderate
Directive: Keep generated summaries evidence-first and short; do not let the scaffold become generic AI filler
Tested: python -m unittest tests/test_lesson_runner_contract.py tests/test_curriculum_links.py -v; python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode scratch; python scripts/run_lesson.py --unit 00_foundations/05_gpu_memory_runtime --mode framework; python scripts/build_lesson_report.py --unit 00_foundations/01_tensor_shapes; python scripts/check_curriculum_links.py
Not-tested: Rich metadata parsing beyond top-level scalars and top-level lists
EOF
```

---

## Self-Review

- **Spec coverage:** This plan covers indexed foundations/bridges, Korean-first docs, root navigation, renaming applied tracks, shared templates, two gold-standard units, and minimal automation.
- **Placeholder scan:** No `TBD`, `TODO`, “implement later”, or “appropriate error handling” placeholders remain.
- **Type consistency:** The plan consistently uses `03_nlp`, `05_multimodal`, `lesson.yaml`, `scratch_lab.py`, `framework_lab.py`, `analysis.md`, and `reflection.md`.

## Final Verification Sequence

```bash
python -m unittest \
  tests/test_curriculum_topology.py \
  tests/test_reindexed_tracks.py \
  tests/test_shared_templates.py \
  tests/test_foundations_unit_contract.py \
  tests/test_lesson_runner_contract.py \
  tests/test_curriculum_links.py \
  tests/test_01_ml_report_contract.py -v

python scripts/check_curriculum_links.py
```

Expected: all tests PASS, the link checker prints `OK: curriculum markdown links resolve`, and both foundation units produce runnable artifacts.
