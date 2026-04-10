# BTB Curriculum Expansion Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reindex BTB around a new `02_deep_learning` track, shift NLP and multimodal tracks to the approved 00→09 ladder, and land the first extensibility-focused skeleton layer (docs, planned-unit scaffolds, status model, tests, and agent-role docs) on `main`.

**Architecture:** Implement the change in four layers: (1) lock the new contract with failing topology/status tests, (2) rename existing track directories and update path-based tests, (3) add new top-level tracks plus planned-unit skeletons and status metadata, and (4) rewrite the root docs/guide/agent docs so the new ladder is readable before any runnable expansion begins. Keep all newly added units explicitly `planned`; do not imply runnable coverage that does not exist.

**Tech Stack:** Markdown, Python `unittest`, Bash/git, existing `scripts/check_curriculum_links.py`

---

## File Structure Lock-In

**Modify:**
- `README.md`
- `docs/00_program_map.md`
- `scripts/README.md`
- `tests/test_curriculum_topology.py`
- `tests/test_reindexed_tracks.py`
- `tests/test_nlp_bridge_unit_contract.py`
- `tests/test_nlp_task_unit_contract.py`
- `tests/test_multimodal_bridge_unit_contract.py`
- `tests/test_multimodal_task_unit_contract.py`

**Create:**
- `docs/02_study_guide.md`
- `docs/03_track_migration_map.md`
- `docs/curriculum_status.json`
- `docs/agents/README.md`
- `docs/agents/program_director.md`
- `docs/agents/curriculum_architect.md`
- `docs/agents/theory_writer.md`
- `docs/agents/researcher_data_scout.md`
- `docs/agents/experiment_runner.md`
- `docs/agents/critic_verifier.md`
- `tests/test_curriculum_expansion_docs.py`
- `tests/test_curriculum_status_model.py`
- `tests/test_agent_role_docs.py`
- `02_deep_learning/README.md`
- `02_deep_learning/01_perceptron_and_mlp/README.md`
- `02_deep_learning/02_cnn_and_image_classification/README.md`
- `02_deep_learning/03_sequence_models_rnn_lstm_gru/README.md`
- `02_deep_learning/04_attention_and_transformers/README.md`
- `02_deep_learning/05_autoencoders_and_representation_learning/README.md`
- `02_deep_learning/06_generative_models_vae_gan/README.md`
- `02_deep_learning/07_training_recipes_and_debugging/README.md`
- `05_advanced_nlp_llm/README.md`
- `05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/README.md`
- `05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/README.md`
- `05_advanced_nlp_llm/03_domain_adaptive_pretraining/README.md`
- `05_advanced_nlp_llm/04_instruction_tuning_and_sft/README.md`
- `05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/README.md`
- `05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/README.md`
- `05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/README.md`
- `05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/README.md`
- `06_training_systems/README.md`
- `06_training_systems/01_torchrun_and_ddp_basics/README.md`
- `06_training_systems/02_accelerate_workflows/README.md`
- `06_training_systems/03_deepspeed_zero/README.md`
- `06_training_systems/04_fsdp_checkpointing_and_offload/README.md`
- `06_training_systems/05_tensor_parallelism/README.md`
- `06_training_systems/06_pipeline_parallelism/README.md`
- `06_training_systems/07_data_parallel_grad_accumulation/README.md`
- `06_training_systems/08_hybrid_parallel_topologies/README.md`
- `06_training_systems/09_profiling_monitoring_and_failure_recovery/README.md`
- `07_frontier_labs/README.md`
- `07_frontier_labs/01_paper_reproduction_playground/README.md`
- `07_frontier_labs/02_capstone_model_building/README.md`
- `07_frontier_labs/03_agentic_training_and_eval_loops/README.md`
- `07_frontier_labs/04_benchmark_and_dataset_construction/README.md`
- `07_frontier_labs/05_open_ended_research_tracks/README.md`

**Rename via `git mv`:**
- `02_nlp_bridge/` -> `03_nlp_bridge/`
- `03_nlp/` -> `04_nlp/`
- `04_multimodal_bridge/` -> `08_multimodal_bridge/`
- `05_multimodal/` -> `09_multimodal/`

---

### Task 1: Lock the new 00→09 contract with failing tests

**Files:**
- Modify: `tests/test_curriculum_topology.py`
- Modify: `tests/test_reindexed_tracks.py`
- Create: `tests/test_curriculum_expansion_docs.py`
- Create: `tests/test_curriculum_status_model.py`
- Create: `tests/test_agent_role_docs.py`

- [ ] **Step 1: Rewrite the ladder-order assertions for the new top-level sequence**

```python
# tests/test_curriculum_topology.py
ladder = [
    ("00_foundations", "00_foundations/README.md"),
    ("01_ml", "01_ml/README.md"),
    ("02_deep_learning", "02_deep_learning/README.md"),
    ("03_nlp_bridge", "03_nlp_bridge/README.md"),
    ("04_nlp", "04_nlp/README.md"),
    ("05_advanced_nlp_llm", "05_advanced_nlp_llm/README.md"),
    ("06_training_systems", "06_training_systems/README.md"),
    ("07_frontier_labs", "07_frontier_labs/README.md"),
    ("08_multimodal_bridge", "08_multimodal_bridge/README.md"),
    ("09_multimodal", "09_multimodal/README.md"),
]

for rel in ["00_foundations", "02_deep_learning", "03_nlp_bridge", "08_multimodal_bridge"]:
    self.assertTrue((ROOT / rel / "README.md").exists(), rel)
```

- [ ] **Step 2: Replace the old reindex test with explicit rename expectations**

```python
# tests/test_reindexed_tracks.py
for rel in [
    '02_deep_learning/README.md',
    '03_nlp_bridge/README.md',
    '04_nlp/README.md',
    '05_advanced_nlp_llm/README.md',
    '06_training_systems/README.md',
    '07_frontier_labs/README.md',
    '08_multimodal_bridge/README.md',
    '09_multimodal/README.md',
]:
    self.assertTrue((ROOT / rel).exists(), rel)

for old in ['02_nlp_bridge', '03_nlp', '04_multimodal_bridge', '05_multimodal']:
    self.assertFalse((ROOT / old).exists(), old)
```

- [ ] **Step 3: Add a docs-focused regression test for the new study guide and migration note**

```python
# tests/test_curriculum_expansion_docs.py
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class TestCurriculumExpansionDocs(unittest.TestCase):
    def test_study_guide_mentions_new_ladder(self) -> None:
        text = (ROOT / 'docs' / '02_study_guide.md').read_text(encoding='utf-8')
        for token in ['02_deep_learning', '05_advanced_nlp_llm', '06_training_systems', '09_multimodal']:
            self.assertIn(token, text)

    def test_migration_map_mentions_old_and_new_paths(self) -> None:
        text = (ROOT / 'docs' / '03_track_migration_map.md').read_text(encoding='utf-8')
        self.assertIn('02_nlp_bridge', text)
        self.assertIn('03_nlp_bridge', text)
        self.assertIn('05_multimodal', text)
        self.assertIn('09_multimodal', text)
```

- [ ] **Step 4: Add a machine-readable status-model test**

```python
# tests/test_curriculum_status_model.py
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]

class TestCurriculumStatusModel(unittest.TestCase):
    def test_status_manifest_covers_new_tracks(self) -> None:
        data = json.loads((ROOT / 'docs' / 'curriculum_status.json').read_text(encoding='utf-8'))
        for track in ['02_deep_learning', '05_advanced_nlp_llm', '06_training_systems', '07_frontier_labs']:
            self.assertIn(track, data['tracks'])

    def test_all_declared_units_have_status_and_readme(self) -> None:
        data = json.loads((ROOT / 'docs' / 'curriculum_status.json').read_text(encoding='utf-8'))
        for track, units in data['tracks'].items():
            for unit_name, status in units.items():
                self.assertIn(status, {'planned', 'outlined', 'runnable'})
                readme = ROOT / track / unit_name / 'README.md'
                self.assertTrue(readme.exists(), str(readme))
```

- [ ] **Step 5: Add an agent-role-doc presence test**

```python
# tests/test_agent_role_docs.py
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class TestAgentRoleDocs(unittest.TestCase):
    def test_agent_docs_exist(self) -> None:
        for rel in [
            'docs/agents/README.md',
            'docs/agents/program_director.md',
            'docs/agents/curriculum_architect.md',
            'docs/agents/theory_writer.md',
            'docs/agents/researcher_data_scout.md',
            'docs/agents/experiment_runner.md',
            'docs/agents/critic_verifier.md',
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)
```

- [ ] **Step 6: Run the new/updated tests to verify they fail before implementation**

Run:
```bash
python -m unittest \
  tests.test_curriculum_topology \
  tests.test_reindexed_tracks \
  tests.test_curriculum_expansion_docs \
  tests.test_curriculum_status_model \
  tests.test_agent_role_docs
```
Expected: FAIL because `02_deep_learning/`, `docs/02_study_guide.md`, `docs/curriculum_status.json`, and `docs/agents/` do not exist yet.

- [ ] **Step 7: Commit the failing-test checkpoint**

```bash
git add \
  tests/test_curriculum_topology.py \
  tests/test_reindexed_tracks.py \
  tests/test_curriculum_expansion_docs.py \
  tests/test_curriculum_status_model.py \
  tests/test_agent_role_docs.py

git commit -m "Lock the 00-09 curriculum contract before reindexing"
```

---

### Task 2: Rename existing tracked directories and update path-based contract tests

**Files:**
- Rename: `02_nlp_bridge/` -> `03_nlp_bridge/`
- Rename: `03_nlp/` -> `04_nlp/`
- Rename: `04_multimodal_bridge/` -> `08_multimodal_bridge/`
- Rename: `05_multimodal/` -> `09_multimodal/`
- Modify: `tests/test_nlp_bridge_unit_contract.py`
- Modify: `tests/test_nlp_task_unit_contract.py`
- Modify: `tests/test_multimodal_bridge_unit_contract.py`
- Modify: `tests/test_multimodal_task_unit_contract.py`

- [ ] **Step 1: Perform the directory renames in collision-safe order**

Run:
```bash
git mv 05_multimodal 09_multimodal
git mv 04_multimodal_bridge 08_multimodal_bridge
git mv 03_nlp 04_nlp
git mv 02_nlp_bridge 03_nlp_bridge
```
Expected: `git status --short` shows four `R` entries.

- [ ] **Step 2: Rewrite NLP bridge test constants to the new `03_nlp_bridge` root**

```python
# tests/test_nlp_bridge_unit_contract.py
UNIT_SPECS = {
    'tokenization': {
        'unit': ROOT / '03_nlp_bridge' / '01_tokenization_and_embeddings',
    },
    'attention': {
        'unit': ROOT / '03_nlp_bridge' / '02_attention_and_transformer_block',
    },
}
```

- [ ] **Step 3: Rewrite applied NLP test constants to the new `04_nlp` root**

```python
# tests/test_nlp_task_unit_contract.py
UNIT_SPECS = {
    'text_classification': {
        'unit': ROOT / '04_nlp' / '01_text_classification',
    },
    'ner': {
        'unit': ROOT / '04_nlp' / '02_named_entity_recognition',
    },
    'mrc': {
        'unit': ROOT / '04_nlp' / '03_machine_reading_comprehension',
    },
}
```

- [ ] **Step 4: Rewrite multimodal contract tests to the new `08` / `09` roots**

```python
# tests/test_multimodal_bridge_unit_contract.py
UNIT = ROOT / '08_multimodal_bridge' / '01_contrastive_alignment'

# tests/test_multimodal_task_unit_contract.py
UNIT = ROOT / '09_multimodal' / '01_image_text_retrieval'
CAPTION_UNIT = ROOT / '09_multimodal' / '02_image_captioning'
VQA_UNIT = ROOT / '09_multimodal' / '03_visual_question_answering'
```

- [ ] **Step 5: Run the renamed contract tests to catch path regressions immediately**

Run:
```bash
python -m unittest \
  tests.test_nlp_bridge_unit_contract \
  tests.test_nlp_task_unit_contract \
  tests.test_multimodal_bridge_unit_contract \
  tests.test_multimodal_task_unit_contract
```
Expected: PASS on path discovery or fail only on still-missing doc/index updates, not on `FileNotFoundError` from old track roots.

- [ ] **Step 6: Commit the directory-reindex checkpoint**

```bash
git add \
  03_nlp_bridge 04_nlp 08_multimodal_bridge 09_multimodal \
  tests/test_nlp_bridge_unit_contract.py \
  tests/test_nlp_task_unit_contract.py \
  tests/test_multimodal_bridge_unit_contract.py \
  tests/test_multimodal_task_unit_contract.py

git commit -m "Reindex the existing NLP and multimodal tracks"
```

---

### Task 3: Create the new top-level tracks, planned-unit skeletons, and status manifest

**Files:**
- Create: `02_deep_learning/**`
- Create: `05_advanced_nlp_llm/**`
- Create: `06_training_systems/**`
- Create: `07_frontier_labs/**`
- Create: `docs/curriculum_status.json`

- [ ] **Step 1: Create the new top-level directories and unit folders**

Run:
```bash
mkdir -p \
  02_deep_learning/{01_perceptron_and_mlp,02_cnn_and_image_classification,03_sequence_models_rnn_lstm_gru,04_attention_and_transformers,05_autoencoders_and_representation_learning,06_generative_models_vae_gan,07_training_recipes_and_debugging} \
  05_advanced_nlp_llm/{01_language_modeling_and_pretraining_objectives,02_corpus_tokenizer_and_data_mixture,03_domain_adaptive_pretraining,04_instruction_tuning_and_sft,05_preference_optimization_dpo_orpo_kto,06_rlhf_and_reasoning_rl,07_retrieval_augmented_generation_and_eval,08_alignment_safety_and_model_behavior} \
  06_training_systems/{01_torchrun_and_ddp_basics,02_accelerate_workflows,03_deepspeed_zero,04_fsdp_checkpointing_and_offload,05_tensor_parallelism,06_pipeline_parallelism,07_data_parallel_grad_accumulation,08_hybrid_parallel_topologies,09_profiling_monitoring_and_failure_recovery} \
  07_frontier_labs/{01_paper_reproduction_playground,02_capstone_model_building,03_agentic_training_and_eval_loops,04_benchmark_and_dataset_construction,05_open_ended_research_tracks}
```
Expected: `find 02_deep_learning 05_advanced_nlp_llm 06_training_systems 07_frontier_labs -maxdepth 1 -type d | wc -l` reports the new track roots and unit folders.

- [ ] **Step 2: Create track-level README files with unit tables and role boundaries**

```markdown
# 02 Deep Learning

이 트랙은 `00_foundations`의 공통 기초 다음에 오는 **본격 딥러닝 모델 패밀리 학습 구간**이다.

| Unit | Status | Focus |
| --- | --- | --- |
| `01_perceptron_and_mlp` | planned | single neuron, perceptron, MLP |
| `02_cnn_and_image_classification` | planned | conv, pooling, image classification |
| `03_sequence_models_rnn_lstm_gru` | planned | RNN/LSTM/GRU |
```

- [ ] **Step 3: Create planned-unit README templates for every new unit**

Use this exact template for each new unit README:

```markdown
# 01 Perceptron and MLP

> Status: planned

## 왜 이 단위를 배우는가
이 단위는 `00_foundations`에서 본 activation / loss / gradient 감각을 가장 작은 supervised neural model로 연결한다.

## 이번 단위에서 들어올 것
- perceptron decision rule
- single neuron to MLP
- hidden layer intuition
- tiny classification experiment outline

## 선행 개념
- `00_foundations/01_tensor_shapes`
- `00_foundations/03_activation_and_loss`
- `00_foundations/04_gradients_and_backpropagation`

## 계획된 산출물
- `README.md`
- `THEORY.md`
- `PREREQS.md`
- `lesson.yaml`
- `scratch_lab.py`
- `framework_lab.py`
- `analysis.md`
- `reflection.md`
```

- [ ] **Step 4: Write the machine-readable status manifest**

```json
{
  "tracks": {
    "02_deep_learning": {
      "01_perceptron_and_mlp": "planned",
      "02_cnn_and_image_classification": "planned",
      "03_sequence_models_rnn_lstm_gru": "planned",
      "04_attention_and_transformers": "planned",
      "05_autoencoders_and_representation_learning": "planned",
      "06_generative_models_vae_gan": "planned",
      "07_training_recipes_and_debugging": "planned"
    },
    "05_advanced_nlp_llm": {
      "01_language_modeling_and_pretraining_objectives": "planned",
      "02_corpus_tokenizer_and_data_mixture": "planned",
      "03_domain_adaptive_pretraining": "planned",
      "04_instruction_tuning_and_sft": "planned",
      "05_preference_optimization_dpo_orpo_kto": "planned",
      "06_rlhf_and_reasoning_rl": "planned",
      "07_retrieval_augmented_generation_and_eval": "planned",
      "08_alignment_safety_and_model_behavior": "planned"
    }
  }
}
```

- [ ] **Step 5: Extend the manifest with `06_training_systems` and `07_frontier_labs`, then rerun the status test**

Run:
```bash
python -m unittest tests.test_curriculum_status_model -v
```
Expected: still FAIL until docs and agent files exist, but no JSON parse errors.

- [ ] **Step 6: Commit the new-track scaffolding checkpoint**

```bash
git add \
  02_deep_learning 05_advanced_nlp_llm 06_training_systems 07_frontier_labs \
  docs/curriculum_status.json

git commit -m "Scaffold the new deep learning, LLM, systems, and frontier tracks"
```

---

### Task 4: Rewrite the root docs, create the new study guide, and publish a migration note

**Files:**
- Modify: `README.md`
- Modify: `docs/00_program_map.md`
- Create: `docs/02_study_guide.md`
- Create: `docs/03_track_migration_map.md`
- Modify: `scripts/README.md`

- [ ] **Step 1: Rewrite the root README ladder and storage map for the 00→09 sequence**

```markdown
## 학습 순서

1. [00_foundations](00_foundations/README.md)
2. [01_ml](01_ml/README.md)
3. [02_deep_learning](02_deep_learning/README.md)
4. [03_nlp_bridge](03_nlp_bridge/README.md)
5. [04_nlp](04_nlp/README.md)
6. [05_advanced_nlp_llm](05_advanced_nlp_llm/README.md)
7. [06_training_systems](06_training_systems/README.md)
8. [07_frontier_labs](07_frontier_labs/README.md)
9. [08_multimodal_bridge](08_multimodal_bridge/README.md)
10. [09_multimodal](09_multimodal/README.md)
```

- [ ] **Step 2: Rewrite `docs/00_program_map.md` so each track has a one-line role boundary**

```markdown
1. `00_foundations` — 공통 수치/텐서/실행 감각
2. `01_ml` — 실험 discipline과 baseline 해석
3. `02_deep_learning` — 딥러닝 모델 패밀리 학습
4. `03_nlp_bridge` — DL에서 NLP로 넘어가는 입력/표현 다리
5. `04_nlp` — applied NLP core
6. `05_advanced_nlp_llm` — pretraining 이후 고급 NLP/LLM
7. `06_training_systems` — distributed and large-model training systems
8. `07_frontier_labs` — reproduction, capstone, agentic experiments
9. `08_multimodal_bridge` — multimodal 연결 다리
10. `09_multimodal` — multimodal applied track
```

- [ ] **Step 3: Create `docs/02_study_guide.md` with a standard route and a compressed route**

```markdown
# 02 Study Guide

## 루트 A. 표준 1-pass
`00_foundations -> 01_ml -> 02_deep_learning -> 03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm -> 06_training_systems -> 07_frontier_labs -> 08_multimodal_bridge -> 09_multimodal`

## 루트 B. NLP/LLM 우선 압축 루트
`00_foundations -> 01_ml -> 02_deep_learning(01,04만 우선) -> 03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm`
```

- [ ] **Step 4: Publish a migration note for old-to-new track paths**

```markdown
# 03 Track Migration Map

| Old Path | New Path |
| --- | --- |
| `02_nlp_bridge` | `03_nlp_bridge` |
| `03_nlp` | `04_nlp` |
| `04_multimodal_bridge` | `08_multimodal_bridge` |
| `05_multimodal` | `09_multimodal` |
```

- [ ] **Step 5: Update `scripts/README.md` examples to the new track paths**

```text
python scripts/train.py --track 04_nlp --stage 01_text_classification --config path/to/config.yaml
python scripts/eval.py --run-dir runs/04_nlp/01_text_classification/<run_id>
python scripts/eval.py --run-dir runs/09_multimodal/01_image_text_retrieval/<run_id>
```

- [ ] **Step 6: Run the docs regression tests and the link checker**

Run:
```bash
python -m unittest \
  tests.test_curriculum_topology \
  tests.test_reindexed_tracks \
  tests.test_curriculum_expansion_docs -v
python scripts/check_curriculum_links.py
```
Expected: unittest PASS; link checker PASS or expose only pre-existing unrelated failures that must be fixed in this branch before merge.

- [ ] **Step 7: Commit the docs-rewrite checkpoint**

```bash
git add README.md docs/00_program_map.md docs/02_study_guide.md docs/03_track_migration_map.md scripts/README.md

git commit -m "Publish the 00-09 curriculum ladder in the root docs"
```

---

### Task 5: Add agent-role docs and tie them into the curriculum expansion tests

**Files:**
- Create: `docs/agents/README.md`
- Create: `docs/agents/program_director.md`
- Create: `docs/agents/curriculum_architect.md`
- Create: `docs/agents/theory_writer.md`
- Create: `docs/agents/researcher_data_scout.md`
- Create: `docs/agents/experiment_runner.md`
- Create: `docs/agents/critic_verifier.md`

- [ ] **Step 1: Create the agents index with the first-phase workflow**

```markdown
# BTB Agent Roles

## Phase 1 workflow
1. Program Director scopes the lane
2. Curriculum Architect defines structure and prerequisites
3. Theory Writer drafts README/THEORY/PREREQS skeletons
4. Researcher/Data Scout gathers papers, datasets, and references
5. Experiment Runner executes runnable units on available compute
6. Critic/Verifier checks docs, links, artifacts, and claims
```

- [ ] **Step 2: Create concise role docs with responsibilities and handoff I/O**

```markdown
# Program Director

## Responsibilities
- prioritize track rollout
- assign lane owners
- decide when a unit moves from planned -> outlined -> runnable

## Inputs
- approved spec
- current curriculum status

## Outputs
- rollout priority
- worker assignments
- merge gate decision
```

- [ ] **Step 3: Repeat the same structure for the other five roles**

Required sections in each file:
```markdown
## Responsibilities
## Inputs
## Outputs
## Done Criteria
## Common Failure Modes
```

- [ ] **Step 4: Run the agent-doc presence test**

Run:
```bash
python -m unittest tests.test_agent_role_docs -v
```
Expected: PASS.

- [ ] **Step 5: Commit the agent-doc checkpoint**

```bash
git add docs/agents tests/test_agent_role_docs.py

git commit -m "Document the curriculum expansion agent roles"
```

---

### Task 6: Final verification, cleanup, and merge-readiness check

**Files:**
- Modify as needed: any files from Tasks 1-5

- [ ] **Step 1: Run the full curriculum-facing regression slice**

Run:
```bash
python -m unittest \
  tests.test_curriculum_topology \
  tests.test_reindexed_tracks \
  tests.test_curriculum_expansion_docs \
  tests.test_curriculum_status_model \
  tests.test_agent_role_docs \
  tests.test_nlp_bridge_unit_contract \
  tests.test_nlp_task_unit_contract \
  tests.test_multimodal_bridge_unit_contract \
  tests.test_multimodal_task_unit_contract
```
Expected: PASS.

- [ ] **Step 2: Run the link checker and capture any regressions immediately**

Run:
```bash
python scripts/check_curriculum_links.py
```
Expected: `OK`.

- [ ] **Step 3: Verify the new ladder appears everywhere it must**

Run:
```bash
rg -n "02_deep_learning|05_advanced_nlp_llm|06_training_systems|07_frontier_labs|08_multimodal_bridge|09_multimodal" README.md docs scripts tests
```
Expected: matches in root docs, migration docs, tests, and scripts README.

- [ ] **Step 4: Review the status manifest for honest coverage**

Run:
```bash
python - <<'PY'
from pathlib import Path
import json
p = Path('docs/curriculum_status.json')
data = json.loads(p.read_text(encoding='utf-8'))
for track, units in data['tracks'].items():
    print(track, sorted(set(units.values())))
PY
```
Expected: newly created units show only `planned`; no accidental `runnable` claims.

- [ ] **Step 5: Commit the final phase-1 integration checkpoint**

```bash
git add README.md docs scripts tests 02_deep_learning 03_nlp_bridge 04_nlp 05_advanced_nlp_llm 06_training_systems 07_frontier_labs 08_multimodal_bridge 09_multimodal

git commit -m "Lay down the first extensible 00-09 BTB curriculum skeleton"
```
