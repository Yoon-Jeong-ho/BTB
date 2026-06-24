# BTB Web Progress + VLA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static BTB study website with local-only per-user progress tracking and add a runnable VLA bridge so the path reaches LLM/RL/multimodal/VLA.

**Architecture:** A Python catalog builder converts repo curriculum metadata into a static JSON file. A dependency-free browser app renders tracks and persists progress in localStorage. A new VLA unit follows the existing runnable unit contract.

**Tech Stack:** Python stdlib, NumPy/PyTorch already used by repo, static HTML/CSS/JS, unittest/pytest.

---

### Task 1: Lock web catalog/progress contract

**Files:**
- Create: `tests/test_web_study_site.py`

- [x] Write failing tests for catalog generation, static asset presence, and localStorage-only progress.
- [ ] Run tests and confirm they fail because implementation files do not exist.

### Task 2: Implement static web catalog and UI

**Files:**
- Create: `scripts/build_web_catalog.py`
- Create: `web/index.html`
- Create: `web/styles.css`
- Create: `web/app.js`
- Generate: `web/catalog.json`
- Create: `web/README.md`

- [ ] Implement deterministic catalog builder.
- [ ] Implement static UI with track cards, unit list, checklist, notes, search/filter, export/import/reset.
- [ ] Generate and commit `web/catalog.json`.

### Task 3: Add VLA runnable unit and curriculum topology

**Files:**
- Create: `10_vla/README.md`
- Create: `10_vla/01_vision_language_action_grounding/*`
- Modify: `docs/curriculum_status.json`, `README.md`, `docs/00_program_map.md`, `docs/02_study_guide.md`, topology tests.
- Create: `tests/test_vla_unit_contract.py`

- [ ] Write failing VLA unit contract test.
- [ ] Add runnable VLA docs/scripts/artifacts contract.
- [ ] Update curriculum ladder docs and manifest.

### Task 4: GPU/conda experiment configuration

**Files:**
- Create: `docs/04_gpu_conda_experiment_plan.md`
- Create: `scripts/check_experiment_environment.py`

- [ ] Capture safe commands for current conda/GPU state.
- [ ] Prefer idle GPUs 4-7 for optional heavier labs; keep runnable tests CPU-safe.

### Task 5: Verification and publish

**Files:** all touched files.

- [ ] Run focused tests first.
- [ ] Run full test suite or the broadest practical equivalent.
- [ ] Build catalog once more and ensure no diff.
- [ ] Commit with Lore-style message.
- [ ] Push branch to GitHub and create/point to PR if available.
