# BTB Curriculum Reliability and Milestones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every behavior change. The controller owns integration commits because all workers share one filesystem.

**Goal:** Make all 48 lessons honestly classifiable, consistently runnable, evidence-based in the web UI, and verifiable on CPU plus bounded idle-GPU smoke experiments.

**Architecture:** Use one canonical metadata loader and audit surface, preserve existing lesson entrypoints, and add optional fields rather than breaking old YAML. Propagate `BTB_DEVICE` into representative Torch labs, keep manual progress separate from a computed mastery-evidence badge, and persist only safe run evidence in localStorage.

**Tech Stack:** Python 3.12, unittest/pytest, PyTorch 2.8, vanilla JavaScript, localStorage, Playwright, OMX ultrawork.

**Execution note:** The user requested parallel subagents and the hook explicitly continued execution on the clean `main` workspace. Independent tasks have disjoint write scopes; the root agent integrates and commits after review.

---

### Task 1: Canonical lesson metadata loader and whole-curriculum audit

**Files:**
- Modify: `scripts/_lesson_metadata.py`
- Modify: `scripts/build_web_catalog.py`
- Create: `scripts/audit_curriculum.py`
- Modify: `tests/test_lesson_runner_contract.py`
- Modify: `tests/test_curriculum_status_model.py`

- [ ] **Step 1: Add failing parser regression tests**

Add tests that parse a shallow nested mapping and every manifest lesson:

```python
def test_metadata_loader_parses_shallow_mapping(self):
    metadata = load_lesson_metadata(path)
    self.assertEqual("scratch_lab.py", metadata["scripts"]["scratch"])

def test_every_manifest_lesson_uses_runner_parser(self):
    for track, units in manifest["tracks"].items():
        for unit in units:
            self.assertIn("objective", load_lesson_metadata(ROOT / track / unit / "lesson.yaml"))
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest tests/test_lesson_runner_contract.py tests/test_curriculum_status_model.py -q`

Expected: nested mapping and seven runnable unit parse failures.

- [ ] **Step 3: Implement one constrained loader**

Extend `LessonValue` to nested string mappings, reject deeper indentation with file/line errors, and import this loader from `build_web_catalog.py` instead of maintaining a duplicate parser.

- [ ] **Step 4: Add the audit command**

`audit_curriculum.py` must load the manifest, validate required metadata fields and lesson resources, and print JSON with `unit_count`, `errors`, and fidelity/runtime coverage.

- [ ] **Step 5: Verify GREEN**

Run the targeted tests and `python scripts/audit_curriculum.py --strict`; expect 48 units and zero parser/resource errors after Task 2 metadata lands.

### Task 2: Honest learner metadata and route documentation

**Files:**
- Modify: all 48 `*/lesson.yaml` files declared in `docs/curriculum_status.json`
- Create: `docs/00_learner_preflight.md`
- Modify: `docs/00_program_map.md`
- Modify: `docs/02_study_guide.md`
- Modify: `README.md`
- Modify: `tests/test_curriculum_track_docs.py`
- Modify: `tests/test_curriculum_status_model.py`

- [ ] **Step 1: Add failing coverage tests**

Require every lesson to declare:

```python
self.assertIn(metadata["fidelity"], {"concept-toy", "framework-toy", "real-data", "gpu-capable"})
self.assertIn(metadata["difficulty"], {"beginner", "intermediate", "advanced"})
self.assertGreater(int(metadata["estimated_minutes"]), 0)
self.assertIn(metadata["compute"], {"cpu", "cpu-or-cuda", "optional-multiprocess"})
```

Also require learner-preflight and optional-sidecar language in the program/study docs.

- [ ] **Step 2: Confirm RED**

Run: `python -m pytest tests/test_curriculum_status_model.py tests/test_curriculum_track_docs.py -q`

Expected: missing metadata and preflight document assertions.

- [ ] **Step 3: Add explicit metadata**

Classify each lesson from its actual implementation, not its title. Use `real-data` only for real dataset stages, `framework-toy` only for actual framework tensor/model computation, and `concept-toy` for simulations. `gpu-capable` means the code honors `BTB_DEVICE` but does not claim a historical validation result.

- [ ] **Step 4: Add learner preflight and route corrections**

Document Python/CLI, linear algebra, probability/metrics, PyTorch, and GPU checks. Reframe Systems and Frontier as optional sidecars while keeping canonical folder order unchanged.

- [ ] **Step 5: Verify GREEN**

Run the targeted tests and the strict curriculum audit.

### Task 3: Runner modes, device propagation, and actionable report evidence

**Files:**
- Modify: `scripts/run_lesson.py`
- Modify: `scripts/build_lesson_report.py`
- Modify: `scripts/README.md`
- Modify: `tests/test_lesson_runner_contract.py`

- [ ] **Step 1: Add failing runner/report tests**

Cover `--mode analysis`, `--mode all`, `--device cpu`, pre-execution context, observed analysis report recognition, and unresolved output disclosure.

```python
result = self._run(str(RUN_LESSON), "--unit", unit, "--mode", "all", "--device", "cpu")
self.assertIn("selected_device=cpu", result.stdout)
self.assertIn("completed_modes=scratch,framework,analysis", result.stdout)
```

- [ ] **Step 2: Confirm RED**

Run: `python -m pytest tests/test_lesson_runner_contract.py -q`.

- [ ] **Step 3: Implement minimal compatible CLI**

Keep existing scratch/framework behavior. For `all`, run in pedagogical order. Set `BTB_DEVICE`; for CPU also set `CUDA_VISIBLE_DEVICES=""`. Add actionable missing-entrypoint errors.

- [ ] **Step 4: Strengthen report output resolution**

Map standard labels and path-like entries to concrete paths/globs. Never silently drop unknown declarations: include them under `unverified declarations` and fail only when a concretely resolved required artifact is absent. Include selected metric values, device, artifact links, and analysis questions in the summary.

- [ ] **Step 5: Correct scripts documentation and verify GREEN**

Remove nonexistent `scripts/train.py`/`scripts/eval.py` examples and run the targeted tests.

### Task 4: Safe shared Torch device contract and representative milestones

**Files:**
- Create: `shared/device_runtime.py`
- Modify: `shared/README.md`
- Modify: `05_advanced_nlp_llm/04_instruction_tuning_and_sft/framework_lab.py`
- Modify: `09_multimodal/01_image_text_retrieval/framework_lab.py`
- Modify: `10_vla/01_vision_language_action_grounding/framework_lab.py`
- Modify: `tests/test_advanced_llm_sft_unit_contract.py`
- Modify: `tests/test_multimodal_task_unit_contract.py`
- Modify: `tests/test_vla_unit_contract.py`
- Create: `tests/test_device_runtime.py`

- [ ] **Step 1: Add failing device tests**

Test CPU resolution, forced CUDA failure without availability, and `BTB_DEVICE` propagation to each artifact. Existing CPU-default tests must continue to pass by explicitly setting `BTB_DEVICE=cpu`.

- [ ] **Step 2: Confirm RED**

Run the four affected test files and verify artifacts still report CPU despite `BTB_DEVICE=cuda` or the shared helper is missing.

- [ ] **Step 3: Implement `resolve_torch_device`**

```python
def resolve_torch_device(requested: str | None = None) -> torch.device:
    value = (requested or os.getenv("BTB_DEVICE", "auto")).lower()
    if value == "cpu": return torch.device("cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BTB_DEVICE=cuda was requested but CUDA is unavailable")
    return torch.device("cuda" if torch.cuda.is_available() and value in {"auto", "cuda"} else "cpu")
```

- [ ] **Step 4: Replace synthetic SFT curve with actual Torch optimization**

Retain chat serialization and assistant-only loss masking, but train a tiny embedding/GRU-or-linear next-token model for a bounded epoch count. Preserve existing metric keys and add `initial_loss`, `final_loss`, `parameter_count`, and actual `device`.

- [ ] **Step 5: Use the resolver in retrieval and VLA**

Move tensors/models to the resolved device, detach metrics back to CPU, and keep deterministic seeds plus failure probes.

- [ ] **Step 6: Verify GREEN on CPU**

Run the four targeted test files with `BTB_DEVICE=cpu`.

### Task 5: Safe study-server GPU policy

**Files:**
- Modify: `scripts/study_server.py`
- Modify: `tests/test_web_study_site.py`
- Modify: `web/README.md`

- [ ] **Step 1: Add failing policy tests**

Cover a pinned busy GPU in auto mode, forced CUDA with no GPU rows, and duplicate active execution rejection.

- [ ] **Step 2: Confirm RED**

Run: `python -m pytest tests/test_web_study_site.py -q`.

- [ ] **Step 3: Implement minimal policy fixes**

Pinned GPUs must pass thresholds in auto mode. Forced CUDA without a selectable device must return an actionable error. Protect the run endpoint with one process-local execution lock and always release it in `finally`.

- [ ] **Step 4: Verify GREEN**

Run the targeted web server tests.

### Task 6: Persistent mastery evidence in the web learner

**Files:**
- Modify: `web/progress-storage.js`
- Modify: `web/app.js`
- Modify: `web/styles.css`
- Modify: `tests/test_web_study_site.py`
- Modify: `scripts/playwright_site_qa.js`

- [ ] **Step 1: Add failing storage and behavior tests**

Require a safe `runEvidence` record to survive export/import and corrupt-data recovery. Require blank short answers not to count as submitted. Require mastery evidence to distinguish manual `done` from verified learning evidence.

- [ ] **Step 2: Confirm RED**

Run the progress-storage Node contract and web tests.

- [ ] **Step 3: Persist safe run evidence**

On exit code zero store only unit path, resource href, device, timestamp, and artifact names. Do not store environment variables or raw command strings.

- [ ] **Step 4: Compute and render mastery evidence**

Display evidence for readings, successful run, quiz submission, and non-empty reflection/note. Preserve manual state buttons for learner autonomy, but label manual completion separately from evidence-backed completion.

- [ ] **Step 5: Reject blank short answers and add live status semantics**

Trim text before submission, show an inline message for empty input, and add `aria-live` to execution/quiz feedback.

- [ ] **Step 6: Verify GREEN**

Run Python/Node tests and the Playwright QA script.

### Task 7: Catalog regeneration and bounded GPU smoke evidence

**Files:**
- Regenerate: `web/catalog.json`
- Modify: `docs/04_gpu_conda_experiment_plan.md`
- Generated/ignored: lesson artifact directories

- [ ] **Step 1: Rebuild and verify catalog**

Run `python scripts/build_web_catalog.py --output web/catalog.json`, then strict audit and catalog contract tests.

- [ ] **Step 2: Select an idle GPU immediately before use**

Run `nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader`. Select only a GPU below configured memory/utilization thresholds.

- [ ] **Step 3: Run bounded GPU smokes**

Run Foundations GPU memory, SFT, multimodal retrieval, and VLA framework labs with `CUDA_VISIBLE_DEVICES=<idle>` and `BTB_DEVICE=cuda`. Run their analysis scripts where applicable.

- [ ] **Step 4: Check CPU/GPU invariants**

Require artifact `device == "cuda"`, nonzero CUDA memory for the runtime lesson, finite decreasing loss for learning labs, unchanged failure-probe schema, and expected retrieval/action metrics.

- [ ] **Step 5: Correct GPU experiment documentation**

Remove RLHF simulation as GPU proof. Document which commands are concept toys, GPU-capable tiny labs, and optional heavy extensions.

### Task 8: Integrated review and completion verification

**Files:**
- Review all changed files
- No new production files unless a failing test proves the need

- [ ] **Step 1: Run targeted suites per task**
- [ ] **Step 2: Run `python -m pytest -q`**
- [ ] **Step 3: Run `node --check web/app.js && node --check web/progress-storage.js && node --check scripts/playwright_site_qa.js`**
- [ ] **Step 4: Run `npm run qa:web`**
- [ ] **Step 5: Run `python scripts/audit_curriculum.py --strict` and `python scripts/check_curriculum_links.py`**
- [ ] **Step 6: Inspect `git diff --check`, generated artifacts, and repository status**
- [ ] **Step 7: Perform spec-compliance and code-quality review, fix all important findings, then rerun fresh verification**
