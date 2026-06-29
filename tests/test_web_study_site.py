from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import warnings
from pathlib import Path

from tests.test_curriculum_topology import CANONICAL_CURRICULUM_LADDER

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
BUILDER = ROOT / "scripts" / "build_web_catalog.py"


class WebStudySiteContractTest(unittest.TestCase):
    maxDiff = None

    def _load_builder(self):
        spec = importlib.util.spec_from_file_location("build_web_catalog", BUILDER)
        self.assertIsNotNone(spec, "scripts/build_web_catalog.py must be importable")
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_study_server(self):
        server_path = ROOT / "scripts" / "study_server.py"
        spec = importlib.util.spec_from_file_location("study_server", server_path)
        self.assertIsNotNone(spec, "scripts/study_server.py must be importable")
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules["study_server"] = module
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ResourceWarning)
            spec.loader.exec_module(module)
        return module

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stderr is not None:
            process.stderr.close()

    def test_root_index_redirects_to_web_app_for_plain_localhost(self) -> None:
        root_index = ROOT / "index.html"
        self.assertTrue(root_index.is_file(), "plain http://localhost:8000 should have a root entrypoint")
        text = root_index.read_text(encoding="utf-8")
        self.assertIn('url=/web/', text)
        self.assertIn('href="web/"', text)

    def test_web_assets_exist_and_explain_local_progress(self) -> None:
        required = ["index.html", "styles.css", "progress-storage.js", "app.js", "catalog.json", "README.md"]
        missing = [name for name in required if not (WEB / name).is_file()]
        self.assertEqual([], missing)
        self.assertTrue((ROOT / "scripts" / "study_server.py").is_file(), "one-click code execution needs the local study server")

        html = (WEB / "index.html").read_text(encoding="utf-8")
        app = (WEB / "app.js").read_text(encoding="utf-8")
        storage = (WEB / "progress-storage.js").read_text(encoding="utf-8")
        readme = (WEB / "README.md").read_text(encoding="utf-8")

        self.assertIn("BTB", html)
        self.assertIn("catalog.json", app)
        self.assertIn("progress-storage.js", html)
        self.assertIn("btb.study.progress.v1", storage)
        self.assertIn("localStorage", storage)
        self.assertIn("GitHub에 공유되지", html + readme)
        self.assertIn("사용자별", html + readme)
        self.assertIn("python -m http.server 8000", readme)
        self.assertIn("python scripts/study_server.py --port 8000", readme)
        self.assertIn("--conda-env", readme)
        self.assertIn("--device auto", readme)
        self.assertIn("http://localhost:8000/web/", readme)
        self.assertIn("run_stage.py", readme)
        self.assertNotIn("python -m http.server -d web", readme + app)

    def test_progress_code_is_local_only(self) -> None:
        combined = (WEB / "app.js").read_text(encoding="utf-8") + "\n" + (WEB / "progress-storage.js").read_text(encoding="utf-8")
        styles = (WEB / "styles.css").read_text(encoding="utf-8")
        qa_script = (ROOT / "scripts" / "playwright_site_qa.js").read_text(encoding="utf-8")
        forbidden_network_writes = [
            "navigator.sendBeacon",
            "XMLHttpRequest",
            "fetch('/progress",
            'fetch("/progress',
        ]
        for snippet in forbidden_network_writes:
            self.assertNotIn(snippet, combined)

        self.assertIn("storage.setItem(PROGRESS_KEY", combined)
        self.assertIn("fetch('/api/run-python'", combined)
        self.assertIn("activeUserId", combined)
        self.assertIn("lessons", combined)
        self.assertNotIn("@import url(", styles)
        self.assertNotIn("fonts.googleapis", styles)
        self.assertNotIn("fonts.gstatic", styles)
        self.assertIn("assets/fonts/NotoSansKR-Regular.ttf", styles)
        self.assertIn("isAllowedRequest", qa_script)
        self.assertIn("externalRequests", qa_script)
        self.assertIn("route.abort", qa_script)

    def test_ui_surfaces_learning_process_metadata(self) -> None:
        app = (WEB / "app.js").read_text(encoding="utf-8")
        styles = (WEB / "styles.css").read_text(encoding="utf-8")
        html = (WEB / "index.html").read_text(encoding="utf-8")
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        web_readme = (WEB / "README.md").read_text(encoding="utf-8")
        ml_dl_bridge = (ROOT / "docs" / "04_feature_matrix_to_neural_training_bridge.md").read_text(encoding="utf-8")

        self.assertIn("track.summary", app)
        self.assertIn("renderInlineSummary(track.summary", app)
        self.assertNotIn("escapeHtml(track.summary", app)
        self.assertIn("선행 확인", app)
        self.assertIn("학습 방향", app)
        self.assertIn("../docs/02_study_guide.md", app)
        self.assertIn("../docs/04_feature_matrix_to_neural_training_bridge.md", app)
        self.assertIn("unit.path.startsWith('01_ml/')", app)
        self.assertIn("이 ML stage를 끝내고 딥러닝으로 넘어갈 때 읽기", app)
        self.assertIn("../docs/05_rl_primer_for_rlhf.md", app)
        self.assertIn("09 멀티모달 복습", app)
        self.assertIn("scopeGateFor", app)
        self.assertIn("VLA 범위 확인", app)
        self.assertIn("renderLessonGuidePlan", app)
        self.assertIn("처음 학습 순서", app)
        self.assertIn("이번 단원 브리핑", app)
        self.assertIn("lessonFocusStepsFor", app)
        self.assertIn("isIntroLesson", app)
        self.assertIn("지난 단원과 달라진 점", app)
        self.assertIn("이번에 꼭 볼 것", app)
        self.assertIn("자주 틀리는 지점", app)
        self.assertIn("recommendedStartingUnit", app)

        self.assertIn("학습 자료", app)
        self.assertIn("fetchLessonDocument", app)
        self.assertIn("renderMarkdown", app)
        self.assertIn("lessonSectionsFor", app)
        self.assertIn("unit.resources", app)
        self.assertIn("실습 구성", app)
        for label in ["README", "THEORY", "PREREQS", "scratch_lab.py", "framework_lab.py", "analysis.py", "reflection.md"]:
            self.assertIn(label, app)
        for display_label in ["단원 안내", "핵심 이론", "준비 확인", "기초 실습 코드", "프레임워크 실습 코드", "결과 해석 코드"]:
            self.assertIn(display_label, app)
        self.assertNotIn("README 열기", app)
        self.assertNotIn('target="_blank"', app)
        self.assertIn("renderSectionTab", app)
        self.assertIn('data-complete="${complete}"', app)
        self.assertIn("tab-done-mark", app + styles)
        self.assertIn('.document-tabs button[data-complete="true"]', styles)

        self.assertIn("reader-panel", html)
        self.assertIn("study-sidebar", html)
        self.assertIn("grid-template-areas", styles)
        self.assertIn('"sidebar reader"', styles)
        self.assertIn("clamp(15rem, 22vw, 20rem) minmax(0, 1fr)", styles)
        self.assertNotIn("minmax(220px, 0.8fr) minmax(360px, 1.4fr) minmax(320px, 1fr)", styles)
        self.assertIn("overflow-x: clip", styles)
        self.assertIn(".study-sidebar", styles)
        self.assertIn(".unit-panel { margin-top:", styles)
        self.assertNotIn("top: calc(42vh + 2rem)", styles)
        self.assertIn("사이트 안에서", root_readme + web_readme)
        self.assertIn("문서 파일을 새 탭으로 직접 여는 방식", root_readme + web_readme)
        self.assertIn("현재 브라우저에만 저장", html)
        self.assertIn("기초부터 VLA까지", html)
        self.assertIn("이 브라우저의 진행률", html)
        self.assertNotIn("읽고, 실행하고", html)
        self.assertNotIn("AI 학습 여정", html)
        self.assertNotIn("현재 사용자별 진행률", html)
        self.assertNotIn("0% ·", html + app)
        self.assertNotIn("내 상태:", app)
        self.assertIn("documentSourceLabel", app)
        self.assertIn("source-badge", app + styles)
        self.assertIn("읽고 바로 실행", app)
        self.assertIn("질문 필요", html + app + (WEB / "progress-storage.js").read_text(encoding="utf-8"))
        self.assertIn("읽은 뒤 실행", app)
        self.assertIn("이 코드를 내 환경에서 확인하기", app)
        self.assertIn("기초 실습 코드와 프레임워크 실습 코드를 먼저 실행", app)
        self.assertIn("서버 재시작이 자동 삭제하지는 않지만", app)
        code_branch = app.split("if (section.type === 'code')", 1)[1].split("} else", 1)[0]
        self.assertIn('<span class="source-badge">${escapeHtml(documentSourceLabel(section))}</span>', code_branch)
        self.assertNotIn("<span>${escapeHtml(sectionLabel)}</span><code>${escapeHtml(cleanHref(section.href))}</code>", code_branch)
        self.assertLess(code_branch.find("${renderCoreCodeSummary"), code_branch.find("<pre class=\"code-block\""))
        self.assertLess(code_branch.find("<pre class=\"code-block\""), code_branch.find("${renderRunPanel"))
        self.assertIn("학습 안내", app)
        self.assertIn("트랙 안내", app)
        self.assertNotIn("Study guide", app)
        self.assertNotIn("Track README", app)
        self.assertIn("DataLoader", ml_dl_bridge)
        self.assertIn("epoch", ml_dl_bridge)
        self.assertIn("zero_grad", ml_dl_bridge)
        self.assertNotIn("catalog.json으로", html)
        self.assertNotIn("로컬 캐시", html)
        self.assertNotIn("막힘", html + app + (WEB / "progress-storage.js").read_text(encoding="utf-8"))

    def test_site_guides_execution_reflection_and_next_route(self) -> None:
        app = (WEB / "app.js").read_text(encoding="utf-8")
        html = (WEB / "index.html").read_text(encoding="utf-8")
        styles = (WEB / "styles.css").read_text(encoding="utf-8")
        storage = (WEB / "progress-storage.js").read_text(encoding="utf-8")

        self.assertIn('id="route-select"', html)
        self.assertIn("route-box", html + styles)
        self.assertIn("학습 경로 먼저 선택", html)
        self.assertIn("현재 읽던 단원은 유지", html)
        self.assertIn("LLM/RLHF 빠른 경로", html + app)
        self.assertIn("Multimodal/VLA 경로", html + app)
        self.assertIn("Systems 심화 경로", html + app)
        self.assertIn("selectedRoute", storage + app)
        self.assertIn("routeDefinitions", app)
        self.assertIn("routeUnits", app)
        self.assertIn("routeProgress", app)
        self.assertIn("nextUnitForRoute", app)
        self.assertIn("다음 단원 추천", app)

        self.assertIn("renderRunInsights", app)
        self.assertIn("runPlanFor", app)
        self.assertIn("expectedArtifactsForRun", app)
        self.assertIn("importantNumbersForRun", app)
        self.assertIn("goodOutcomeForRun", app)
        self.assertIn("run-primer", app + styles)
        self.assertIn("extractMetricHighlights", app)
        self.assertIn("실행 관찰 카드", app)
        self.assertIn("봐야 할 숫자", app)
        self.assertIn("예상 산출물", app)
        self.assertIn("좋은 결과 기준", app)
        self.assertIn("다음 질문", app)
        self.assertIn("data-run-insights", app)

        self.assertIn("자가 점검", app)
        self.assertIn("selfChecksFor", app)
        self.assertIn("selfCheckProgress", app)
        self.assertIn("data-self-check", app)
        self.assertIn("data-self-check-summary", app)
        self.assertIn("self-check-meter", app + styles)
        self.assertIn("설명할 수 있다", app)
        self.assertIn("selfChecks", storage + app)

        route_change_block = app.split("routeSelect.addEventListener('change'", 1)[1].split("$('#reset-filters')", 1)[0]
        self.assertNotIn("selectedUnitPath =", route_change_block)
        self.assertNotIn("selectedResourceHref = ''", route_change_block)
        self.assertIn("renderRouteCard", route_change_block)

        self.assertIn("route-card", styles)
        self.assertIn("run-insights", styles)
        self.assertIn("self-checklist", styles)
        self.assertIn("-webkit-line-clamp: 2", styles)

    def test_site_has_playwright_qa_and_code_reading_guidance(self) -> None:
        package_path = ROOT / "package.json"
        self.assertTrue(package_path.is_file(), "Playwright should be installed through package.json")
        package = json.loads(package_path.read_text(encoding="utf-8"))
        app = (WEB / "app.js").read_text(encoding="utf-8")
        styles = (WEB / "styles.css").read_text(encoding="utf-8")
        readme = (WEB / "README.md").read_text(encoding="utf-8") + (ROOT / "README.md").read_text(encoding="utf-8")
        qa_script = ROOT / "scripts" / "playwright_site_qa.js"

        self.assertIn("playwright", package.get("dependencies", {}) | package.get("devDependencies", {}))
        self.assertEqual("node scripts/playwright_site_qa.js", package["scripts"]["qa:web"])
        self.assertTrue(qa_script.is_file(), "Playwright QA should be runnable by future maintainers")

        for token in [
            "renderCodeExplanation",
            "codeExplanationFor",
            "코드 읽기 안내",
            "renderCoreCodeSummary",
            "coreCodeGuideFor",
            "automaticCoreCodeSteps",
            "extractPythonSymbolNames",
            "extractFunctionExcerpt",
            "extractPatternExcerpt",
            "핵심 코드 먼저 보기",
            "Gradient 실습은 이 네 덩어리만 먼저 읽으면 됩니다",
            "긴 프레임워크 실습은 데이터→모델→평가 흐름만 먼저 보세요",
            "긴 기초 실습은 핵심 계산만 먼저 훑어보세요",
            "전체 코드를 한 번에 읽기 어렵다면, 아래 핵심 발췌",
            "예측값과 loss 만들기",
            "chain rule로 손미분 gradient 계산하기",
            "finite difference로 미분값 검산하기",
            "gradient 방향으로 파라미터 업데이트하기",
            "loss·metric·판정 기준",
            "이 파일은 무엇인가",
            "어떻게 읽으면 좋은가",
            "실행하면 남는 결과",
            "읽어볼 함수",
            "annotateCodeWithInlineHints",
            "# 학습 포인트:",
            "# 코드 읽기 힌트:",
            "functionRoleHintsFor",
            "sectionSpecificRunHint",
            "runPythonSection",
            "staticServerHelp",
            "formatRunnerSummary",
            "data-run-code",
            "run-output",
            "읽은 뒤 실행",
            "이 코드를 내 환경에서 확인하기",
            "501 Unsupported method",
            "기초 실습 코드: 작은 숫자로 원리 확인하기",
            "프레임워크 실습 코드: 실제 도구로 같은 아이디어 확인하기",
            "결과 해석 코드: 실행 결과를 공부 노트로 바꾸기",
            "실험 실행 코드: 한 번에 실행하고 결과 모으기",
            "데이터 준비 코드: 실험에 넣을 표 만들기",
            "실험 흐름 코드: 준비·학습·평가 연결하기",
            "formatStaticServerDetail",
        ]:
            self.assertIn(token, app)

        for removed in [
            "# 학습자용 한글 주석",
            "# 이 파일은 무엇인가",
            "# 어떻게 읽으면 좋은가",
            "# 실행하면 남는 결과",
            "# 아래부터 원본 Python 코드입니다.",
            "# 핵심 함수",
            "핵심 함수",
            "파일을 실행했을 때 전체 흐름이 모이는 진입점입니다.",
            "annotateCodeWithKoreanComments",
            "functionHint",
        ]:
            self.assertNotIn(removed, app)
        self.assertNotIn("stdout/stderr", app + readme)

        self.assertIn("code-explanation", styles)
        self.assertIn("core-code-summary", styles)
        self.assertIn("mini-code", styles)
        self.assertIn("run-panel", styles)
        self.assertIn("run-output", styles)
        self.assertIn("reader-shell", styles)
        self.assertIn("sticky", styles)
        self.assertNotIn("learner-comment", app + styles)
        self.assertIn("npm run qa:web", readme)
        self.assertIn("Playwright", readme)
        self.assertIn("study_server.py", readme)
        self.assertIn("conda", readme)
        self.assertIn("GPU", readme)

    def test_inline_summary_renderer_strips_visible_markdown_syntax(self) -> None:
        module = self._load_builder()
        catalog = module.build_catalog(ROOT)
        track_with_bold = next(track for track in catalog["tracks"] if "**" in track["summary"])
        self.assertIn("공통 기초 트랙", track_with_bold["summary"])

        app = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn("function renderInlineSummary", app)
        self.assertIn("<strong>$1</strong>", app)
        self.assertIn("stripMarkdownLinks", app)

    @unittest.skipIf(shutil.which("node") is None, "node is not installed")
    def test_progress_storage_profiles_corrupt_recovery_and_import(self) -> None:
        script = r'''
const assert = require('assert');
const Progress = require('./web/progress-storage.js');

const store = Progress.defaultProgress();
store.users.alice = { displayName: 'Alice', lessons: {} };
store.users.bob = { displayName: 'Bob', lessons: {} };
Progress.upsertLessonProgress(store, 'alice', '00_foundations/01_tensor_shapes', { state: 'done', percent: 100 }, '2026-06-24T00:00:00Z');
Progress.upsertLessonProgress(store, 'bob', '00_foundations/01_tensor_shapes', { state: 'blocked', percent: 20 }, '2026-06-24T00:01:00Z');
Progress.upsertLessonProgress(store, 'alice', '10_vla/01_vision_language_action_grounding', {
  state: 'in_progress',
  percent: 40,
  selfChecks: { goal: true },
  note: 'wrong-note: action token vs safety gate',
}, '2026-06-24T00:02:00Z');
Progress.updateUserUI(store, 'alice', {
  selectedTrack: '00_foundations',
  selectedUnit: '00_foundations/01_tensor_shapes',
  selectedRoute: 'systems',
  filters: { query: 'tensor', progressState: 'done' },
});
Progress.updateUserUI(store, 'bob', {
  selectedTrack: '10_vla',
  selectedUnit: '10_vla/01_vision_language_action_grounding',
  selectedRoute: 'multimodal',
  filters: { query: 'VLA', progressState: 'blocked' },
});
assert.strictEqual(Progress.lessonState(store, 'alice', '00_foundations/01_tensor_shapes').state, 'done');
assert.strictEqual(Progress.lessonState(store, 'bob', '00_foundations/01_tensor_shapes').state, 'blocked');
assert.strictEqual(Progress.userUI(store, 'alice').selectedUnit, '00_foundations/01_tensor_shapes');
assert.strictEqual(Progress.userUI(store, 'bob').selectedUnit, '10_vla/01_vision_language_action_grounding');
assert.strictEqual(Progress.userUI(store, 'alice').filters.query, 'tensor');
assert.strictEqual(Progress.userUI(store, 'bob').filters.query, 'VLA');
assert.strictEqual(Progress.userUI(store, 'alice').selectedRoute, 'systems');
assert.strictEqual(Progress.userUI(store, 'bob').selectedRoute, 'multimodal');
assert.strictEqual(Progress.lessonState(store, 'alice', '10_vla/01_vision_language_action_grounding').selfChecks.goal, true);
assert.strictEqual(Progress.lessonState(store, 'alice', '10_vla/01_vision_language_action_grounding').note, 'wrong-note: action token vs safety gate');

const storage = Progress.createMemoryStorage();
Progress.saveProgress(store, storage);
const reloaded = Progress.loadProgress(storage);
assert.strictEqual(reloaded.users.alice.lessons['00_foundations/01_tensor_shapes'].percent, 100);
assert.strictEqual(reloaded.users.alice.lessons['10_vla/01_vision_language_action_grounding'].selfChecks.goal, true);
assert.strictEqual(reloaded.users.alice.lessons['10_vla/01_vision_language_action_grounding'].note, 'wrong-note: action token vs safety gate');
assert.strictEqual(reloaded.users.alice.ui.selectedRoute, 'systems');
assert.strictEqual(reloaded.users.bob.ui.selectedRoute, 'multimodal');

const corrupt = Progress.createMemoryStorage({ [Progress.PROGRESS_KEY]: '{broken-json' });
const recovered = Progress.loadProgress(corrupt);
assert.strictEqual(recovered.schemaVersion, 1);
assert.ok(corrupt.keys().some((key) => key.startsWith(`${Progress.PROGRESS_KEY}.corrupt.`)));

Progress.mergeImportedProgress(recovered, {
  schemaVersion: 1,
  activeUserId: 'carol',
  users: { carol: { displayName: 'Carol', lessons: { '10_vla/01_vision_language_action_grounding': { state: 'in_progress' } } } },
  ui: { selectedTrack: '10_vla', selectedUnit: '10_vla/01_vision_language_action_grounding', filters: { progressState: 'in_progress' } }
});
assert.strictEqual(recovered.activeUserId, 'carol');
assert.strictEqual(recovered.users.carol.lessons['10_vla/01_vision_language_action_grounding'].state, 'in_progress');
'''
        result = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_remaining_roadmap_features_are_implemented(self) -> None:
        app = (WEB / "app.js").read_text(encoding="utf-8")
        server = (ROOT / "scripts" / "study_server.py").read_text(encoding="utf-8")
        styles = (WEB / "styles.css").read_text(encoding="utf-8")
        storage = (WEB / "progress-storage.js").read_text(encoding="utf-8")

        for token in [
            "quizForUnit",
            "renderQuizPanel",
            "data-quiz-submit",
            "quizAnswers",
            "단원 점검 퀴즈",
            "답하려면 무엇을 확인해야 하나요",
            "wrongNotes",
            "wrong-note-panel",
            "openMistakeReview",
            "예시와 비교 저장",
            "자동 채점 대신 예시와 비교하세요",
            "artifact-viewer",
            "renderArtifactViewer",
            "renderArtifactCard",
            "artifact-grid",
            "산출물 뷰어",
            "이번 실행에서 새로",
            "지표 요약",
            "partial-experiment",
            "cell-probe",
            "renderCellProbe",
            "선택 함수 미리보기",
            "prereq-gate",
            "data-prereq-href",
            "llmFastPath",
            "!nextCheckpoint && answered < quizItems.length",
            "next-action-card",
        ]:
            self.assertIn(token, app + server + styles + storage)
        for removed in [
            "이 단원의 가장 중요한 학습 목표는 무엇인가요?",
            "실행 후 확인해야 할 산출물을 고르세요.",
            "자기 말로 한 문장으로 설명해 보세요.",
            "미니 퀴즈",
        ]:
            self.assertNotIn(removed, app)

    def test_documented_static_server_resolves_app_and_lesson_links(self) -> None:
        port = self._free_port()
        server = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.addCleanup(self._terminate_process, server)
        base = f"http://127.0.0.1:{port}"
        last_error: Exception | None = None
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"{base}/web/", timeout=1) as response:
                    self.assertEqual(200, response.status)
                    response.read()
                break
            except Exception as exc:  # pragma: no cover - timing dependent
                last_error = exc
                time.sleep(0.1)
        else:
            self.fail(f"http.server did not start: {last_error}")

        with urllib.request.urlopen(f"{base}/", timeout=2) as response:
            self.assertEqual(200, response.status)
            self.assertIn("BTB", response.read().decode("utf-8"))

        for path in [
            "/web/catalog.json",
            "/00_foundations/01_tensor_shapes/README.md",
            "/docs/02_study_guide.md",
            "/docs/05_rl_primer_for_rlhf.md",
            "/10_vla/01_vision_language_action_grounding/README.md",
        ]:
            with urllib.request.urlopen(f"{base}{path}", timeout=2) as response:
                self.assertEqual(200, response.status, path)
                response.read()

    def test_local_study_server_runs_allowlisted_python_files(self) -> None:
        port = self._free_port()
        server = subprocess.Popen(
            [sys.executable, "scripts/study_server.py", "--port", str(port), "--bind", "127.0.0.1"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(self._terminate_process, server)
        base = f"http://127.0.0.1:{port}"
        last_error: Exception | None = None
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"{base}/web/", timeout=1) as response:
                    self.assertEqual(200, response.status)
                    response.read()
                break
            except Exception as exc:  # pragma: no cover - timing dependent
                last_error = exc
                time.sleep(0.1)
        else:
            self.fail(f"study_server.py did not start: {last_error}; stderr={server.stderr.read() if server.stderr else ''}")

        artifact_dir = ROOT / "00_foundations" / "01_tensor_shapes" / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        stale_artifact = artifact_dir / "stale-from-test.txt"
        stale_artifact.write_text("이 파일은 이번 실행 결과가 아닙니다.", encoding="utf-8")
        self.addCleanup(lambda: stale_artifact.exists() and stale_artifact.unlink())
        symlink_artifact = artifact_dir / "symlink-from-test.txt"
        if symlink_artifact.exists() or symlink_artifact.is_symlink():
            symlink_artifact.unlink()
        if hasattr(symlink_artifact, "symlink_to"):
            try:
                symlink_artifact.symlink_to(ROOT / "README.md")
                self.addCleanup(lambda: (symlink_artifact.exists() or symlink_artifact.is_symlink()) and symlink_artifact.unlink())
            except OSError:
                pass

        request = urllib.request.Request(
            f"{base}/api/run-python",
            data=json.dumps({"path": "00_foundations/01_tensor_shapes/scratch_lab.py"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            self.assertEqual(200, response.status)
            payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(0, payload["returncode"])
        self.assertIn("00_foundations/01_tensor_shapes/scratch_lab.py", payload["path"])
        self.assertFalse(Path(payload["path"]).is_absolute(), payload["path"])
        self.assertEqual("00_foundations/01_tensor_shapes/scratch_lab.py", payload["command"][-1])
        self.assertFalse(Path(payload["command"][-1]).is_absolute(), payload["command"])
        self.assertNotIn(str(ROOT), json.dumps(payload, ensure_ascii=False))
        self.assertIn("matmul_shape", payload["stdout"])
        self.assertIn(payload["runner"]["device"], {"cpu", "cuda"})
        self.assertIn("python", payload["runner"])
        self.assertIn("environment", payload["runner"])
        artifact_paths = [artifact["path"] for artifact in payload["artifacts"]]
        self.assertTrue(artifact_paths, "run-python should report changed artifacts")
        self.assertTrue(all(not Path(path).is_absolute() for path in artifact_paths), artifact_paths)
        self.assertTrue(
            any(path.endswith("metrics.json") for path in artifact_paths),
            f"run-python should return previewable artifacts, got: {artifact_paths}",
        )
        self.assertFalse(any("stale-from-test" in path for path in artifact_paths), artifact_paths)
        self.assertFalse(any("symlink-from-test" in path for path in artifact_paths), artifact_paths)
        metric_artifact = next(artifact for artifact in payload["artifacts"] if artifact["path"].endswith("metrics.json"))
        self.assertEqual("json", metric_artifact["type"])
        self.assertEqual("json", metric_artifact["preview"]["kind"])

        cell_probe = urllib.request.Request(
            f"{base}/api/partial-experiment",
            data=json.dumps({"path": "00_foundations/01_tensor_shapes/scratch_lab.py", "symbol": "run"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(cell_probe, timeout=10) as response:
            self.assertEqual(200, response.status)
            cell_payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(0, cell_payload["returncode"])
        self.assertEqual("function_probe", cell_payload["cell"]["mode"])
        self.assertEqual("run", cell_payload["cell"]["symbol"])
        self.assertIn("line_range", cell_payload["cell"])
        self.assertIn("source_excerpt", cell_payload["cell"])
        self.assertIn("ARTIFACT_DIR", cell_payload["cell"]["artifact_names"])

        forbidden = urllib.request.Request(
            f"{base}/api/run-python",
            data=json.dumps({"path": "README.md"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(forbidden, timeout=5)
        self.assertEqual(403, ctx.exception.code)

        server_module = self._load_study_server()
        runnable = server_module._resolve_runnable_path("01_ml/01_tabular_classification/run_stage.py")
        self.assertEqual(ROOT / "01_ml" / "01_tabular_classification" / "run_stage.py", runnable)

    def test_study_server_builds_conda_gpu_and_cpu_fallback_invocations(self) -> None:
        server = self._load_study_server()
        script_path = ROOT / "00_foundations" / "01_tensor_shapes" / "scratch_lab.py"
        gpu_rows = server._parse_nvidia_smi_rows("0, 24000, 128, 0\n1, 16000, 12000, 95\n")

        conda_config = server.RunnerConfig(conda_env="btb", device="auto")
        command, env, runner = server._build_runner_invocation(script_path, conda_config, gpu_rows)
        self.assertEqual(["conda", "run", "--no-capture-output", "-n", "btb", "python"], command[:6])
        self.assertEqual("cuda", runner["device"])
        self.assertEqual("0", runner["gpu_index"])
        self.assertEqual("0", env["CUDA_VISIBLE_DEVICES"])
        self.assertEqual("cuda", env["BTB_DEVICE"])
        self.assertEqual("conda:btb", runner["environment"])

        cpu_config = server.RunnerConfig(device="auto")
        command, env, runner = server._build_runner_invocation(script_path, cpu_config, [])
        self.assertEqual(sys.executable, command[0])
        self.assertEqual("python", runner["python"])
        self.assertEqual("python:current", runner["environment"])
        self.assertEqual("cpu", runner["device"])
        self.assertEqual("", env["CUDA_VISIBLE_DEVICES"])
        self.assertEqual("cpu", env["BTB_DEVICE"])

    def test_study_server_rejects_symlinked_artifact_roots(self) -> None:
        server = self._load_study_server()
        with tempfile.TemporaryDirectory(dir=ROOT) as unit_dir, tempfile.TemporaryDirectory() as external_dir:
            unit_path = Path(unit_dir)
            external_path = Path(external_dir)
            script_path = unit_path / "scratch_lab.py"
            script_path.write_text("print('ok')\n", encoding="utf-8")
            (external_path / "secret_metrics.json").write_text('{"secret": true}', encoding="utf-8")
            (unit_path / "artifacts").symlink_to(external_path, target_is_directory=True)

            self.assertEqual({}, server._artifact_snapshot(script_path))
            self.assertEqual([], server._collect_artifacts(script_path))

    def test_catalog_builder_covers_manifest_tracks_and_units(self) -> None:
        module = self._load_builder()
        catalog = module.build_catalog(ROOT)

        self.assertEqual(1, catalog["schema_version"])
        self.assertEqual(CANONICAL_CURRICULUM_LADDER, [track["id"] for track in catalog["tracks"]])

        status = json.loads((ROOT / "docs" / "curriculum_status.json").read_text(encoding="utf-8"))["tracks"]
        catalog_units = {
            track["id"]: {unit["id"]: unit for unit in track["units"]}
            for track in catalog["tracks"]
        }
        for track_id, units in status.items():
            self.assertEqual(set(units), set(catalog_units[track_id]))
            for unit_id, status_value in units.items():
                unit = catalog_units[track_id][unit_id]
                self.assertEqual(status_value, unit["status"])
                self.assertEqual(f"{track_id}/{unit_id}", unit["path"])
                self.assertTrue(unit["readme"].endswith("README.md"))
                self.assertIn("checkpoints", unit)
                self.assertIn("README", unit["checkpoints"])

        vla_units = catalog_units["10_vla"]
        self.assertIn("01_vision_language_action_grounding", vla_units)
        self.assertIn("VLA", vla_units["01_vision_language_action_grounding"]["title"])

        ml_unit = catalog_units["01_ml"]["01_tabular_classification"]
        ml_labels = [resource["label"] for resource in ml_unit["resources"]]
        self.assertIn("README", ml_labels)
        self.assertIn("THEORY", ml_labels)
        self.assertIn("dataset.py", ml_labels)
        self.assertIn("experiment.py", ml_labels)
        self.assertIn("run_stage.py", ml_labels)
        self.assertIn("analysis.py", ml_labels)
        self.assertNotIn("scratch_lab.py", ml_labels)
        self.assertNotIn("framework_lab.py", ml_labels)
        self.assertIn("실습 구성", ml_unit["checkpoints"])
        self.assertTrue(ml_unit["objective"])
        for unit in catalog_units["01_ml"].values():
            for field in ["prereqs", "key_terms", "required_outputs", "analysis_questions"]:
                self.assertTrue(unit[field], f"{unit['path']} needs learner-facing metadata for {field}")

        llm_track = next(track for track in catalog["tracks"] if track["id"] == "05_advanced_nlp_llm")
        self.assertFalse(llm_track["summary"].endswith("preference optimizat"))
        self.assertIn("preference optimization", llm_track["summary"])

        nlp_tokenization = catalog_units["03_nlp_bridge"]["01_tokenization_and_embeddings"]
        llm_objectives = catalog_units["05_advanced_nlp_llm"]["01_language_modeling_and_pretraining_objectives"]
        accelerate = catalog_units["06_training_systems"]["02_accelerate_workflows"]
        self.assertIn("[UNK]가 생기면 어떤 정보 손실이 생기는가?", nlp_tokenization["analysis_questions"])
        self.assertIn("[MASK] 및 sentinel token의 역할 이해", llm_objectives["prereqs"])
        self.assertIn("`prepare()` 이후에도 사용자가 직접 이해해야 하는 것은 무엇인가?", accelerate["analysis_questions"])
        for unit in [nlp_tokenization, llm_objectives, accelerate]:
            for field in ["prereqs", "key_terms", "required_outputs", "analysis_questions"]:
                for item in unit[field]:
                    self.assertFalse(
                        (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")),
                        f"{unit['path']} {field} leaked YAML quote characters: {item}",
                    )

    def test_committed_catalog_matches_builder_output(self) -> None:
        module = self._load_builder()
        expected = module.build_catalog(ROOT)
        committed = json.loads((WEB / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(expected, committed)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "catalog.json"
            result = subprocess.run(
                [sys.executable, str(BUILDER), "--root", str(ROOT), "--output", str(output)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(expected, json.loads(output.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
