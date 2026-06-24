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
        self.assertIn("catalog.json", html)
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

        self.assertIn("track.summary", app)
        self.assertIn("renderInlineSummary(track.summary", app)
        self.assertNotIn("escapeHtml(track.summary", app)
        self.assertIn("선행 확인", app)
        self.assertIn("학습 방향", app)
        self.assertIn("../docs/02_study_guide.md", app)
        self.assertIn("../docs/05_rl_primer_for_rlhf.md", app)
        self.assertIn("09 Multimodal recap", app)
        self.assertIn("학습 순서", app)
        self.assertIn("이론 읽기", app)
        self.assertIn("scratch 실행", app)
        self.assertIn("framework 실행", app)
        self.assertIn("analysis 정리", app)
        self.assertIn("reflection 작성", app)
        self.assertIn("recommendedStartingUnit", app)

        self.assertIn("학습 자료", app)
        self.assertIn("fetchLessonDocument", app)
        self.assertIn("renderMarkdown", app)
        self.assertIn("lessonSectionsFor", app)
        for label in ["README", "THEORY", "PREREQS", "scratch_lab.py", "framework_lab.py", "analysis.py", "reflection.md"]:
            self.assertIn(label, app)
        self.assertNotIn("README 열기", app)
        self.assertNotIn('target="_blank"', app)

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
        self.assertIn("README 파일을 새 탭으로 직접 여는 방식", root_readme + web_readme)

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
            "이 파일은 무엇인가",
            "어떻게 읽으면 좋은가",
            "실행하면 남는 결과",
            "핵심 함수",
            "annotateCodeWithInlineHints",
            "# 학습 포인트:",
            "# 핵심 함수",
            "runPythonSection",
            "staticServerHelp",
            "formatRunnerSummary",
            "data-run-code",
            "run-output",
            "501 Unsupported method",
            "scratch_lab.py는",
            "framework_lab.py는",
            "analysis.py는",
        ]:
            self.assertIn(token, app)

        for removed in [
            "# 학습자용 한글 주석",
            "# 이 파일은 무엇인가",
            "# 어떻게 읽으면 좋은가",
            "# 실행하면 남는 결과",
            "# 아래부터 원본 Python 코드입니다.",
            "annotateCodeWithKoreanComments",
        ]:
            self.assertNotIn(removed, app)

        self.assertIn("code-explanation", styles)
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
Progress.updateUserUI(store, 'alice', { selectedTrack: '00_foundations', selectedUnit: '00_foundations/01_tensor_shapes', filters: { query: 'tensor', progressState: 'done' } });
Progress.updateUserUI(store, 'bob', { selectedTrack: '10_vla', selectedUnit: '10_vla/01_vision_language_action_grounding', filters: { query: 'VLA', progressState: 'blocked' } });
assert.strictEqual(Progress.lessonState(store, 'alice', '00_foundations/01_tensor_shapes').state, 'done');
assert.strictEqual(Progress.lessonState(store, 'bob', '00_foundations/01_tensor_shapes').state, 'blocked');
assert.strictEqual(Progress.userUI(store, 'alice').selectedUnit, '00_foundations/01_tensor_shapes');
assert.strictEqual(Progress.userUI(store, 'bob').selectedUnit, '10_vla/01_vision_language_action_grounding');
assert.strictEqual(Progress.userUI(store, 'alice').filters.query, 'tensor');
assert.strictEqual(Progress.userUI(store, 'bob').filters.query, 'VLA');

const storage = Progress.createMemoryStorage();
Progress.saveProgress(store, storage);
const reloaded = Progress.loadProgress(storage);
assert.strictEqual(reloaded.users.alice.lessons['00_foundations/01_tensor_shapes'].percent, 100);

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
        self.assertIn("matmul_shape", payload["stdout"])
        self.assertIn(payload["runner"]["device"], {"cpu", "cuda"})
        self.assertIn("python", payload["runner"])
        self.assertIn("environment", payload["runner"])

        forbidden = urllib.request.Request(
            f"{base}/api/run-python",
            data=json.dumps({"path": "README.md"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(forbidden, timeout=5)
        self.assertEqual(403, ctx.exception.code)

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
        self.assertEqual("cpu", runner["device"])
        self.assertEqual("", env["CUDA_VISIBLE_DEVICES"])
        self.assertEqual("cpu", env["BTB_DEVICE"])

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
