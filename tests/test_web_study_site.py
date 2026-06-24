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
import urllib.request
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

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


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
        self.assertIn("http://localhost:8000/web/", readme)
        self.assertNotIn("python -m http.server -d web", readme + app)

    def test_progress_code_is_local_only(self) -> None:
        combined = (WEB / "app.js").read_text(encoding="utf-8") + "\n" + (WEB / "progress-storage.js").read_text(encoding="utf-8")
        forbidden_network_writes = [
            "navigator.sendBeacon",
            "XMLHttpRequest",
            "fetch('/progress",
            'fetch("/progress',
            "method: 'POST'",
            'method: "POST"',
            "method:'POST'",
            'method:"POST"',
        ]
        for snippet in forbidden_network_writes:
            self.assertNotIn(snippet, combined)

        self.assertIn("storage.setItem(PROGRESS_KEY", combined)
        self.assertIn("activeUserId", combined)
        self.assertIn("lessons", combined)

    def test_ui_surfaces_learning_process_metadata(self) -> None:
        app = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn("track.summary", app)
        self.assertIn("선행 확인", app)
        self.assertIn("학습 방향", app)
        self.assertIn("../docs/02_study_guide.md", app)
        self.assertIn("../docs/05_rl_primer_for_rlhf.md", app)
        self.assertIn("09 Multimodal recap", app)

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
assert.strictEqual(Progress.lessonState(store, 'alice', '00_foundations/01_tensor_shapes').state, 'done');
assert.strictEqual(Progress.lessonState(store, 'bob', '00_foundations/01_tensor_shapes').state, 'blocked');

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
        self.addCleanup(server.terminate)
        base = f"http://127.0.0.1:{port}"
        last_error: Exception | None = None
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"{base}/web/", timeout=1) as response:
                    self.assertEqual(200, response.status)
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
