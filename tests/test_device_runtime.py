from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

from shared.device_runtime import resolve_torch_device


ROOT = Path(__file__).resolve().parents[1]


class DeviceRuntimeContractTest(unittest.TestCase):
    def test_explicit_cpu_is_always_respected(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual("cpu", resolve_torch_device("cpu").type)

    def test_auto_uses_cuda_when_available(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual("cuda", resolve_torch_device("auto").type)

    def test_forced_cuda_fails_when_cuda_is_unavailable(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                resolve_torch_device("cuda")

    def test_environment_value_is_used_and_invalid_values_are_rejected(self) -> None:
        with mock.patch.dict(os.environ, {"BTB_DEVICE": "invalid"}, clear=False):
            with self.assertRaisesRegex(ValueError, "auto, cpu, or cuda"):
                resolve_torch_device()

    def test_gpu_runtime_lesson_uses_the_shared_device_contract(self) -> None:
        result = subprocess.run(
            [sys.executable, "00_foundations/05_gpu_memory_runtime/framework_lab.py"],
            cwd=ROOT,
            env={**os.environ, "BTB_DEVICE": "invalid"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("auto, cpu, or cuda", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
