from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestAgentRoleDocs(unittest.TestCase):
    def test_agent_role_docs_exist(self) -> None:
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


if __name__ == '__main__':
    unittest.main()
