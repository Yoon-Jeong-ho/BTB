"""Analyze deterministic capstone contract artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


UNIT_DIR = Path(__file__).resolve().parent
DEFAULT_SCRATCH_CONTRACT = UNIT_DIR / "artifacts" / "scratch-manual" / "capstone_contract.json"
DEFAULT_FRAMEWORK_BOARD = UNIT_DIR / "artifacts" / "framework-manual" / "project_board.json"
DEFAULT_OUTPUT = UNIT_DIR / "artifacts" / "analysis-manual" / "latest_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize capstone model-building contract artifacts.")
    parser.add_argument("--scratch-contract", type=Path, default=DEFAULT_SCRATCH_CONTRACT)
    parser.add_argument("--framework-board", type=Path, default=DEFAULT_FRAMEWORK_BOARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        print(
            f"Missing required capstone artifact: {joined}. Run scratch_lab.py and framework_lab.py first.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def build_observed_summary(contract: dict[str, Any], board: dict[str, Any]) -> dict[str, Any]:
    eval_contract = contract["eval_contract"]
    target_delta = round(eval_contract["target_score"] - eval_contract["baseline_score"], 2)
    return {
        "status": "runnable",
        "project_id": contract["project_id"],
        "problem_statement": contract["problem_statement"],
        "primary_metric": eval_contract["primary_metric"],
        "baseline_score": eval_contract["baseline_score"],
        "target_score": eval_contract["target_score"],
        "target_delta": target_delta,
        "milestone_ids": [milestone["id"] for milestone in contract["milestones"]],
        "acceptance_gates": contract["acceptance_gates"],
        "final_gate_verdict": board["acceptance_gate_verdicts"][-1]["verdict"],
        "top_risks": [risk["id"] for risk in contract["risk_register"][:3]],
        "failure_columns": contract["failure_analysis_outline"]["columns"],
        "report_outline": board["report_outline"],
        "next_handoff": board["handoff_to_agentic_loop"]["planner_inputs"],
        "observed_report": "artifacts/analysis-manual/latest_report.md",
    }


def render_report(contract: dict[str, Any], board: dict[str, Any], observed: dict[str, Any]) -> str:
    non_goals = "\n".join(f"- {item}" for item in contract["non_goals"])
    gates = "\n".join(
        f"- {gate['gate']}: {gate['verdict']} — {gate['evidence']}"
        for gate in board["acceptance_gate_verdicts"]
    )
    risks = "\n".join(
        f"- {risk['id']} ({risk['severity']}): {risk['mitigation']}"
        for risk in contract["risk_register"]
    )
    failure_columns = ", ".join(contract["failure_analysis_outline"]["columns"])
    report_outline = "\n".join(f"- {section}" for section in observed["report_outline"])
    return f"""# 02 Capstone Model Building 실행 관측

## Problem statement / non-goals
{contract['problem_statement']}

{non_goals}

## Dataset / model / eval contract
- Dataset: {contract['dataset_contract']['source']} / split {contract['dataset_contract']['split']}
- Model: {contract['model_contract']['baseline']} vs {', '.join(contract['model_contract']['candidates'])}
- Eval: {observed['primary_metric']} baseline {observed['baseline_score']} → target {observed['target_score']} (delta {observed['target_delta']})
- Frozen constraints: {', '.join(contract['model_contract']['frozen_constraints'])}

## Acceptance gates
{gates}

## Risk register
{risks}

## Failure-analysis outline
Columns: {failure_columns}

Required slices: {', '.join(contract['failure_analysis_outline']['required_slices'])}

## Report outline
{report_outline}

## Korean-first reading
이 capstone은 모델을 크게 만드는 일이 아니라, problem statement / non-goals / dataset contract / model contract / eval contract / acceptance gate / risk register / failure analysis를 같은 문서 계약 안에 묶는 연습이다. 마지막 gate가 `{observed['final_gate_verdict']}`인 이유는 실제 실험 artifact가 아직 없다는 사실을 숨기지 않기 위해서다.
"""


def main() -> None:
    args = parse_args()
    require([args.scratch_contract, args.framework_board])
    contract = load_json(args.scratch_contract)
    board = load_json(args.framework_board)
    observed = build_observed_summary(contract, board)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(contract, board, observed), encoding="utf-8")
    print(json.dumps(observed, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
