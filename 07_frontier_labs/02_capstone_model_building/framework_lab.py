"""Framework-style deterministic project board for the capstone unit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


UNIT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_DIR / "artifacts" / "framework-manual"
BOARD_PATH = ARTIFACT_DIR / "project_board.json"


def build_project_board() -> dict[str, Any]:
    milestone_board = [
        {
            "id": "M0",
            "state": "done",
            "decision_closed": "problem_scope_frozen",
            "exit_artifacts": ["problem_statement.md", "non_goals.md", "dataset_split_manifest.json"],
        },
        {
            "id": "M1",
            "state": "ready",
            "decision_closed": "baseline_reproducibility",
            "exit_artifacts": ["baseline_card.md", "eval_metrics.json", "seed_and_runtime_log.json"],
        },
        {
            "id": "M2",
            "state": "planned",
            "decision_closed": "candidate_improvement_claim",
            "exit_artifacts": ["ablation_table.md", "slice_metrics.json", "qualitative_review.md"],
        },
        {
            "id": "M3",
            "state": "planned",
            "decision_closed": "interpretable_capstone_report",
            "exit_artifacts": ["failure_analysis_table.md", "risk_register_update.md", "final_capstone_report.md"],
        },
    ]

    risk_register = [
        {
            "id": "scope_creep",
            "gate": "M0",
            "owner_action": "reject serving/personalization requests until next_steps",
            "current_status": "controlled_by_non_goals",
        },
        {
            "id": "dataset_leakage",
            "gate": "M1",
            "owner_action": "review near_duplicate_group_holdout before baseline run",
            "current_status": "open_until_split_manifest_review",
        },
        {
            "id": "metric_gaming",
            "gate": "M2",
            "owner_action": "block pass verdict if brand/OCR slices regress",
            "current_status": "open_until_slice_metrics_exist",
        },
        {
            "id": "report_drift",
            "gate": "M3",
            "owner_action": "map every run artifact to a report section",
            "current_status": "open_until_final_report",
        },
    ]

    acceptance_gate_verdicts = [
        {"gate": "problem_scope_frozen", "verdict": "pass", "evidence": "problem statement and non-goals are explicit"},
        {"gate": "baseline_reproduces_on_fixed_split", "verdict": "ready", "evidence": "split and evaluator are specified"},
        {"gate": "target_delta_exceeds_minimum_with_slice_review", "verdict": "not_started", "evidence": "candidate metrics not generated in this CPU-safe lesson"},
        {"gate": "report_ready_with_failure_table", "verdict": "blocked_until_artifacts_complete", "evidence": "failure table is outlined but not populated by real runs"},
    ]

    return {
        "status": "runnable",
        "cpu_safe_simulation": True,
        "framework": "cpu_capstone_project_board_sim",
        "project_id": "korean_catalog_retrieval_capstone",
        "dataset_model_eval_matrix": {
            "dataset": "fixed synthetic split",
            "model_comparison": "lexical baseline vs tiny dual encoder",
            "eval_protocol": "Recall@10 + slice review",
            "frozen_invariants": ["same split", "same evaluator", "same qualitative buckets"],
        },
        "milestone_board": milestone_board,
        "acceptance_gate_verdicts": acceptance_gate_verdicts,
        "risk_register": risk_register,
        "report_outline": [
            "problem_and_non_goals",
            "dataset_contract",
            "baseline_and_model_choices",
            "eval_protocol",
            "milestone_results",
            "failure_analysis",
            "risk_register",
            "next_steps",
        ],
        "failure_report_template": {
            "row_shape": ["slice", "failure_bucket", "evidence", "hypothesis", "next_action"],
            "example_row": {
                "slice": "brand queries",
                "failure_bucket": "brand_mismatch",
                "evidence": "top-10 retrieves same category but wrong brand",
                "hypothesis": "brand token is underweighted in tiny embedding features",
                "next_action": "add brand-weighted ablation before expanding model size",
            },
        },
        "handoff_to_agentic_loop": {
            "planner_inputs": ["problem_statement", "frozen_constraints", "acceptance_gates"],
            "verifier_inputs": ["artifact_manifest", "protocol_match", "slice_metrics"],
            "stop_rules": [
                "retry budget exhausted before target_delta clears variance band",
                "benchmark or split drift is detected",
                "high-risk slice regresses after candidate change",
            ],
        },
    }


def main() -> None:
    board = build_project_board()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    BOARD_PATH.write_text(json.dumps(board, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(board, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
