from __future__ import annotations

import json
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / "artifacts" / "framework-manual"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def build_framework_contract() -> dict[str, object]:
    return {
        "status": "runnable",
        "framework": "cpu_deterministic_agentic_loop_contract",
        "contract_id": "framework-agentic-loop-contract-v1",
        "role_contract": {
            "separation_order": ["planner", "executor", "verifier", "critic"],
            "planner_outputs": ["experiment_contract", "bounded_change_set", "retry_budget", "stop_rule"],
            "executor_outputs": ["run_log", "config_hash", "metric_json", "artifact_manifest", "failure_slice_report"],
            "verifier_outputs": ["protocol_match", "artifact_completeness", "baseline_comparability", "benchmark_drift"],
            "critic_outputs": ["triage_verdict", "evidence_refs", "next_action", "escalation_note"],
            "anti_self_approval_rules": [
                "planner_does_not_approve_own_plan",
                "executor_does_not_interpret_metric_claims",
                "verifier_blocks_protocol_mismatch_even_when_metric_improves",
                "critic_must_reference_evidence_fields",
            ],
        },
        "experiment_contract": {
            "task": "agentic_retrieval_eval",
            "primary_metric": "recall_at_10",
            "baseline_value": 0.412,
            "variance_band": 0.015,
            "frozen_protocol_fields": [
                "dataset_version=frozen_dev_v1",
                "split=dev",
                "metric=Recall@10",
                "preprocessing=capstone_preprocess_v3",
                "budget_tier=cpu_teaching_simulation",
            ],
        },
        "retry_policy": {
            "max_retries": 3,
            "attempts_used": 2,
            "retry_allowed_when": [
                "protocol_match is true",
                "artifact_completeness is true",
                "delta exceeds variance band but needs seed confirmation",
            ],
            "retry_blocked_when": [
                "protocol mismatch",
                "artifact gap",
                "benchmark drift above threshold",
                "same failure repeats twice",
            ],
        },
        "stop_rules": [
            "same_failure_twice",
            "delta_inside_variance_band_after_two_valid_runs",
            "artifact_manifest_incomplete_after_retry",
            "benchmark_drift_above_0.12",
        ],
        "escalation_rules": [
            "protocol_mismatch_changes_comparability",
            "benchmark_drift_above_0.12",
            "critic_cannot_cite_evidence_refs",
            "human_claim_boundary_needed_before_public_result",
        ],
        "gate_summary": {
            "protocol_match_required": True,
            "artifact_completeness_required": True,
            "baseline_comparability_required": True,
            "evidence_bundle_required": True,
            "final_gate": "needs_human_review",
            "why": "Benchmark drift and long-tail slice regression make more automated retries less trustworthy than a benchmark review.",
        },
        "evidence_bundle": {
            "required_fields": [
                "experiment_contract_id",
                "planner_change_set",
                "seed",
                "config_hash",
                "dataset_version",
                "eval_protocol_version",
                "run_log",
                "metric_json",
                "failure_slice_report",
                "artifact_manifest",
                "verifier_gate",
                "critic_triage",
            ],
            "complete_example_count": 3,
            "incomplete_example_count": 1,
        },
        "benchmark_drift": {
            "threshold": 0.12,
            "observed_score": 0.18,
            "signals": [
                "long_tail_slice_regression",
                "drift_probe_warning",
                "metric_gain_not_reproduced_across_seed",
            ],
            "decision": "pause_loop_and_review_benchmark_contract",
        },
        "artifacts": {
            "metrics": str(METRICS_PATH.relative_to(UNIT_ROOT)),
        },
    }


def run() -> dict[str, object]:
    contract = build_framework_contract()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    return contract


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
