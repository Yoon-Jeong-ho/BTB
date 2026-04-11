from __future__ import annotations

import json
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / "artifacts" / "scratch-manual"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
TRACE_PATH = ARTIFACT_DIR / "iteration_trace.jsonl"
SVG_PATH = ARTIFACT_DIR / "agentic_loop_trace.svg"


ExperimentIteration = dict[str, object]


def experiment_contract() -> dict[str, object]:
    return {
        "contract_id": "agentic-retrieval-loop-contract-v1",
        "task": "agentic_retrieval_eval",
        "goal": "Improve Recall@10 against the frozen capstone baseline without changing the evaluation protocol.",
        "baseline": {"recall_at_10": 0.412, "slice": "frozen_dev_v1", "variance_band": 0.015},
        "primary_metric": "recall_at_10",
        "frozen_constraints": [
            "same_eval_split",
            "same_metric_definition",
            "same_preprocessing_pipeline",
            "same_budget_tier",
            "same_artifact_schema",
        ],
        "retry_budget": 3,
        "stop_rules": [
            "stop if the same verifier failure appears twice",
            "stop if improvement is below variance band after two valid retries",
            "stop if benchmark drift exceeds 0.12",
        ],
        "escalation_rules": [
            "escalate on protocol mismatch that changes comparability",
            "escalate on missing evidence bundle fields after retry",
            "escalate on benchmark drift or contamination warning",
        ],
    }


def build_iterations() -> list[ExperimentIteration]:
    return [
        {
            "iteration": 1,
            "planner": {
                "proposal_id": "plan-001",
                "change_set": ["learning_rate=3e-5"],
                "frozen": ["eval_split", "metric", "preprocessing", "budget"],
                "expected_evidence": ["config_hash", "eval_metrics", "failure_slice_report"],
            },
            "executor": {
                "run_id": "run-001",
                "status": "finished",
                "seed": 17,
                "runtime_minutes": 24,
                "config_hash": "cfg-a17e",
                "artifacts": ["train_log.jsonl", "eval_metrics.json", "failure_slices.json", "config.json"],
                "metric": {"recall_at_10": 0.431, "delta_vs_baseline": 0.019},
            },
            "verifier": {
                "protocol_match": True,
                "artifact_complete": True,
                "baseline_comparable": True,
                "benchmark_drift_score": 0.03,
                "warnings": ["single seed; variance still uncertain"],
            },
            "critic": {
                "verdict": "retry_with_seed_confirmation",
                "evidence_refs": ["eval_metrics.json", "failure_slices.json"],
                "reason": "Gain clears variance band once, but one seed is not enough for a stronger claim.",
                "next_action": "repeat same change with seed 23",
            },
        },
        {
            "iteration": 2,
            "planner": {
                "proposal_id": "plan-002",
                "change_set": ["hard_negative_ratio=0.35"],
                "frozen": ["eval_split", "metric", "budget"],
                "expected_evidence": ["config_hash", "eval_metrics", "artifact_manifest"],
            },
            "executor": {
                "run_id": "run-002",
                "status": "finished_with_warning",
                "seed": 23,
                "runtime_minutes": 28,
                "config_hash": "cfg-b91c",
                "artifacts": ["train_log.jsonl", "eval_metrics.json"],
                "metric": {"recall_at_10": 0.447, "delta_vs_baseline": 0.035},
                "operator_note": "A convenience preprocessing cache was refreshed during the run.",
            },
            "verifier": {
                "protocol_match": False,
                "artifact_complete": False,
                "baseline_comparable": False,
                "benchmark_drift_score": 0.07,
                "missing_artifacts": ["failure_slices.json", "config.json"],
                "warnings": ["preprocessing cache changed", "artifact manifest incomplete"],
            },
            "critic": {
                "verdict": "rollback_protocol_mismatch",
                "evidence_refs": ["operator_note", "missing_artifacts"],
                "reason": "Metric is not comparable because a frozen preprocessing path changed and the evidence bundle is incomplete.",
                "next_action": "rollback and rerun only after restoring frozen preprocessing",
            },
        },
        {
            "iteration": 3,
            "planner": {
                "proposal_id": "plan-003",
                "change_set": ["learning_rate=3e-5", "seed=23"],
                "frozen": ["eval_split", "metric", "preprocessing", "budget"],
                "expected_evidence": ["config_hash", "eval_metrics", "failure_slice_report", "verifier_gate"],
            },
            "executor": {
                "run_id": "run-003",
                "status": "finished",
                "seed": 23,
                "runtime_minutes": 25,
                "config_hash": "cfg-a17e",
                "artifacts": ["train_log.jsonl", "eval_metrics.json", "failure_slices.json", "config.json"],
                "metric": {"recall_at_10": 0.425, "delta_vs_baseline": 0.013},
            },
            "verifier": {
                "protocol_match": True,
                "artifact_complete": True,
                "baseline_comparable": True,
                "benchmark_drift_score": 0.04,
                "warnings": ["delta is inside variance band"],
            },
            "critic": {
                "verdict": "stop_low_information_retry",
                "evidence_refs": ["eval_metrics.json", "verifier_gate"],
                "reason": "Second comparable run falls inside the variance band, so more identical retries add little information.",
                "next_action": "try one bounded diagnostic slice check before claiming improvement",
            },
        },
        {
            "iteration": 4,
            "planner": {
                "proposal_id": "plan-004",
                "change_set": ["diagnostic_failure_slice=long_tail_queries"],
                "frozen": ["eval_split", "metric", "preprocessing", "budget"],
                "expected_evidence": ["slice_metrics", "drift_probe", "critic_triage"],
            },
            "executor": {
                "run_id": "run-004",
                "status": "finished",
                "seed": 17,
                "runtime_minutes": 9,
                "config_hash": "cfg-slice4",
                "artifacts": ["slice_metrics.json", "drift_probe.json", "artifact_manifest.json"],
                "metric": {"long_tail_recall_at_10": 0.392, "delta_vs_baseline": -0.021},
            },
            "verifier": {
                "protocol_match": True,
                "artifact_complete": True,
                "baseline_comparable": True,
                "benchmark_drift_score": 0.18,
                "warnings": ["benchmark drift exceeds contract threshold", "long-tail slice regressed"],
            },
            "critic": {
                "verdict": "stop_and_escalate",
                "evidence_refs": ["drift_probe.json", "slice_metrics.json"],
                "reason": "The loop is now measuring a drifting benchmark signal rather than a trustworthy model improvement.",
                "next_action": "escalate_to_human_benchmark_review",
            },
        },
    ]


def summarize_iterations(iterations: list[ExperimentIteration]) -> dict[str, object]:
    valid = [item for item in iterations if item["verifier"]["protocol_match"] and item["verifier"]["artifact_complete"]]
    rejected = [item for item in iterations if not item["verifier"]["protocol_match"] or not item["verifier"]["artifact_complete"]]
    drift_scores = [float(item["verifier"]["benchmark_drift_score"]) for item in iterations]
    attempts_used = sum(1 for item in iterations if "retry" in str(item["critic"]["verdict"]))
    return {
        "valid_iteration_count": len(valid),
        "rejected_iteration_count": len(rejected),
        "attempts_used": attempts_used,
        "max_benchmark_drift_score": round(max(drift_scores), 3),
        "last_comparable_delta": valid[-1]["executor"]["metric"].get("delta_vs_baseline"),
        "dominant_risks": ["protocol_mismatch", "artifact_gap", "benchmark_drift"],
    }


def write_trace(iterations: list[ExperimentIteration]) -> None:
    lines = []
    for item in iterations:
        lines.append(
            json.dumps(
                {
                    "iteration": item["iteration"],
                    "planner_change_set": item["planner"]["change_set"],
                    "executor_run_id": item["executor"]["run_id"],
                    "protocol_match": item["verifier"]["protocol_match"],
                    "artifact_complete": item["verifier"]["artifact_complete"],
                    "critic_verdict": item["critic"]["verdict"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    TRACE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(iterations: list[ExperimentIteration]) -> None:
    boxes: list[str] = []
    x = 52
    colors = {
        "retry_with_seed_confirmation": "#dbeafe",
        "rollback_protocol_mismatch": "#fee2e2",
        "stop_low_information_retry": "#fef3c7",
        "stop_and_escalate": "#f3e8ff",
    }
    for item in iterations:
        verdict = str(item["critic"]["verdict"])
        fill = colors[verdict]
        y = 112
        boxes.append(
            f'<rect x="{x}" y="{y}" width="170" height="145" rx="12" fill="{fill}" stroke="#334155"/>'
        )
        boxes.append(
            f'<text x="{x + 16}" y="{y + 28}" font-family="monospace" font-size="14" fill="#0f172a">iteration {item["iteration"]}</text>'
        )
        boxes.append(
            f'<text x="{x + 16}" y="{y + 54}" font-family="monospace" font-size="11" fill="#334155">plan → exec</text>'
        )
        boxes.append(
            f'<text x="{x + 16}" y="{y + 76}" font-family="monospace" font-size="11" fill="#334155">protocol={str(item["verifier"]["protocol_match"]).lower()}</text>'
        )
        boxes.append(
            f'<text x="{x + 16}" y="{y + 98}" font-family="monospace" font-size="11" fill="#334155">artifact={str(item["verifier"]["artifact_complete"]).lower()}</text>'
        )
        boxes.append(
            f'<text x="{x + 16}" y="{y + 120}" font-family="monospace" font-size="11" fill="#334155">{verdict[:22]}</text>'
        )
        if item["iteration"] < len(iterations):
            boxes.append(
                f'<line x1="{x + 170}" y1="{y + 72}" x2="{x + 212}" y2="{y + 72}" stroke="#64748b" stroke-width="3" marker-end="url(#arrow)"/>'
            )
        x += 212

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="930" height="340" viewBox="0 0 930 340">
  <rect width="930" height="340" fill="#f8fafc"/>
  <text x="36" y="42" font-family="monospace" font-size="18" fill="#0f172a">Agentic loop trace (CPU deterministic teaching artifact)</text>
  <text x="36" y="68" font-family="monospace" font-size="12" fill="#475569">Planner proposes, executor records, verifier gates, critic chooses retry / rollback / stop / escalation.</text>
  {''.join(boxes)}
  <text x="54" y="302" font-family="monospace" font-size="12" fill="#7c2d12">Final: benchmark drift triggers human escalation instead of more automated retries.</text>
  <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#64748b"/></marker></defs>
</svg>
"""
    SVG_PATH.write_text(svg, encoding="utf-8")


def run() -> dict[str, object]:
    iterations = build_iterations()
    contract = experiment_contract()
    summary = summarize_iterations(iterations)
    metrics: dict[str, object] = {
        "status": "runnable",
        "loop_id": "agentic-train-eval-v1",
        "simulation": "deterministic_cpu_agentic_training_eval_loop",
        "cpu_safe_simulation": True,
        "experiment_contract": contract,
        "role_sequence": ["planner", "executor", "verifier", "critic"],
        "role_separation": {
            "planner": "sets bounded change set, frozen constraints, and expected evidence before execution",
            "executor": "runs only the approved proposal and records config, seed, logs, metrics, and artifacts",
            "verifier": "checks protocol match, artifact completeness, baseline comparability, and benchmark drift before claims",
            "critic": "uses verifier-approved evidence to choose retry, rollback, stop, or escalation",
        },
        "iterations": iterations,
        "summary": summary,
        "final_decision": {
            "action": "escalate_to_human",
            "reasons": ["benchmark_drift", "long_tail_slice_regression", "low_information_retry_budget"],
            "retry_budget_remaining": contract["retry_budget"] - summary["attempts_used"],
            "claim_boundary": "Do not claim model improvement until benchmark/dataset contract is reviewed.",
        },
        "artifacts": {
            "metrics": str(METRICS_PATH.relative_to(UNIT_ROOT)),
            "iteration_trace": str(TRACE_PATH.relative_to(UNIT_ROOT)),
            "figure": str(SVG_PATH.relative_to(UNIT_ROOT)),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_trace(iterations)
    write_svg(iterations)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
