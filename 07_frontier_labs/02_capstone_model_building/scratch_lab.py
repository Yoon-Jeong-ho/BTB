"""Deterministic capstone model-building contract lab.

This script does not train a model or download data. It turns one small
frontier-lab capstone idea into a reproducible project contract so students can
inspect scope, non-goals, dataset/model/eval contracts, milestones, acceptance
gates, risks, and failure-analysis columns before any expensive run begins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


UNIT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_DIR / "artifacts" / "scratch-manual"
CONTRACT_PATH = ARTIFACT_DIR / "capstone_contract.json"
SVG_PATH = ARTIFACT_DIR / "milestone_gates.svg"


def build_contract() -> dict[str, Any]:
    """Return the stable capstone contract used by the lesson."""

    milestones = [
        {
            "id": "M0",
            "goal": "problem, non-goals, dataset split, baseline, and metric are frozen",
            "acceptance_gate": "scope_freeze_signed_off",
            "required_artifacts": [
                "problem_statement.md",
                "non_goals.md",
                "dataset_split_manifest.json",
                "baseline_card.md",
            ],
        },
        {
            "id": "M1",
            "goal": "minimal reproducible train/eval pipeline emits comparable baseline metrics",
            "acceptance_gate": "baseline_reproduces_on_fixed_split",
            "required_artifacts": [
                "train_config.yaml",
                "eval_metrics.json",
                "seed_and_runtime_log.json",
            ],
        },
        {
            "id": "M2",
            "goal": "one model improvement path is compared with the baseline under the same protocol",
            "acceptance_gate": "target_delta_exceeds_minimum_with_slice_review",
            "required_artifacts": [
                "candidate_config.yaml",
                "ablation_table.md",
                "slice_metrics.json",
            ],
        },
        {
            "id": "M3",
            "goal": "failure analysis, risk update, and final report make next steps explicit",
            "acceptance_gate": "report_ready_with_failure_table",
            "required_artifacts": [
                "failure_analysis_table.md",
                "risk_register_update.md",
                "final_capstone_report.md",
            ],
        },
    ]

    risk_register = [
        {
            "id": "dataset_leakage",
            "severity": "high",
            "signal": "near-duplicate products cross train/test boundaries",
            "mitigation": "near_duplicate_group_holdout and split manifest review before M1",
        },
        {
            "id": "baseline_weakness",
            "severity": "medium",
            "signal": "lexical baseline misses synonym-heavy queries but is not documented",
            "mitigation": "baseline card states known blind spots and minimum fair comparison",
        },
        {
            "id": "scope_creep",
            "severity": "high",
            "signal": "serving latency, new data collection, and personalization enter the plan",
            "mitigation": "non-goals block expansion until final report next-steps section",
        },
        {
            "id": "metric_gaming",
            "severity": "medium",
            "signal": "Recall@10 improves while brand and OCR slices regress",
            "mitigation": "slice review gate must pass before claiming M2 success",
        },
        {
            "id": "budget_mismatch",
            "severity": "medium",
            "signal": "candidate requires GPU training outside the course runtime envelope",
            "mitigation": "CPU toy embedding table and cached deterministic metrics are canonical",
        },
        {
            "id": "report_drift",
            "severity": "low",
            "signal": "experiment notes no longer map to final report sections",
            "mitigation": "analysis outline is fixed before experiments start",
        },
    ]

    return {
        "status": "runnable",
        "cpu_safe_simulation": True,
        "contract_type": "capstone_model_building_contract",
        "project_id": "korean_catalog_retrieval_capstone",
        "problem_statement": (
            "한국어 상품 검색에서 text query와 image_caption을 입력으로 받아 "
            "lexical baseline 대비 Recall@10을 5pt 이상 개선한다."
        ),
        "non_goals": [
            "real-time serving optimization is out of scope",
            "external dataset collection or network download is out of scope",
            "personalization, payment ranking, and production A/B testing are out of scope",
            "large GPU fine-tuning is optional evidence, not the canonical unit path",
        ],
        "dataset_contract": {
            "source": "synthetic_korean_catalog_seed_v1",
            "split": {"train": 1200, "valid": 200, "test": 200},
            "schema_fields": [
                "query",
                "product_title",
                "image_caption",
                "category",
                "brand",
                "relevance_label",
                "near_duplicate_group",
            ],
            "label_quality_checks": [
                "two_positive_examples_per_query_when_available",
                "manual_spot_check_20_queries_per_category",
                "ambiguous_label_bucket_is_reported_not_hidden",
            ],
            "leakage_controls": [
                "near_duplicate_group_holdout",
                "brand_category_stratified_split",
                "test_queries_never_used_for_candidate_mining",
            ],
        },
        "model_contract": {
            "baseline": "lexical_title_baseline",
            "candidates": ["tiny_dual_encoder", "caption_augmented_dual_encoder", "lightweight_reranker"],
            "frozen_constraints": [
                "same split",
                "same Recall@10 evaluator",
                "same query set",
                "same failure buckets",
            ],
            "runtime_budget": "cpu toy embedding table",
            "comparison_note": "model size is less important than protocol-matched baseline comparison",
        },
        "eval_contract": {
            "primary_metric": "Recall@10",
            "baseline_score": 0.42,
            "target_score": 0.49,
            "minimum_delta": 0.05,
            "secondary_metrics": ["MRR", "category_slice_recall", "brand_slice_recall"],
            "qualitative_buckets": [
                "brand_mismatch",
                "fine_grained_visual_confusion",
                "ocr_text_failure",
                "category_boundary_error",
            ],
            "acceptance_note": "target_score must clear baseline_score by at least minimum_delta and no high-risk slice may regress",
        },
        "milestones": milestones,
        "acceptance_gates": [milestone["acceptance_gate"] for milestone in milestones],
        "risk_register": risk_register,
        "failure_analysis_outline": {
            "columns": ["slice", "failure_bucket", "evidence", "hypothesis", "next_action"],
            "required_slices": ["category", "brand", "query_length", "image_caption_presence"],
            "report_rule": "every failed or regressed slice receives one hypothesis and one next action",
        },
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
    }


def write_svg(contract: dict[str, Any]) -> None:
    """Write a tiny deterministic SVG showing milestone gates."""

    cards = []
    x = 20
    for milestone in contract["milestones"]:
        cards.append(
            f'<rect x="{x}" y="30" width="150" height="70" rx="10" fill="#eef6ff" stroke="#245" />'
        )
        cards.append(
            f'<text x="{x + 15}" y="60" font-size="16" font-family="monospace">{milestone["id"]}</text>'
        )
        cards.append(
            f'<text x="{x + 15}" y="84" font-size="10" font-family="sans-serif">{milestone["acceptance_gate"]}</text>'
        )
        x += 170
    svg = "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="140" viewBox="0 0 720 140">',
            '<title>Capstone milestone gates</title>',
            '<text x="20" y="20" font-size="14" font-family="sans-serif">Capstone milestone gates</text>',
            *cards,
            '</svg>',
            '',
        ]
    )
    SVG_PATH.write_text(svg, encoding="utf-8")


def main() -> None:
    contract = build_contract()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_svg(contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
