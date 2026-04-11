from __future__ import annotations

import json
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
SVG_PATH = ARTIFACT_DIR / 'research_track_map.svg'


HYPOTHESES = [
    {
        'id': 'H1',
        'claim': 'planner brief를 shorter-but-stricter 형식으로 제한하면 long-horizon rollback drift가 줄어든다',
        'mechanism_guess': 'planner가 한 iteration 안에서 바꿀 수 있는 변수를 줄이면 executor retry가 덜 흔들린다',
        'iteration boundary': {
            'changed_variable': 'planner_brief_format',
            'fixed_protocol': ['same benchmark v0.4', 'same tool schema', 'same evaluator rubric'],
            'max_retries': 2,
            'budgeted_runs': 4,
        },
        'evidence standard': {
            'baseline_metric': 'rollback_drift_rate',
            'minimum_effect_size': -0.08,
            'variance_band': 0.03,
            'qualitative_examples_min': 2,
        },
        'kill criteria': [
            'rollback_drift_delta stays inside variance band twice',
            'verifier mismatch increases above baseline',
        ],
        'reopen condition': 'new planner prompt family or larger long-horizon holdout becomes available',
    },
    {
        'id': 'H2',
        'claim': 'critic memory를 긴 transcript 대신 짧은 error taxonomy로 압축하면 retry drift가 줄어든다',
        'mechanism_guess': 'critic state가 짧으면 이전 실패를 더 안정적으로 참조하지만, 과도한 압축은 원인 정보를 잃을 수 있다',
        'iteration boundary': {
            'changed_variable': 'critic_memory_format',
            'fixed_protocol': ['same benchmark v0.4', 'same planner brief', 'same executor temperature'],
            'max_retries': 2,
            'budgeted_runs': 4,
        },
        'evidence standard': {
            'baseline_metric': 'verifier_reopen_rate',
            'minimum_effect_size': -0.05,
            'variance_band': 0.04,
            'qualitative_examples_min': 2,
        },
        'kill criteria': [
            'effect size remains inside variance band across two independent slices',
            'taxonomy loses root-cause notes needed by verifier',
        ],
        'reopen condition': 'new error taxonomy with explicit root-cause slots is drafted',
    },
    {
        'id': 'H3',
        'claim': 'rollback-after-verifier-warning slice를 별도 curriculum으로 만들면 repair success가 오른다',
        'mechanism_guess': 'warning 직후의 recovery examples가 부족해서 agent가 same failure를 반복한다',
        'iteration boundary': {
            'changed_variable': 'slice_specific_training_examples',
            'fixed_protocol': ['same holdout ids', 'same evaluator', 'no benchmark schema changes'],
            'max_retries': 1,
            'budgeted_runs': 3,
        },
        'evidence standard': {
            'baseline_metric': 'repair_success_after_warning',
            'minimum_effect_size': 0.06,
            'variance_band': 0.05,
            'qualitative_examples_min': 3,
        },
        'kill criteria': [
            'dataset audit finds contamination risk',
            'holdout coverage is too thin to support the claim',
        ],
        'reopen condition': 'private holdout expansion passes contamination review',
    },
    {
        'id': 'H4',
        'claim': 'current benchmark의 rollback slice score가 실제 long-horizon reliability를 충분히 대표한다',
        'mechanism_guess': 'benchmark slice가 production-like verifier warning을 충분히 포함한다',
        'iteration boundary': {
            'changed_variable': 'none_observation_only',
            'fixed_protocol': ['benchmark v0.4 audit only', 'no prompt or model change'],
            'max_retries': 0,
            'budgeted_runs': 1,
        },
        'evidence standard': {
            'baseline_metric': 'slice_representativeness_audit',
            'minimum_effect_size': 0.0,
            'variance_band': 0.0,
            'qualitative_examples_min': 4,
        },
        'kill criteria': [
            'protocol mismatch appears between benchmark and production incident log',
            'slice taxonomy cannot explain recent verifier warnings',
        ],
        'reopen condition': 'benchmark contract is revised and linked to incident taxonomy',
    },
]

EVIDENCE_LOG = [
    {
        'hypothesis_id': 'H1',
        'result_type': 'success stop',
        'baseline_comparison': {'baseline': 0.31, 'observed': 0.19, 'delta': -0.12},
        'failure_slice_notes': 'rollback-after-warning slice improved without verifier mismatch increase',
        'negative_result_log': [],
        'inconclusive_reason': None,
    },
    {
        'hypothesis_id': 'H2',
        'result_type': 'negative result',
        'baseline_comparison': {'baseline': 0.22, 'observed': 0.21, 'delta': -0.01},
        'failure_slice_notes': 'taxonomy compression made examples easier to scan but did not change retry behavior',
        'negative_result_log': ['effect stayed inside variance band on planning_drift and tool_reuse slices'],
        'inconclusive_reason': None,
    },
    {
        'hypothesis_id': 'H3',
        'result_type': 'inconclusive result',
        'baseline_comparison': {'baseline': 0.44, 'observed': 0.49, 'delta': 0.05},
        'failure_slice_notes': 'small improvement but holdout examples overlap with training draft annotations',
        'negative_result_log': [],
        'inconclusive_reason': 'measurement trust is not high enough because holdout coverage and contamination audit are incomplete',
    },
    {
        'hypothesis_id': 'H4',
        'result_type': 'trust failure',
        'baseline_comparison': {'baseline': 0.78, 'observed': 0.61, 'delta': -0.17},
        'failure_slice_notes': 'recent production incidents include tool-permission rollback not represented in benchmark v0.4',
        'negative_result_log': [],
        'inconclusive_reason': 'benchmark contract mismatch blocks direct capability interpretation',
    },
]


def build_research_scope() -> dict[str, object]:
    return {
        'research scope': 'agentic long-horizon planning reliability under verifier warnings',
        'north-star question': 'tool-using agent의 장기 계획 안정성을 어떻게 더 재현 가능하게 높일 수 있는가?',
        'this_iteration_focus': [
            'rollback-after-verifier-warning behavior',
            'planner brief and critic memory intervention only',
        ],
        'out_of_scope': [
            'base model pretraining changes',
            'new benchmark schema redesign',
            'external service or live model calls',
        ],
        'fixed_constraints': {
            'benchmark_version': 'frontier-agent-reliability-v0.4',
            'compute_budget': 'CPU-safe deterministic simulation only',
            'review_gate': 'archive every iteration before next run',
        },
    }


def build_svg() -> None:
    decision_colors = {
        'success stop': '#86efac',
        'negative result': '#fca5a5',
        'inconclusive result': '#fde68a',
        'trust failure': '#c4b5fd',
    }
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="360" viewBox="0 0 920 360">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="28" y="38" font-family="monospace" font-size="19" fill="#0f172a">Open-ended research track map</text>',
        '<text x="28" y="62" font-family="monospace" font-size="12" fill="#475569">research scope → hypothesis registry → evidence standard → stop/pause/escalate/archive</text>',
    ]
    for idx, (hypothesis, evidence) in enumerate(zip(HYPOTHESES, EVIDENCE_LOG)):
        x = 42 + idx * 215
        color = decision_colors[evidence['result_type']]
        parts.extend([
            f'<rect x="{x}" y="96" width="176" height="168" rx="14" fill="{color}" stroke="#334155"/>',
            f'<text x="{x + 16}" y="126" font-family="monospace" font-size="18" fill="#0f172a">{hypothesis["id"]}</text>',
            f'<text x="{x + 16}" y="150" font-family="monospace" font-size="12" fill="#0f172a">hypothesis registry</text>',
            f'<text x="{x + 16}" y="174" font-family="monospace" font-size="12" fill="#0f172a">{evidence["result_type"]}</text>',
            f'<text x="{x + 16}" y="202" font-family="monospace" font-size="11" fill="#334155">boundary+kill criteria</text>',
            f'<text x="{x + 16}" y="226" font-family="monospace" font-size="11" fill="#334155">evidence standard locked</text>',
        ])
    parts.extend([
        '<text x="42" y="314" font-family="monospace" font-size="13" fill="#334155">Negative is not inconclusive: archive vs pause are different operating decisions.</text>',
        '</svg>',
    ])
    SVG_PATH.write_text('\n'.join(parts), encoding='utf-8')


def build_metrics() -> dict[str, object]:
    return {
        'status': 'runnable',
        'cpu_safe_simulation': True,
        'deterministic_seed': 20260412,
        'track_id': 'frontier-open-ended-research-v1',
        'research_scope': build_research_scope(),
        'hypothesis_registry': {
            'type': 'hypothesis registry',
            'hypotheses': HYPOTHESES,
        },
        'evidence_log': EVIDENCE_LOG,
        'iteration_boundary_summary': {
            'changed_variables': [h['iteration boundary']['changed_variable'] for h in HYPOTHESES],
            'fixed_protocols_checked': sorted({item for h in HYPOTHESES for item in h['iteration boundary']['fixed_protocol']}),
            'total_budgeted_runs': sum(h['iteration boundary']['budgeted_runs'] for h in HYPOTHESES),
        },
        'evidence_standard': {
            'required_fields': [
                'baseline_comparison',
                'failure_slice_notes',
                'negative_result_log',
                'inconclusive_reason',
                'reopen condition',
            ],
            'negative_vs_inconclusive_rule': {
                'negative result': 'evidence is sufficient and the current hypothesis is not supported',
                'inconclusive result': 'measurement or boundary quality is insufficient, so the claim cannot be judged yet',
            },
        },
        'artifacts': {
            'metrics': str(METRICS_PATH.relative_to(UNIT_ROOT)),
            'research_track_map_svg': str(SVG_PATH.relative_to(UNIT_ROOT)),
        },
    }


def run() -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    build_svg()
    metrics = build_metrics()
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    return metrics


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
