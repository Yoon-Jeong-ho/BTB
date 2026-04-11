from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from xml.sax.saxutils import escape


UNIT_ROOT = Path(__file__).resolve().parent
OUT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = OUT_DIR / 'metrics.json'
FIGURE_PATH = OUT_DIR / 'paper_reproduction_matrix.svg'

CLAIMS = [
    {
        'claim_id': 'C1_adapter_efficiency',
        'claim': 'Adapter tuning preserves most full fine-tune accuracy with lower compute.',
        'evidence_type': 'same-protocol accuracy delta plus compute budget note',
        'acceptance_rule': 'reproduced method beats reproduced baseline and stays within 1.0pt of reported accuracy',
        'observed_signal': 'accuracy +0.4pt over baseline, -0.5pt vs reported',
        'decision': 'direction_reproduced_with_small_gap',
    },
    {
        'claim_id': 'C2_regularization_ablation',
        'claim': 'Removing the regularization term weakens the reported improvement.',
        'evidence_type': 'ablation metric drop under identical split and evaluator',
        'acceptance_rule': 'ablation drop is positive and larger than one seed standard deviation',
        'observed_signal': 'ablation drop 1.5pt vs seed std 0.2pt',
        'decision': 'supported_in_reduced_scope',
    },
    {
        'claim_id': 'C3_budget_claim',
        'claim': 'The reduced method reaches the useful trend under a strict playground budget.',
        'evidence_type': 'runtime and artifact completeness rather than absolute paper-scale convergence',
        'acceptance_rule': 'scope is explicitly reduced and all required run artifacts are present',
        'observed_signal': 'CPU deterministic proxy plus complete artifact checklist',
        'decision': 'scope_limited_not_full_reproduction',
    },
]

SEED_RUNS = [
    {'seed': 11, 'accuracy': 0.844, 'macro_f1': 0.812},
    {'seed': 17, 'accuracy': 0.846, 'macro_f1': 0.815},
    {'seed': 23, 'accuracy': 0.848, 'macro_f1': 0.817},
]


Comparison = dict[str, object]


def rounded(value: float) -> float:
    return round(value, 6)


def comparison_layers() -> dict[str, Comparison]:
    comparisons: dict[str, Comparison] = {
        'C1_adapter_efficiency': {
            'metric': 'accuracy',
            'baseline': {'name': 'full_finetune_small_reproduced', 'accuracy': 0.842, 'protocol': 'same split/evaluator'},
            'reported': {'name': 'paper_adapter_table2', 'accuracy': 0.851, 'seed_count': 3},
            'reproduced': {'name': 'adapter_proxy_reproduced', 'accuracy': 0.846, 'seed_count': 3},
        },
        'C2_regularization_ablation': {
            'metric': 'macro_f1',
            'baseline': {'name': 'adapter_without_regularizer', 'macro_f1': 0.799, 'protocol': 'same split/evaluator'},
            'reported': {'name': 'paper_regularized_adapter', 'macro_f1': 0.818, 'seed_count': 3},
            'reproduced': {'name': 'regularized_adapter_proxy', 'macro_f1': 0.814, 'seed_count': 3},
        },
        'C3_budget_claim': {
            'metric': 'wall_clock_minutes',
            'baseline': {'name': 'full_finetune_proxy', 'wall_clock_minutes': 44.0, 'protocol': 'fixed CPU simulation budget'},
            'reported': {'name': 'paper_budget_claim_normalized', 'wall_clock_minutes': 30.0, 'note': 'reported as relative compute claim'},
            'reproduced': {'name': 'adapter_proxy_budget', 'wall_clock_minutes': 28.0, 'seed_count': 1},
        },
    }
    for item in comparisons.values():
        metric = str(item['metric'])
        baseline_value = float(dict(item['baseline'])[metric])
        reported_value = float(dict(item['reported'])[metric])
        reproduced_value = float(dict(item['reproduced'])[metric])
        item['delta_vs_baseline'] = rounded(reproduced_value - baseline_value)
        item['delta_vs_reported'] = rounded(reproduced_value - reported_value)
        item['within_same_protocol'] = True
    return comparisons


def artifact_hygiene() -> dict[str, object]:
    required = [
        'scope_boundary',
        'claim_evidence_matrix',
        'baseline_reported_reproduced_table',
        'seed_variance_summary',
        'mismatch_hypotheses',
        'artifact_manifest',
    ]
    present = {name: True for name in required}
    return {
        'required_artifacts': required,
        'present': present,
        'missing_required_artifacts': [name for name, ok in present.items() if not ok],
        'ready_for_handoff': all(present.values()),
        'hygiene_rule': 'record what was reduced, what was measured, and what cannot be claimed',
    }


def render_svg(matrix: list[dict[str, str]], comparisons: dict[str, Comparison]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 980, 430
    rows: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="28" y="42" font-family="monospace" font-size="20" fill="#0f172a">Claim/evidence reproduction matrix</text>',
        '<text x="28" y="68" font-family="monospace" font-size="12" fill="#475569">CPU-safe deterministic toy reproduction; no network or paper download.</text>',
    ]
    y = 112
    colors = {
        'direction_reproduced_with_small_gap': '#7dd3fc',
        'supported_in_reduced_scope': '#86efac',
        'scope_limited_not_full_reproduction': '#fde68a',
    }
    for row in matrix:
        decision = row['decision']
        rows.extend([
            f'<rect x="32" y="{y - 24}" width="916" height="72" rx="10" fill="{colors[decision]}" opacity="0.35" stroke="#94a3b8"/>',
            f'<text x="52" y="{y}" font-family="monospace" font-size="14" fill="#0f172a">{escape(row["claim_id"])}</text>',
            f'<text x="52" y="{y + 22}" font-family="monospace" font-size="12" fill="#334155">evidence: {escape(row["evidence_type"])}</text>',
            f'<text x="52" y="{y + 42}" font-family="monospace" font-size="12" fill="#334155">decision: {escape(decision)}</text>',
        ])
        y += 88
    c1 = comparisons['C1_adapter_efficiency']
    rows.extend([
        '<line x1="32" y1="374" x2="948" y2="374" stroke="#cbd5e1"/>',
        '<text x="52" y="402" font-family="monospace" font-size="13" fill="#0f172a">C1 baseline/reported/reproduced accuracy: '
        f'{dict(c1["baseline"])["accuracy"]:.3f} / {dict(c1["reported"])["accuracy"]:.3f} / {dict(c1["reproduced"])["accuracy"]:.3f}</text>',
        '</svg>',
    ])
    FIGURE_PATH.write_text('\n'.join(rows), encoding='utf-8')


def build_metrics() -> dict[str, object]:
    comparisons = comparison_layers()
    accuracy_values = [row['accuracy'] for row in SEED_RUNS]
    matrix = [
        {
            'claim_id': row['claim_id'],
            'claim': row['claim'],
            'evidence_type': row['evidence_type'],
            'acceptance_rule': row['acceptance_rule'],
            'observed_signal': row['observed_signal'],
            'decision': row['decision'],
        }
        for row in CLAIMS
    ]
    render_svg(matrix, comparisons)
    return {
        'status': 'runnable',
        'mode': 'claim_level_reproduction_playground',
        'cpu_safe': True,
        'deterministic_seed': 20260412,
        'paper_stub': {
            'paper_id': 'offline_adapter_efficiency_example_2024',
            'source_policy': 'toy paper card embedded in script; no network or real paper download',
            'citation_note': 'educational proxy for claim-level reproduction mechanics',
        },
        'scope_control': {
            'principle': 'scope control: reduce compute and claim breadth together',
            'claim_scope': 'reduced_claim',
            'dataset_scope': 'classification_proxy_slice',
            'excluded_scope': ['full paper benchmark suite', 'real training cluster', 'external paper/code download'],
            'allowed_claim': 'directional trend under same-protocol toy proxy',
            'not_allowed_claim': 'absolute paper-scale reproduction',
        },
        'claim_evidence_columns': ['claim_id', 'claim', 'evidence_type', 'acceptance_rule', 'observed_signal', 'decision'],
        'claim_evidence_matrix': matrix,
        'comparisons': comparisons,
        'seed_runs': SEED_RUNS,
        'variance_summary': {
            'seed_count': len(SEED_RUNS),
            'accuracy_mean': rounded(mean(accuracy_values)),
            'accuracy_std': rounded(pstdev(accuracy_values)),
            'variance_note': 'reported margin and seed spread must be compared before overclaiming',
        },
        'mismatch_hypotheses': [
            {
                'hypothesis_id': 'preprocessing_alignment',
                'evidence': 'absolute reproduced accuracy is lower than reported while same-protocol delta remains positive',
                'next_check': 'compare normalization/token filtering and evaluator post-processing',
            },
            {
                'hypothesis_id': 'seed_variance',
                'evidence': 'seed std is small but non-zero relative to the observed gap',
                'next_check': 'run more seeds before calling the gap meaningful',
            },
            {
                'hypothesis_id': 'budget_mismatch',
                'evidence': 'proxy budget is intentionally reduced from the full paper setting',
                'next_check': 'separate trend claim from absolute convergence claim',
            },
        ],
        'artifact_hygiene': artifact_hygiene(),
        'artifacts': {
            'metrics': str(METRICS_PATH.relative_to(UNIT_ROOT)),
            'figure': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
        },
    }


def run() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics()
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    return metrics


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
