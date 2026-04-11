from __future__ import annotations

import json
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'


RESULT_TO_DECISION = {
    'success stop': 'stop',
    'negative result': 'archive',
    'inconclusive result': 'pause',
    'trust failure': 'escalate',
}

EVIDENCE_ITEMS = [
    {
        'hypothesis_id': 'H1',
        'result_type': 'success stop',
        'primary_signal': 'rollback_drift_delta=-0.12 exceeds required improvement and verifier mismatch stays flat',
        'reopen condition': 'reopen only if production rollback incidents shift to a new failure slice',
    },
    {
        'hypothesis_id': 'H2',
        'result_type': 'negative result',
        'primary_signal': 'effect stayed inside variance band twice under the locked iteration boundary',
        'reopen condition': 'reopen after a new error taxonomy includes explicit root-cause slots',
    },
    {
        'hypothesis_id': 'H3',
        'result_type': 'inconclusive result',
        'primary_signal': 'observed gain is close to threshold but holdout coverage and contamination audit are incomplete',
        'reopen condition': 'resume after private holdout expansion passes contamination review',
    },
    {
        'hypothesis_id': 'H4',
        'result_type': 'trust failure',
        'primary_signal': 'benchmark slice no longer represents production rollback warning incidents',
        'reopen condition': 'restart only after benchmark contract review updates slice taxonomy',
    },
]


def decide(item: dict[str, str]) -> dict[str, object]:
    decision = RESULT_TO_DECISION[item['result_type']]
    if decision == 'stop':
        rationale = 'success stop: the track learned the intended lesson under its evidence standard; stop before scope creep.'
        owner = 'research lead'
    elif decision == 'archive':
        rationale = 'negative result: sufficient evidence rejected the current hypothesis, so preserve it and do not rerun unchanged.'
        owner = 'experiment owner'
    elif decision == 'pause':
        rationale = 'inconclusive result: the claim may be useful, but measurement quality is not strong enough to decide.'
        owner = 'benchmark owner'
    else:
        rationale = 'trust failure: evidence contract is broken, so escalate before spending more iteration budget.'
        owner = 'benchmark/data review group'

    return {
        'hypothesis_id': item['hypothesis_id'],
        'result_type': item['result_type'],
        'decision': decision,
        'primary_signal': item['primary_signal'],
        'rationale': rationale,
        'owner': owner,
        'reopen condition': item['reopen condition'],
    }


def build_report() -> dict[str, object]:
    decision_log = [decide(item) for item in EVIDENCE_ITEMS]
    decision_counts = {name: 0 for name in ['stop', 'pause', 'escalate', 'archive']}
    for item in decision_log:
        decision_counts[item['decision']] += 1

    return {
        'status': 'runnable',
        'framework': 'cpu_deterministic_open_research_ops_sim',
        'deterministic_seed': 20260412,
        'operation_contract': {
            'required_fields': [
                'research scope',
                'north-star question',
                'hypothesis registry',
                'iteration boundary',
                'kill criteria',
                'evidence standard',
                'negative result',
                'inconclusive result',
                'stop / pause / escalate / archive decision',
                'reopen condition',
            ],
            'archive_every_iteration': True,
            'no_external_services': True,
            'cpu_safe': True,
        },
        'decision_by_result_type': RESULT_TO_DECISION,
        'decision_log': decision_log,
        'decision_summary': {
            'decision_counts': decision_counts,
            'key_rule': 'negative result is archived; inconclusive result is paused; trust failure is escalated; success stop closes the iteration.',
        },
        'archive_contract': {
            'must_preserve': [
                'scope memo',
                'hypothesis registry',
                'iteration boundary',
                'kill criteria',
                'evidence standard',
                'negative vs inconclusive classification',
                'decision rationale',
            ],
            'reopen condition': [item['reopen condition'] for item in decision_log],
            'anti_wandering_guard': 'no new experiment may start until the previous decision note is archived',
        },
    }


def run() -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    METRICS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
