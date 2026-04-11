from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from statistics import mean, median


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
SVG_PATH = ARTIFACT_DIR / 'profiling_timeline.svg'

TOKENS_PER_STEP = 4096
WORLD_SIZE = 4
STEP_RECORDS = [
    {'step': 120, 'data_wait_ms': 11, 'compute_ms': 55, 'communication_wait_ms': 16, 'checkpoint_io_ms': 5, 'misc_sync_ms': 3, 'rank': 0},
    {'step': 121, 'data_wait_ms': 12, 'compute_ms': 55, 'communication_wait_ms': 18, 'checkpoint_io_ms': 5, 'misc_sync_ms': 3, 'rank': 1},
    {'step': 122, 'data_wait_ms': 12, 'compute_ms': 56, 'communication_wait_ms': 19, 'checkpoint_io_ms': 6, 'misc_sync_ms': 3, 'rank': 2},
    {'step': 123, 'data_wait_ms': 11, 'compute_ms': 55, 'communication_wait_ms': 21, 'checkpoint_io_ms': 6, 'misc_sync_ms': 3, 'rank': 3},
    {'step': 124, 'data_wait_ms': 12, 'compute_ms': 56, 'communication_wait_ms': 31, 'checkpoint_io_ms': 8, 'misc_sync_ms': 4, 'rank': 2},
    {'step': 125, 'data_wait_ms': 13, 'compute_ms': 56, 'communication_wait_ms': 39, 'checkpoint_io_ms': 8, 'misc_sync_ms': 4, 'rank': 2},
    {'step': 126, 'data_wait_ms': 14, 'compute_ms': 57, 'communication_wait_ms': 43, 'checkpoint_io_ms': 10, 'misc_sync_ms': 4, 'rank': 2},
    {'step': 127, 'data_wait_ms': 14, 'compute_ms': 57, 'communication_wait_ms': 47, 'checkpoint_io_ms': 13, 'misc_sync_ms': 4, 'rank': 2},
]
MEMORY_SERIES = [
    {'step': 120, 'allocated_mb': 10820, 'reserved_mb': 11840, 'phase': 'steady_train'},
    {'step': 121, 'allocated_mb': 10848, 'reserved_mb': 11872, 'phase': 'steady_train'},
    {'step': 122, 'allocated_mb': 10910, 'reserved_mb': 11968, 'phase': 'steady_train'},
    {'step': 123, 'allocated_mb': 11024, 'reserved_mb': 12160, 'phase': 'steady_train'},
    {'step': 124, 'allocated_mb': 11240, 'reserved_mb': 12672, 'phase': 'eval_boundary'},
    {'step': 125, 'allocated_mb': 11312, 'reserved_mb': 12928, 'phase': 'checkpoint_flush'},
    {'step': 126, 'allocated_mb': 11136, 'reserved_mb': 13056, 'phase': 'post_checkpoint'},
    {'step': 127, 'allocated_mb': 11080, 'reserved_mb': 13056, 'phase': 'post_checkpoint'},
]
HEARTBEATS = [
    {'rank': 0, 'last_heartbeat_lag_ms': 15, 'last_seen_step': 127},
    {'rank': 1, 'last_heartbeat_lag_ms': 16, 'last_seen_step': 127},
    {'rank': 2, 'last_heartbeat_lag_ms': 280, 'last_seen_step': 126},
    {'rank': 3, 'last_heartbeat_lag_ms': 17, 'last_seen_step': 127},
]


def rounded(value: float) -> float:
    return round(value, 6)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = max(0, ceil((p / 100.0) * len(ordered)) - 1)
    return float(ordered[index])


def step_total(record: dict[str, int]) -> int:
    return sum(record[key] for key in ['data_wait_ms', 'compute_ms', 'communication_wait_ms', 'checkpoint_io_ms', 'misc_sync_ms'])


def build_time_breakdown() -> dict[str, float]:
    totals = {
        'data_wait': sum(record['data_wait_ms'] for record in STEP_RECORDS),
        'forward_backward_compute': sum(record['compute_ms'] for record in STEP_RECORDS),
        'communication_wait': sum(record['communication_wait_ms'] for record in STEP_RECORDS),
        'checkpoint_io': sum(record['checkpoint_io_ms'] for record in STEP_RECORDS),
        'misc_sync': sum(record['misc_sync_ms'] for record in STEP_RECORDS),
    }
    total_ms = sum(totals.values())
    percentages = {name: rounded(value / total_ms * 100.0) for name, value in totals.items()}
    drift = rounded(100.0 - sum(percentages.values()))
    percentages['misc_sync'] = rounded(percentages['misc_sync'] + drift)
    return percentages


def build_svg(step_times: list[int]) -> None:
    width, height = 820, 320
    left, top = 72, 72
    cell_w = 78
    max_time = max(step_times)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="24" y="36" font-family="monospace" font-size="18" fill="#0f172a">Profiling timeline: step time, communication wait, checkpoint boundary</text>',
        '<text x="24" y="58" font-family="monospace" font-size="12" fill="#475569">CPU-safe deterministic simulation; no GPU profiler is required.</text>',
    ]
    for index, (record, total_ms) in enumerate(zip(STEP_RECORDS, step_times)):
        x = left + index * cell_w
        bar_h = total_ms / max_time * 170
        y = top + 180 - bar_h
        comm_h = record['communication_wait_ms'] / max_time * 170
        parts.extend([
            f'<rect x="{x}" y="{y:.2f}" width="46" height="{bar_h:.2f}" fill="#bae6fd" stroke="#0369a1"/>',
            f'<rect x="{x}" y="{top + 180 - comm_h:.2f}" width="46" height="{comm_h:.2f}" fill="#fb923c" opacity="0.82"/>',
            f'<text x="{x - 4}" y="{top + 198}" font-family="monospace" font-size="11">s{record["step"]}</text>',
            f'<text x="{x - 2}" y="{y - 7:.2f}" font-family="monospace" font-size="11">{total_ms}ms</text>',
        ])
    parts.extend([
        '<rect x="584" y="252" width="18" height="12" fill="#bae6fd" stroke="#0369a1"/>',
        '<text x="610" y="263" font-family="monospace" font-size="12">total step time</text>',
        '<rect x="584" y="274" width="18" height="12" fill="#fb923c" opacity="0.82"/>',
        '<text x="610" y="285" font-family="monospace" font-size="12">communication wait share</text>',
        '</svg>',
    ])
    SVG_PATH.write_text('\n'.join(parts), encoding='utf-8')


def compute_metrics() -> dict[str, object]:
    step_times = [step_total(record) for record in STEP_RECORDS]
    baseline_step_ms = mean(step_times[:4])
    observed_step_ms = mean(step_times[-4:])
    peak_allocated = max(item['allocated_mb'] for item in MEMORY_SERIES)
    peak_reserved = max(item['reserved_mb'] for item in MEMORY_SERIES)
    peak_step = max(MEMORY_SERIES, key=lambda item: item['reserved_mb'])

    return {
        'status': 'runnable',
        'cpu_safe_simulation': True,
        'deterministic_seed': 20260412,
        'profile_window': {
            'steps': len(STEP_RECORDS),
            'world_size': WORLD_SIZE,
            'tokens_per_step': TOKENS_PER_STEP,
            'start_step': STEP_RECORDS[0]['step'],
            'end_step': STEP_RECORDS[-1]['step'],
        },
        'step_time_ms': {
            'p50': rounded(median(step_times)),
            'p95': rounded(percentile([float(value) for value in step_times], 95)),
            'mean_first_half': rounded(baseline_step_ms),
            'mean_second_half': rounded(observed_step_ms),
            'jitter_ratio_second_over_first': rounded(observed_step_ms / baseline_step_ms),
        },
        'throughput': {
            'baseline_tokens_per_sec': rounded(TOKENS_PER_STEP / (baseline_step_ms / 1000.0)),
            'observed_tokens_per_sec': rounded(TOKENS_PER_STEP / (observed_step_ms / 1000.0)),
            'drop_pct': rounded((1.0 - baseline_step_ms / observed_step_ms) * 100.0),
        },
        'time_breakdown_pct': build_time_breakdown(),
        'dominant_bottleneck': 'communication_wait_due_to_rank_2_heartbeat_lag',
        'memory_snapshot': {
            'peak_allocated_mb': peak_allocated,
            'peak_reserved_mb': peak_reserved,
            'peak_step': peak_step['step'],
            'peak_phase': peak_step['phase'],
            'reserved_minus_allocated_mb': peak_reserved - peak_allocated,
            'fragmentation_hint': 'reserved remains high after checkpoint_flush while allocated has fallen',
        },
        'per_rank_heartbeat': HEARTBEATS,
        'checkpoint_freshness_minutes': 42,
        'alerts': [
            'throughput_drop_gt_20pct',
            'step_time_p95_exceeds_p50_by_30pct',
            'rank_2_heartbeat_lag',
            'checkpoint_age_exceeds_target',
            'reserved_memory_stays_high_after_checkpoint',
        ],
        'diagnostic_next_steps': [
            'open a profiler window around steps 124-127',
            'compare rank_2 dataloader and communication timestamps',
            'verify last_good_checkpoint manifest before retry',
        ],
        'timeline_head': [
            {'step': record['step'], 'total_ms': step_total(record), 'communication_wait_ms': record['communication_wait_ms']}
            for record in STEP_RECORDS[:4]
        ],
        'timeline_tail': [
            {'step': record['step'], 'total_ms': step_total(record), 'communication_wait_ms': record['communication_wait_ms']}
            for record in STEP_RECORDS[-4:]
        ],
        'artifacts': {
            'metrics': str(METRICS_PATH.relative_to(UNIT_ROOT)),
            'profiling_timeline_svg': str(SVG_PATH.relative_to(UNIT_ROOT)),
        },
    }


def run() -> dict[str, object]:
    metrics = compute_metrics()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    build_svg([step_total(record) for record in STEP_RECORDS])
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    return metrics


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
