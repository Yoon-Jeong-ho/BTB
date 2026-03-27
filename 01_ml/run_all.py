#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from track_index import write_results_index

TRACK_ROOT = Path(__file__).resolve().parent
STAGE_RUNNERS = [
    TRACK_ROOT / '01_tabular_classification' / 'run_stage.py',
    TRACK_ROOT / '02_tabular_regression' / 'run_stage.py',
    TRACK_ROOT / '03_model_selection_and_interpretation' / 'run_stage.py',
    TRACK_ROOT / '04_large_scale_tabular' / 'run_stage.py',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the full 01_ml track from the track-local entrypoint.')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    results = []
    for runner in STAGE_RUNNERS:
        completed = subprocess.run(
            [sys.executable, str(runner), '--gpu', str(args.gpu)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        payload = completed.stdout.strip()
        if payload:
            results.append(json.loads(payload))
    write_results_index(results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
