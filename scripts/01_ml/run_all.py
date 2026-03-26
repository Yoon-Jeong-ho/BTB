#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "01_ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from common import run_stage_01, run_stage_02, run_stage_03, run_stage_04, set_seed
from reporting import localize_ml_reports, write_track_index_ko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full 01_ml track.")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage_results = [
        run_stage_01(device),
        run_stage_02(device),
        run_stage_03(device),
        run_stage_04(device),
    ]
    localize_ml_reports(stage_results)
    write_track_index_ko(stage_results)
    print(json.dumps(stage_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
