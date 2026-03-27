#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from experiment import run_stage
from _runtime import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run 04_large_scale_tabular from its stage-local experiment module.')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(json.dumps(run_stage(device), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
