#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
conda env create -f 01_ml/env/environment.yml || conda env update -n btb-01-ml -f 01_ml/env/environment.yml
conda list --explicit -n btb-01-ml > 01_ml/env/conda-linux-64.lock.txt
