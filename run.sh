#!/usr/bin/env bash
# Full pipeline: download -> train -> evaluate -> figures. Every step is deterministic given config.yaml.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data results figures
python src/data.py        # ~15 min the first time (NOAA ERDDAP), cached afterwards
python src/train.py       # ~35 min on an Apple GPU, longer on CPU
python src/evaluate.py
python src/figures.py
