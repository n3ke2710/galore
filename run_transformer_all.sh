#!/usr/bin/env bash
set -euo pipefail

python3 transformer_experiment.py --opt adamw "$@"
python3 transformer_experiment.py --opt galore "$@"
python3 transformer_experiment.py --opt prox "$@"
