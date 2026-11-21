#!/bin/bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [ ! -x .venv/bin/python ]; then /bin/bash "$project_dir/Встановити.command"; fi
run_name="new_training_$(date +%Y%m%d_%H%M%S)"
echo "Нове навчання. Готова демонстраційна модель збережеться без змін."
.venv/bin/python train.py --steps 2000000 --envs 6 --out "runs/$run_name"
echo "Збережено у runs/$run_name/latest.zip"
