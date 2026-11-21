#!/bin/bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [ ! -x .venv/bin/python ]; then /bin/bash "$project_dir/Встановити.command"; fi
run_name="live_$(date +%Y%m%d_%H%M%S)_$$"
echo "Починаю справжнє навчання PPO з випадкових ваг."
echo "Space — пауза; N — наочно/швидко; S — зупинити й зберегти."
echo "Журнал і нові ваги: $project_dir/runs/$run_name"
.venv/bin/python train.py --live --steps 2000000 --envs 6 --out "runs/$run_name" --screenshot-dir "runs/$run_name/screenshots" "$@"
