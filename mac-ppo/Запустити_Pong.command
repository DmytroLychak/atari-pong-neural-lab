#!/bin/bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [ ! -x .venv/bin/python ]; then /bin/bash "$project_dir/Встановити.command"; fi
echo "Відкриваю Pong. Space — пауза; R — нова партія; Esc — вихід."
.venv/bin/python play.py "$@"
